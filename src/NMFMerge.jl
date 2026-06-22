module NMFMerge

using LinearAlgebra: LinearAlgebra, norm
using DataStructures: DataStructures, PriorityQueue
using NMF: NMF, nnmf
using GsvdInitialization: GsvdInitialization, gsvdrecover
using TSVD: TSVD, tsvd

export nmfmerge,
       colnormalize,
       merge_pq,
       merge_replay

@static if VERSION >= v"1.11"
    # `public` is a parse error before 1.11, so build the declaration as an
    # `Expr` rather than writing it literally.
    eval(Expr(:public, :ssdpenalty))
end

"""
    result = nmfmerge([queuepenalty], X, ncomponents; tol_final=1e-4, tol_intermediate=sqrt(tol_final), W0=nothing, H0=nothing, kwargs...)

Performs "NMF-Merge" on data matrix `X`.

Arguments:

-`queuepenalty`: a function of the form `f(E, h1sq, h2sq)` that computes the penalty for merging two components, where `E` is the the merge error described in the paper, default: [`ssdpenalty`](@ref) (`f(E, h1sq, h2sq) = E`). h1sq and h2sq are the squared norms of the corresponding rows in H.

- `X::AbstractMatrix`: the data matrix to be factorized

- `ncomponents::Pair{Int,Int}`: in the form of `n1 => n2`, merging from `n1` components to `n2`components,
  where `n1` is the number of components for overcomplete NMF, and `n2` is the number of components for the final NMF.
  We require `n1 >= n2`.

Alternatively, `ncomponents` can be an integer denoting the final number of components. In this case, `nmfmerge`
defaults to an approximate 20% component excess before merging.


Keyword arguments:

- `tol_final`: The tolerance of final NMF

- `tol_intermediate`: The tolerence of initial and overcomplete NMF

`W0`, `H0`: initialization for the initial NMF. If at least one of `W0` and `H0` is `nothing`, NNDSVD is used for initialization.


Other keywords arguments are passed to `NMF.nnmf`.
"""
function nmfmerge(queuepenalty, X, ncomponents::Pair{Int,Int}; tol_final=1e-4, tol_intermediate=sqrt(tol_final), W0=nothing, H0=nothing, kwargs...)
    n1, n2 = ncomponents
    f = tsvd(X, n2)
    Un, Sn, Vn = f
    if W0 === nothing || H0 === nothing
        W0, H0 = NMF.nndsvd(X, n2, initdata=(U = Un, S = Sn, V = Vn))
    end
    result_initial = nnmf(X, n2; kwargs..., init=:custom, tol=tol_intermediate, W0=copy(W0), H0=copy(H0))
    W_initial, H_initial = result_initial.W, result_initial.H
    kadd = n1 - n2
    kadd >= 0 || throw(ArgumentError("Cannot merge to more components than original"))
    if kadd == 0
        # No overcomplete components to add and nothing to merge; refine the
        # initial factorization to the final tolerance.
        return nnmf(X, n2; kwargs..., init=:custom, tol=tol_final, W0=W_initial, H0=H_initial)
    end
    W_over_init, H_over_init, _ = gsvdrecover(X, W_initial, H_initial, kadd, f)
    result_over = nnmf(X, n1; kwargs..., init=:custom, tol=tol_intermediate, W0=W_over_init, H0=H_over_init)
    W_over, H_over = result_over.W, result_over.H
    W_over_normed, H_over_normed = colnormalize(W_over, H_over)
    Wmerge, Hmerge, _ = merge_pq(queuepenalty, W_over_normed, H_over_normed; nstop=n2)
    result_renmf = nnmf(X, n2; kwargs..., init=:custom, tol=tol_final, W0=Wmerge, H0=Hmerge)
    return result_renmf
end
nmfmerge(queuepenalty, X, ncomponents::Pair{<:Integer,<:Integer}; kwargs...) = nmfmerge(queuepenalty, X, Int(ncomponents.first) => Int(ncomponents.second); kwargs...)
nmfmerge(queuepenalty, X, ncomponents::Integer; kwargs...) = nmfmerge(queuepenalty, X, ncomponents+max(1, round(Int, 0.2*ncomponents)) => Int(ncomponents); kwargs...)
nmfmerge(X, ncomponents::Pair{<:Integer,<:Integer}; kwargs...) = nmfmerge(ssdpenalty, X, ncomponents; kwargs...)
nmfmerge(X, ncomponents::Integer; kwargs...) = nmfmerge(ssdpenalty, X, ncomponents; kwargs...)

function colnormalize!(W, H, p::Real=2)
    check_component_axis(W, H)
    nonzerocolids = Int[]
    for (j, w) in pairs(eachcol(W))
        normw = norm(w, p)
        if !iszero(normw)
            W[:, j] = w/normw
            H[j, :] = H[j, :]*normw
            push!(nonzerocolids, j)
        end
    end
    W, H = W[:, nonzerocolids], H[nonzerocolids, :]
    return W, H
end

"""
    Wnormalized, Hnormalized = colnormalize(W, H, p=2)

Normalize the factorization so that each column satisfies `||W[:, i]||_p ≈ 1`.
`p` is the norm order passed to `LinearAlgebra.norm`, so any real order is
accepted (e.g. `1`, `2`, `Inf`).

[`merge_pq`](@ref) and [`nmfmerge`](@ref) require unit *2-norm* columns, so use
the default `p=2` when preparing input for the merge. Other orders are available
for unrelated normalization needs.

The component axis (columns of `W`, rows of `H`) must be one-based; the feature
axis (rows of `W`) and sample axis (columns of `H`) may have any axes.

"""
colnormalize(W, H, p::Real=2) = colnormalize!(float(copy(W)), float(copy(H)), p)

"""
    Wmerge, Hmerge, mergeseq = merge_pq([queuepenalty], W::AbstractArray, H::AbstractArray; nstop=1, errstop=typemax(...))

Merge components in `W` and `H` (columns in `W` and rows in `H`). Merging stops
at whichever of the two criteria `nstop` and `errstop` is reached first.

Arguments:

-`queuepenalty`: The same as in `nmfmerge`. Default: [`ssdpenalty`](@ref) (`f(E, h1sq, h2sq) = E`).

- `W::AbstractArray`: The basis matrix with unit 2-norm columns (e.g. from
  [`colnormalize`](@ref) with the default `p=2`). Columns that are not 2-normalized
  throw an `ArgumentError`.

- `H::AbstractArray`: The coefficient matrix.

Keyword arguments:

- `nstop::Integer` (default: 1): a floor on the number of components. Merging never produces
  fewer than `nstop` components. The default `nstop=1` merges to a single component.

- `errstop` (default: `typemax(...)`): a ceiling on the per-merge cost. Merging never performs a merge
  costing more than `errstop`. The cost is the `queuepenalty` value; under the
  default penalty it is the merge error.

The defaults on these keywords allow merging all the way down to a single component.

The component axis (columns of `W`, rows of `H`) is enumeration and must be
one-based. The feature axis (rows of `W`) and sample axis (columns of `H`) may
have any axes and are carried through to the output; the merged component axis
is one-based.

Outputs:

`Wmerge` and `Hmerge` are the merged results; the number of surviving components
is `nstop` unless `errstop` stopped merging earlier.

`mergeseq` is the sequence of merges as `(id1, id2, err)` tuples, in the order
they were performed. `id1` and `id2` are the ids of the merged components and
`err` is the reconstruction error (the merge penalty) incurred by that merge.
Ids larger than the number of columns in `W` refer to components produced by
earlier merges. The accumulating `err` values let a caller merge all the way
down (`nstop == 1`) and locate a "knee" at which to stop, then replay the
corresponding prefix of `mergeseq` with [`merge_replay`](@ref).
"""
function merge_pq(queuepenalty, W::AbstractArray, H::AbstractArray;
                  nstop::Integer=1, errstop=typemax(float(promote_type(eltype(W), eltype(H)))))
    check_component_axis(W, H)
    # Merge errors are floating-point combinations of the W and H entries.
    T = float(promote_type(eltype(W), eltype(H)))
    # Tolerance for the unit-2-norm check, scaled to the precision of W so that
    # e.g. Float32-normalized columns (norm error ~ eps(Float32)) are accepted.
    normtol = sqrt(eps(float(real(eltype(W)))))
    mrgseq = Tuple{Int, Int, T}[]
    # One-based stack of component columns (of `W`) and rows (of `H`); each merge
    # appends a new component. The feature axis (rows of `W`) and sample axis
    # (columns of `H`) ride along on these vectors and reappear in the output.
    # The component axis is one-based (checked above), so `1:size` enumerates it.
    Wcols = [W[:, j] for j in 1:size(W, 2)]
    Hrows = [H[i, :] for i in 1:size(H, 1)]
    for (id, w) in enumerate(Wcols)
        wnorm = norm(w)
        (abs(wnorm-1)<normtol || iszero(wnorm)) || throw(ArgumentError("W columns must have unit 2-norm; $(id)-th column 2-norm = $(wnorm). Use `colnormalize` with the default `p=2`."))
    end
    Nt = length(Wcols)
    Nt >= 2 || throw(ArgumentError("Cannot do 2 to 1 merge: Matrix size smaller than 2"))
    Nt >= nstop || throw(ArgumentError("Final solution more than original size"))
    # Merging marks components dead rather than deleting them, keeping ids stable.
    alive = trues(Nt)
    pq = PriorityQueue{Tuple{Int,Int},Float64}()
    for id0 in Nt:-1:2
        pq = pqupdate2to1!(queuepenalty, pq, Wcols, Hrows, alive, id0, 1:id0-1)
    end
    m = Nt
    while m > nstop && !isempty(pq)
        (id0, id1), penalty = first(pq)
        if !alive[id0] || !alive[id1]
            popfirst!(pq)
            continue
        end
        penalty > errstop && break
        popfirst!(pq)
        id01, loss = mergecol2to1!(Wcols, Hrows, alive, id0, id1)
        push!(mrgseq, (id0, id1, loss))
        pqupdate2to1!(queuepenalty, pq, Wcols, Hrows, alive, id01, 1:id01-1)
        m -= 1
    end
    return stack(Wcols[alive]), stack(Hrows[alive]; dims=1), mrgseq
end
merge_pq(W::AbstractArray, H::AbstractArray; kwargs...) = merge_pq(ssdpenalty, W, H; kwargs...)

# The component axis (columns of `W`, rows of `H`) is plain enumeration, so it
# must be shared and one-based; the feature axis (rows of `W`) and sample axis
# (columns of `H`) may carry any axes.
function check_component_axis(W, H)
    axes(W, 2) == axes(H, 1) || throw(DimensionMismatch("W has $(size(W, 2)) components but H has $(size(H, 1))"))
    isone(first(axes(W, 2))) || throw(ArgumentError("the component axis (columns of `W`, rows of `H`) must be one-based"))
    return nothing
end

function pqupdate2to1!(queuepenalty::Function, pq, S::AbstractVector, T::AbstractVector, alive::AbstractVector{Bool}, id01::Integer, overlapids::AbstractRange{To}) where To
    alive[id01] || return pq
    for id in overlapids
        if alive[id]
            _, loss, _, t1sq, t2sq = solve_remix(S, T, id, id01)
            push!(pq, (id, id01) => queuepenalty(loss, t1sq, t2sq))
        end
    end
    return pq
end

function solve_remix(S::AbstractVector, T::AbstractVector, id1::Integer, id2::Integer)
    τ, δ, c, h1h1, h1h2, h2h2 = build_tr_det(S, T, id1, id2)
    if iszero(h1h1)
        return c, zero(c), (zero(c), one(c)), h1h1, h2h2
    end
    if iszero(h2h2)
        return c, zero(c), (one(c), zero(c)), h1h1, h2h2
    end
    if iszero(c)
        # Check whether W1 or W2 is zero
        iszero(sum(abs2, S[id1])) && return c, zero(h1h1), (zero(c), one(c)), h1h1, h2h2
        iszero(sum(abs2, S[id2])) && return c, zero(h2h2), (one(c), zero(c)), h1h1, h2h2
    end
    b = sqrt(τ^2/4-δ)
    λ_max = τ/2+b
    λ_min = δ/λ_max
    den = (h1h2+c*h2h2)*2
    if iszero(den)
        u = h1h1 >= h2h2 ? (one(c), zero(c)) : (zero(c), one(c))
    else
        ξ = (h1h1-h2h2+2b)/den
        u = (ξ, 1)./sqrt(1+2ξ*c+ξ^2)
    end
    return c, λ_min, u, h1h1, h2h2
end

function build_tr_det(W::AbstractVector, H::AbstractVector, id1::Integer, id2::Integer)
    c = W[id1]'*W[id2]
    h1h1 = H[id1]'*H[id1]
    h1h2 = H[id1]'*H[id2]
    h2h2 = H[id2]'*H[id2]
    τ = h1h1+2c*h1h2+h2h2
    δ = (1-c^2)*(h1h1*h2h2-h1h2^2)
    return τ, δ, c, h1h1, h1h2, h2h2
end

function mergecol2to1!(S::AbstractVector, T::AbstractVector, alive::AbstractVector{Bool}, id0::Integer, id1::Integer)
    S01, T01, loss = mergepair(S, T, id0, id1)
    alive[id0] = alive[id1] = false
    push!(S, S01)
    push!(T, T01)
    push!(alive, true)
    return length(S), loss
end

function mergepair(S::AbstractVector, T::AbstractVector, id1::Integer, id2::Integer)
    c, loss, u, _, _ = solve_remix(S, T, id1, id2)
    S12, T12 = remix_enact(S, T, id1, id2, c, u)
    return S12, T12, loss
end

function remix_enact(S::AbstractVector, T::AbstractVector, id1::Integer, id2::Integer, c, w)
    S12 = w[1] .* S[id1] .+ w[2] .* S[id2]
    T12 = (w[1]+w[2]*c) .* T[id1] .+ (w[1]*c+w[2]) .* T[id2]
    return S12, T12
end

"""
    Wmerge, Hmerge = merge_replay(W, H, mergeseq)

Merge components in `W` and `H` (columns in `W` and rows in `H`) by replaying the
sequence of merge-pair ids in `mergeseq`, returning the merged factors.

Each entry of `mergeseq` supplies the pair of ids `(id1, id2)` to merge; any
further fields are ignored, so the `(id1, id2, err)` triples returned by
[`merge_pq`](@ref) can be replayed directly. Replaying a prefix of a `merge_pq`
schedule reproduces the factors `merge_pq` would have returned had it stopped at
the corresponding number of components.

As in [`merge_pq`](@ref), the component axis must be one-based while the feature
and sample axes are carried through to the output.
"""
function merge_replay(W::AbstractArray, H::AbstractArray, mergeseq::AbstractArray)
    check_component_axis(W, H)
    # One-based component stack carrying the feature/sample axes (see `merge_pq`).
    Wcols = [W[:, j] for j in 1:size(W, 2)]
    Hrows = [H[i, :] for i in 1:size(H, 1)]
    alive = trues(length(Wcols))
    for mergeids in mergeseq
        id0, id1 = mergeids
        mergecol2to1!(Wcols, Hrows, alive, id0, id1)
    end
    return stack(Wcols[alive]), stack(Hrows[alive]; dims=1)
end

"""
    ssdpenalty(E, h1sq, h2sq)

The default merge penalty: the merge error `E` itself, ignoring the squared
norms `h1sq`, `h2sq` of the two `H` rows. With this penalty `merge_pq`
and `nmfmerge` merge purely in order of increasing reconstruction error.

Pass a custom `f(E, h1sq, h2sq)` as the leading argument to those functions to
weight merges differently.
"""
ssdpenalty(E, h1sq, h2sq) = E

end
