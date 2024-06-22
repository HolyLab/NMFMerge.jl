module NMFMerge

using LinearAlgebra, DataStructures, NMF, GsvdInitialization

export nmfmerge,
       colnormalize,
       colmerge2to1pq,
       mergecolumns

function nmfmerge(X, ncomponents::Pair{Int,Int}; tol_final=1e-4, tol_intermediate=sqrt(tol_final), W0=nothing, H0=nothing, kwargs...)
    f = svd(X)
    if W0 === nothing && H0 === nothing
        W0, H0 = NMF.nndsvd(X, ncomponents[2], initdata=f)
    end
    result_initial = nnmf(X, ncomponents[2]; kwargs..., init=:custom, tol=tol_intermediate, W0=copy(W0), H0=copy(H0))
    W_initial, H_initial = result_initial.W, result_initial.H
    kadd = ncomponents[1] - ncomponents[2]
    W_over_init, H_over_init = overnmfinit(X, copy(W_initial), copy(H_initial), kadd, initdata=f)
    result_over = nnmf(X, ncomponents[1]; kwargs..., init=:custom, tol=tol_intermediate, W0=copy(W_over_init), H0=copy(H_over_init))
    W_over, H_over = result_over.W, result_over.H
    W_over_normed, H_over_normed = colnormalize(W_over, H_over)
    Wmerge, Hmerge, _ = colmerge2to1pq(W_over_normed, H_over_normed, ncomponents[2])
    result_renmf = nnmf(X, ncomponents[2]; kwargs..., init=:custom, tol=tol_final, W0=copy(Wmerge), H0=copy(Hmerge))
    return result_renmf
end
    
"""
    colnormalize(W, H, p)

This function normalize ||W[:, i]||_p = 1 for i in 1:size(W, 2)

To use this function:
Wnormalized, Hnormalized = colnormalize(W, H, p)
"""
       
function colnormalize!(W, H, p::Integer=2)
    for (j, w) in pairs(eachcol(W))
        normw = norm(w, p)
        if !iszero(normw)
            W[:, j] = w/normw
            H[j, :] = H[j, :]*normw
        end
    end
    return W, H
end
colnormalize(W, H, p::Integer=2) = colnormalize!(float(copy(W)), float(copy(H)), p)

"""
    colmerge2to1pq(S::AbstractArray, T::AbstractArray, n::Integer)

This function merges components in W and H (columns in W and rows in H) from original number of components to n components (n columns and rows left in W and H respectively).

To use this function:
Wmerge, Hmerge, mergeseq = colmerge2to1pq(W, H, n), where Wmerge and Hmerge are the merged results with n components. mergeseq is the sequence of merge pair ids (id1, id2), which is the components id of single merge.
"""
function colmerge2to1pq(S::AbstractArray, T::AbstractArray, n::Integer)
    mrgseq = Tuple{Int, Int}[]
    S = [S[:, j] for j in axes(S, 2)];
    T = [T[i, :] for i in axes(T, 1)];
    for s in S
        abs(norm(s)-1)<1e-12 || throw(ArgumentError("W columns must be normalized"))
    end
    Nt = length(S)
    Nt >= 2 || throw(ArgumentError("Cannot do 2 to 1 merge: Matrix size smaller than 2"))
    Nt >= n || throw(ArgumentError("Final solution more than original size"))
    pq = initialize_pq_2to1(S, T)
    m = Nt
    while m > n
        id0, id1 = dequeue!(pq)
        if isempty(S[id0])||isempty(S[id1])
            continue
        end
        push!(mrgseq, (id0, id1))
        S, T, id01, _ = mergecol2to1!(S, T, id0, id1);
        pqupdate2to1!(pq, S, T, id01, 1:id01-1);
        m -= 1
    end
    Smtx, Tmtx = reduce(hcat, filter(!isempty, S)), reduce(hcat, filter(!isempty, T))'
    return Smtx, Matrix(Tmtx), mrgseq
end

function initialize_pq_2to1(S::AbstractVector, T::AbstractVector)
    err_pq = PriorityQueue{Tuple{Int, Int},Float64}()
    for id0 in length(S):-1:2
        err_pq = pqupdate2to1!(err_pq, S, T, id0, 1:id0-1)
    end
    return err_pq
end

function pqupdate2to1!(pq, S::AbstractVector, T::AbstractVector, id01::Integer, overlapids::AbstractRange{To}) where To
    for id in overlapids
        if !isempty(S[id]) && !isempty(S[id01])
            loss = solve_remix(S, T, id, id01)[2]
            enqueue!(pq, (id, id01), loss)    
        end
    end
    return pq
end

function solve_remix(S, T, id1, id2)
    τ, δ, c, h1h1, h1h2, h2h2 = build_tr_det(S, T, id1, id2)
    if h1h1 == 0
        return c, zero(c), (zero(c),one(c))
    end
    if h2h2 == 0
        return c, zero(c), (one(c),zero(c))
    end
    b = sqrt(τ^2/4-δ)
    λ_max = τ/2+b
    λ_min = δ/λ_max
    ξ = (h1h1-h2h2+2b)/((h1h2+c*h2h2)*2)
    u = (ξ, 1)./sqrt(1+2ξ*c+ξ^2)
    return c, λ_min, u
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

function mergecol2to1!(S::AbstractVector, T::AbstractVector, id0::Integer, id1::Integer)
    S01, T01, loss = mergepair(S, T, id0, id1)
    S[id0] = S[id1] = T[id0] = T[id1] = eltype(S[1])[]
    id01 = length(S)+1
    push!(S, S01)
    push!(T, T01)
    return S, T, id01, loss
end

function mergepair(S::AbstractVector, T::AbstractVector, id1::Integer, id2::Integer)
    c, loss, u, = solve_remix(S, T, id1, id2)
    S12, T12 = remix_enact(S, T, id1, id2, c, u)
    return S12, T12, loss
end

function remix_enact(S::AbstractVector{TS}, T::AbstractVector, id1::Integer, id2::Integer, c::AbstractFloat, w::Tuple{Tw, Tw}) where {Tw, TS}
    S12 = zeros(eltype(TS), length(S[id1]))
    S12 += w[1]*S[id1]
    S12 += w[2]*S[id2]
    T1, T2 = (w[1]+w[2]*c)*T[id1], (w[1]*c+w[2])*T[id2]
    T12 = T1+T2
    return S12, T12
end
  
"""
    mergecolumns(W, H, mergeseq; tracemerge=false)

This function merges components in ``W`` and ``H`` (columns in ``W`` and rows in ``H``) according to the sequence of merge pair ids ``mergeseq``.

To use this function:
Wmerge, Hmerge, WHstage, Err = mergecolumns(W, H, mergeseq; tracemerge), where ``Wmerge`` and ``Hmerge`` are the merged results. ``WHstage::Vector{Tuple{Matrix, Matrix}}`` includes the results of each merge stage. ``WHstage=[]`` if ``tracemerge=false``. ``Err::Vector`` includes merge penalty of each merge stage.
"""
function mergecolumns(W::AbstractArray, H::AbstractArray, mergeseq::AbstractArray; tracemerge::Bool = false)
    Err = Float64[]
    S = [W[:, j] for j in axes(W, 2)]
    T = [H[i, :] for i in axes(H, 1)]
    STstage = []
    for mergeids in mergeseq
        id0, id1 = mergeids
        if tracemerge
            push!(STstage, (copy(S), copy(T)))
        end
        S, T, _, loss = mergecol2to1!(S, T, id0, id1)
        err = loss
        push!(Err, err)
    end
    Smtx, Tmtx = hcat(filter(x -> x != [], S)...), hcat(filter(x -> x != [], T)...)'
    return Smtx, Matrix(Tmtx), STstage, Err
end

end
