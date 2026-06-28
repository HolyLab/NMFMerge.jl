using NMFMerge, NMF, LinearAlgebra, DataStructures, ForwardDiff
using Test
using Aqua
using ExplicitImports
using OffsetArrays

@testset "Aqua" begin
    Aqua.test_all(NMFMerge)
end

@testset "ExplicitImports" begin
    # nndsvd is accessed as NMF.nndsvd; it is not public in NMF, but NMF
    # provides no public NNDSVD entry point, so the access is unavoidable.
    test_explicit_imports(NMFMerge;
                          all_explicit_imports_are_public   = VERSION >= v"1.11",
                          all_qualified_accesses_are_public = VERSION >= v"1.11",
                          ignore = (:nndsvd,))
end

function build_Qs(S::AbstractVector, T::AbstractVector, id1::Integer, id2::Integer)
    c = S[id1]'*S[id2]
    τ1τ1 = T[id1]'*T[id1]
    τ1τ2 = T[id1]'*T[id2]
    τ2τ2 = T[id2]'*T[id2]
    q1 = τ1τ1 + 2*c*τ1τ2 + c^2*τ2τ2
    q12 = c*τ1τ1 + (1+c^2)*τ1τ2 + c*τ2τ2
    q2 = c^2*τ1τ1 + 2*c*τ1τ2 + τ2τ2
    Q1 = [q1 q12; q12 q2]
    s1s1 = 1
    s2s2 = 1
    Q2 = [s1s1 c; c s2s2]
    return Q1, Q2, c, τ1τ1, τ1τ2, τ2τ2
end

W_GT = [6 0 4 9;
     0 4 8 3;
     4 4 0 7;
     9 1 1 1;
     0 3 0 4;
     8 1 4 0;
     0 0 4 2;
     0 9 5 5
    ]

H_GT = [6 10 8 2 0 1 2 10;
     0 10 2 9 10 6 0 0;
     3 5 0 2 4 0 0 8;
     4 9 10 7 7 0 0 0
    ]

@testset "test top wrapper" begin
    W = W_GT[:, 3:4]
    H = H_GT[3:4, :]
    X = W*H
    result_renmf = nmfmerge(float(X), 3=>2; alg = :cd, maxiter = 10^5, tol_final=1e-12, tol_intermediate = 1e-12);
    W_renmf, H_renmf = result_renmf.W, result_renmf.H
    @test size(W_renmf, 2) == 2
    @test size(H_renmf, 1) == 2
    @test sum(abs2, X - W_renmf*H_renmf) <= 1e-12

    standard_nmf = nnmf(float(X), 2; init=:nndsvd, tol=1e-12, initdata=svd(float(X)))
    result_renmf = nmfmerge(float(X), 2=>2; alg=:cd, maxiter=10^5, tol_final=1e-12, tol_intermediate=1e-12)
    W_standard, H_standard = standard_nmf.W, standard_nmf.H
    W_renmf, H_renmf = result_renmf.W, result_renmf.H
    W_standard, H_standard = colnormalize(W_standard, H_standard)
    W_renmf, H_renmf = colnormalize(W_renmf, H_renmf)
    @test sum(abs2, W_standard - W_renmf) <= 1e-12
    @test sum(abs2, H_standard - H_renmf) <= 1e-12

    X = rand(30, 20)
    result_1 = nmfmerge(X, 10; alg=:cd)
    result_2 = nmfmerge(X, 12 => 10; alg=:cd)
    @test sum(abs2, result_1.W - result_2.W) <= 1e-12*sum(abs2, result_1.W)
    @test sum(abs2, result_1.H - result_2.H) <= 1e-12*sum(abs2, result_1.H)

    result_1 = nmfmerge(X, 4; alg=:cd)
    result_2 = nmfmerge(X, 5 => 4; alg=:cd)
    @test sum(abs2, result_1.W - result_2.W) <= 1e-12*sum(abs2, result_1.W)
    @test sum(abs2, result_1.H - result_2.H) <= 1e-12*sum(abs2, result_1.H)

    result_1 = nmfmerge(X, 8; alg=:cd)
    result_2 = nmfmerge(X, 10 => 8; alg=:cd)

    @test sum(abs2, result_1.W - result_2.W) <= 1e-12*sum(abs2, result_1.W)
    @test sum(abs2, result_1.H - result_2.H) <= 1e-12*sum(abs2, result_1.H)
end

@testset "ncomponents accepts any integer pair" begin
    X = rand(30, 20)
    # The widening method only coerces the pair's integer types; the numerical
    # result is exercised elsewhere. Asserting equality across separate solves
    # would instead test bit-reproducibility of multithreaded BLAS.
    for nc in (Int32(12) => Int32(10), 12 => Int32(10), Int32(12) => 10)
        res = nmfmerge(X, nc; alg=:cd)
        @test size(res.W, 2) == 10
        @test size(res.H, 1) == 10
    end
end

@testset "merge coefficients" begin
    for i in 1:3, j in i+1:4
        W = W_GT[:, [i,j]]
        H = H_GT[[i,j], :]
        Wn, Hn = colnormalize(W, H)
        W_v = [Wn[:, j] for j in axes(Wn, 2)]
        H_v = [Hn[i, :] for i in axes(Hn, 1)]

        Q1, Q2, _, _, _, _ =build_Qs(W_v, H_v, 1, 2)
        @test issymmetric(Q1)
        @test issymmetric(Q2)
        F = eigen(Q1, Q2)
        Fvals, Fvecs = F.values, F.vectors
        idx = argmax(Fvals)
        w = Fvecs[:,idx]

        τ, δ, c, h1h1, h1h2, h2h2 = NMFMerge.build_tr_det(W_v, H_v, 1, 2)
        c, p, u, h1h1, h2h2 = NMFMerge.solve_remix(W_v, H_v, 1, 2)
        u = [u[1], u[2]]
        b = sqrt(τ^2/4-δ)
        λ_max = τ/2+b
        λ_min = δ/λ_max

        @test abs(λ_max - maximum(F.values))<=1e-10
        @test abs(λ_min - minimum(F.values))<=1e-10

        @test abs(u[1]*w[2] - w[1]*u[2])<1e-10

        @test norm(u[1].*W_v[1].+u[2].*W_v[2]) ≈ 1
        @test norm(Q1*u - maximum(F.values)*Q2*u) <= 1e-10
        @test norm(Q1*u - λ_max*Q2*u) <= 1e-10

        W12, H12, loss = NMFMerge.mergepair(W_v, H_v, 1, 2)
        Err(Hm) = sum(abs2, W12 * Hm' - W * H)
        @test norm(ForwardDiff.gradient(Err, H12)) <= 1e-10
    end
end

@testset "colnormalize norm order" begin
    W = rand(6, 4)
    H = rand(4, 9)
    for p in (1, 2, Inf)
        Wn, Hn = colnormalize(W, H, p)
        # Columns of W are unit-`p`-norm and the factorization is preserved.
        for j in axes(Wn, 2)
            @test norm(Wn[:, j], p) ≈ 1
        end
        @test Wn * Hn ≈ W * H
    end
end

@testset "Single-component image" begin
    ns = 31
    nt = 100
    nthalf = nt>>1
    w = 7
    W = exp.(-((1:ns) .- (ns+1)>>1).^2/(2*w^2))
    H = rand(Float64, 1, nt)
    img = W*H+eps()*randn(ns, nt)

    W0, H0 = NMF.nndsvd(img, 2)
    imgnf = NMF.solve!(NMF.CoordinateDescent{Float64}(), img, W0, H0)
    W1, H1 = imgnf.W, imgnf.H
    W1n, H1n = colnormalize(W1, H1)
    [@test abs(norm(W1n[:,j], 2)-1) <= 1e-12 for j in axes(W1n, 2)]

    W2 = [W1n[:, j] for j in axes(W1n, 2)];
    H2 = [H1n[i, :] for i in axes(H1n, 1)];

    Q1, Q2, _, _, _, _ =build_Qs(W2, H2, 1, 2)
    @test issymmetric(Q1)
    @test issymmetric(Q2)
    F = eigen(Q1, Q2)
    Fvals, Fvecs = F.values, F.vectors
    idx = argmax(Fvals)
    w = Fvecs[:,idx]


    τ, δ, c, h1h1, h1h2, h2h2 = NMFMerge.build_tr_det(W2, H2, 1, 2)
    c, p, u, h1h1, h2h2 = NMFMerge.solve_remix(W2, H2, 1, 2)
    u = [u[1], u[2]]
    b = sqrt(τ^2/4-δ)
    λ_max = τ/2+b
    λ_min = δ/λ_max

    @test abs(λ_max - maximum(F.values))<=1e-12
    @test abs(λ_min - minimum(F.values))<=1e-10

    @test abs(u[1]*w[2] - w[1]*u[2])<1e-12

    @test norm(u[1].*W2[1].+u[2].*W2[2]) ≈ 1
    @test norm(Q1*u - maximum(F.values)*Q2*u) <= 1e-10
    @test norm(Q1*u - λ_max*Q2*u) <= 1e-10

    W12, H12, _ = NMFMerge.mergepair(W2, H2, 1, 2)
    Err(Hm) = sum(abs2, W12*Hm'-W1*H1)
    @test norm(ForwardDiff.gradient(Err, H12)) <= 1e-12

end

@testset "Merge by min err" begin
    # Two cells, one is bright and the other dim. The bright cell is split into two tiles that alternate time points
    S1 = [0.1, 0.5, 0.4, 0.0, 0.0, 0.0]; S1 = S1 / norm(S1);
    S2 = [0.0, 0.0, 0.1, 0.9, 0.0, 0.0]; S2 = S2 / norm(S2);
    T1 = rand(20)
    T1a, T1b = copy(T1), copy(T1)
    T1a[1:2:end] .= 0
    T1b[2:2:end] .= 0
    coef = 0.1
    N1a = randn(length(S1)); N1a = N1a / norm(N1a) * coef
    N1b = randn(length(S1)); N1b = N1b / norm(N1b) * coef
    T2 = zero(T1)
    T2[15] = 0.25 * sqrt(min(sum(abs2, N1a) * sum(abs2, T1a), sum(abs2, N1b) * sum(abs2, T1b)))

    W, H = [S1 S1 S2], [T1a'; T1b'; T2']
    W0, H0 = [S1 S2], [T1'; T2']
    Wn, Hn = colnormalize(W, H)
    @test sum(abs2, W*H - Wn*Hn) < 1e-16

    Wm, Hm, mergids = merge_pq(copy(Wn), copy(Hn); nstop=2)
    Wn1 = [Wn[:, j] for j in axes(Wn, 2)];
    Hn1 = [Hn[i, :] for i in axes(Hn, 1)];
    Ids = [(1,2), (1,3), (2,3)]
    loss1 = NMFMerge.mergepair(Wn1, Hn1, Ids[1][1], Ids[1][2])[end]
    loss2 = NMFMerge.mergepair(Wn1, Hn1, Ids[2][1], Ids[2][2])[end]
    loss3 = NMFMerge.mergepair(Wn1, Hn1, Ids[3][1], Ids[3][2])[end]
    i = findmin([loss1, loss2, loss3])[2]

    @test pop!(copy(mergids))[1:2] == Ids[i]
    @test pop!(mergids)[1:2] == (1,2)

end

@testset "Merge zero component" begin
    wrand = rand(5)
    wrand ./= norm(wrand)
    Ws = [wrand, zeros(5)]
    Hs = [rand(10), rand(10)]
    W12, H12, loss = NMFMerge.mergepair(Ws, Hs, 1, 2)
    @test W12 ≈ wrand
    @test H12 ≈ Hs[1]
    @test iszero(loss)

    wrand1 = rand(5)
    wrand1 ./= norm(wrand1)
    Ws = [wrand zeros(5) wrand wrand1]
    Hs = [rand(10) rand(10) zeros(10) rand(10)]'
    Wmerge, Hmerge, _ = merge_pq(Ws, Hs; nstop=1)
    @test size(Wmerge, 2) == 1

    W14, H14, _ = NMFMerge.mergepair([Ws[:,1], Ws[:,4]], [Hs[1,:], Hs[4,:]], 1, 2)
    @test W14 ≈ Wmerge[:]
    @test H14 ≈ Hmerge[:]
end

@testset "test customized merge function" begin
    Ws = [rand(5) rand(5) rand(5)]
    Hs = [rand(10) rand(10) rand(10)]'
    Wsn, Hsn = colnormalize(Ws, Hs)
    Wsn1 = [Wsn[:, j] for j in axes(Wsn, 2)]
    Hsn1 = [Hsn[i, :] for i in axes(Hsn, 1)]
    mergepenalty_custom(E, t1sq, t2sq) = -E
    idpair_loss = []
    for id1 in 1:2, id2 in id1+1:3
        W12, H12, loss2 = NMFMerge.mergepair(Wsn1, Hsn1, id1, id2)
        push!(idpair_loss, ((id1, id2), loss2))
    end
    idpair_loss = sort(idpair_loss, by=x->x[2])
    merge_sequence = merge_pq(Wsn, Hsn; nstop=1)[end]
    merge_sequence_custom = merge_pq(mergepenalty_custom, Wsn, Hsn; nstop=1)[end]
    @test merge_sequence[1][1:2] == idpair_loss[1][1]
    @test merge_sequence_custom[1][1:2] == idpair_loss[3][1]

end

@testset "Knee-finding workflow" begin
    Wn, Hn = colnormalize(float.(W_GT), float.(H_GT))
    ncols = size(Wn, 2)

    # Merge all the way down to a single component, recording every merge error.
    Wfull, Hfull, schedule = merge_pq(Wn, Hn; nstop=1)
    @test size(Wfull, 2) == 1
    @test length(schedule) == ncols - 1
    errs = [s[3] for s in schedule]
    @test all(>=(0), errs)

    # Replaying the full schedule reproduces the final factors.
    Wr, Hr = merge_replay(Wn, Hn, schedule)
    @test Wr ≈ Wfull
    @test Hr ≈ Hfull

    # Knee: stopping the merge at rank k equals replaying the first ncols-k
    # merges of the full schedule.
    for k in 1:ncols-1
        Wk, Hk, _ = merge_pq(Wn, Hn; nstop=k)
        Wkr, Hkr = merge_replay(Wn, Hn, schedule[1:ncols-k])
        @test size(Wkr, 2) == k
        @test Wkr ≈ Wk
        @test Hkr ≈ Hk
    end
end

@testset "merge_pq requires unit 2-norm columns" begin
    # Columns normalized in a norm other than 2 are rejected, and the message
    # points at the 2-norm requirement.
    W = rand(6, 4)
    H = rand(4, 9)
    W1, H1 = colnormalize(W, H, 1)   # unit 1-norm, not unit 2-norm
    @test_throws "unit 2-norm" merge_pq(W1, H1; nstop=2)
end

@testset "merge_pq error eltype follows factors" begin
    W = rand(6, 4); H = rand(4, 9)
    for Tf in (Float64, Float32)
        Wn, Hn = @inferred colnormalize(Tf.(W), Tf.(H))
        @test eltype(Wn) === Tf
        Wm, Hm, seq = @inferred merge_pq(Wn, Hn; nstop=1)
        @test eltype(Wm) === Tf
        @test eltype(seq) === Tuple{Int,Int,Tf}
    end
end

@testset "errstop stopping criterion" begin
    Wn, Hn = colnormalize(float.(W_GT), float.(H_GT))
    ncols = size(Wn, 2)
    _, _, schedule = merge_pq(Wn, Hn; nstop=1)
    errs = [s[3] for s in schedule]
    @test issorted(errs)  # the cut tests below rely on increasing merge errors

    # A threshold between two successive merge errors stops just before the
    # costlier merge, leaving more than the `nstop` floor of components.
    thresh = (errs[2] + errs[3]) / 2
    W2, _, seq2 = merge_pq(Wn, Hn; nstop=1, errstop=thresh)
    @test length(seq2) == 2
    @test size(W2, 2) == ncols - 2

    # A threshold below the cheapest merge performs no merges.
    W0, _, seq0 = merge_pq(Wn, Hn; nstop=1, errstop=errs[1] - 1)
    @test isempty(seq0)
    @test size(W0, 2) == ncols

    # The `nstop` floor still caps merging even when errstop permits more.
    Wf, _, seqf = merge_pq(Wn, Hn; nstop=ncols - 1, errstop=Inf)
    @test length(seqf) == 1
    @test size(Wf, 2) == ncols - 1

    # The default errstop reproduces the unrestricted merge.
    Wd, _, seqd = merge_pq(Wn, Hn; nstop=1)
    @test length(seqd) == length(schedule)
    @test size(Wd, 2) == 1
end

@testset "generic axes" begin
    Wn, Hn = colnormalize(float.(W_GT), float.(H_GT))

    # The component axis (columns of `W`, rows of `H`) is enumeration and must be
    # one-based, but the feature axis (rows of `W`) and sample axis (columns of
    # `H`) may be offset and are carried through to the output. Offset only those
    # two; the merged component axis comes out one-based.
    Wo = OffsetArray(collect(Wn), -2, 0)    # feature shift -2, component one-based
    Ho = OffsetArray(collect(Hn), 0, -1)    # component one-based, sample shift -1
    vals(A) = collect(A)[:]                 # values in column-major order, axes discarded

    @testset "colnormalize" begin
        rW, rH = @inferred colnormalize(Wn, Hn)
        oW, oH = @inferred colnormalize(Wo, Ho)
        @test vals(oW) == vals(rW) && vals(oH) == vals(rH)
        @test axes(oW, 1) == axes(Wo, 1)        # feature axis preserved
        @test axes(oH, 2) == axes(Ho, 2)        # sample axis preserved
        vW, vH = @inferred colnormalize(view(Wn, :, :), view(Hn, :, :))
        @test vW == rW && vH == rH
    end

    @testset "merge_pq" begin
        rW, rH, rseq = @inferred merge_pq(Wn, Hn; nstop=2)
        oW, oH, oseq = @inferred merge_pq(Wo, Ho; nstop=2)
        @test vals(oW) == vals(rW) && vals(oH) == vals(rH) && oseq == rseq
        @test axes(oW, 1) == axes(Wo, 1)        # feature axis preserved
        @test axes(oW, 2) == 1:2                # components re-enumerated, one-based
        @test axes(oH, 2) == axes(Ho, 2)        # sample axis preserved
        vW, vH, vseq = @inferred merge_pq(view(Wn, :, :), view(Hn, :, :); nstop=2)
        @test vW == rW && vH == rH && vseq == rseq
    end

    @testset "merge_replay" begin
        _, _, seq = merge_pq(Wn, Hn; nstop=1)
        rW, rH = @inferred merge_replay(Wn, Hn, seq)
        oW, oH = @inferred merge_replay(Wo, Ho, seq)
        @test vals(oW) == vals(rW) && vals(oH) == vals(rH)
        @test axes(oW, 1) == axes(Wo, 1)        # feature axis preserved
        @test axes(oH, 2) == axes(Ho, 2)        # sample axis preserved
        vW, vH = @inferred merge_replay(view(Wn, :, :), view(Hn, :, :), seq)
        @test vW == rW && vH == rH
    end

    @testset "mismatched component dimension" begin
        @test_throws DimensionMismatch colnormalize(rand(6, 4), rand(3, 9))
        @test_throws DimensionMismatch merge_pq(rand(6, 4), rand(3, 9); nstop=2)
        @test_throws DimensionMismatch merge_replay(rand(6, 4), rand(3, 9), [(1, 2)])
    end

    @testset "non-one-based component axis rejected" begin
        Wc = OffsetArray(collect(Wn), 0, -3)    # component axis shifted off one
        Hc = OffsetArray(collect(Hn), -3, 0)
        @test_throws "one-based" colnormalize(Wc, Hc)
        @test_throws "one-based" merge_pq(Wc, Hc; nstop=2)
        @test_throws "one-based" merge_replay(Wc, Hc, [(1, 2)])
    end

    @testset "nmfmerge rejects offset input" begin
        # `nmfmerge` delegates to TSVD/NMF, which require one-based indexing.
        X = OffsetArray(rand(20, 15), -2, -3)
        @test_throws "offset arrays are not supported" nmfmerge(X, 4; alg=:cd)
    end
end
