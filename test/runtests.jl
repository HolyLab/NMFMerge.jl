using NMF_Merge, NMF, LinearAlgebra, DataStructures, ForwardDiff
using Test

function build_Qs(S::AbstractVector, T::AbstractVector, id1::Integer, id2::Integer; drop_crossterm::Bool=false)
    mS, mT = NMF_Merge.sdot(S, id1, id2), NMF_Merge.tdot(T, id1, id2)
    c = mS[1,2]
    τ1τ1 = mT[1,1]
    τ1τ2 = drop_crossterm ? zero(τ1τ1) : mT[1,2]
    τ2τ2 = mT[2,2]
    q1 = τ1τ1 + 2*c*τ1τ2 + c^2*τ2τ2
    q12 = c*τ1τ1 + (1+c^2)*τ1τ2 + c*τ2τ2
    q2 = c^2*τ1τ1 + 2*c*τ1τ2 + τ2τ2
    Q1 = [q1 q12; q12 q2]
    s1s1 = mS[1,1]
    s2s2 = mS[2,2]
    Q2 = [s1s1 c; c s2s2]
    return Q1, Q2, c, τ1τ1, τ1τ2, τ2τ2
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
    @test Q1 == Q1'
    @test Q2 == Q2
    F = eigen(Q1, Q2)
    Fvals, Fvecs = F.values::Vector{eltype(Q2)}, F.vectors::Matrix{eltype(Q2)}
    idx = argmax(Fvals)
    w = Fvecs[:,idx]

    τ, δ, c, h1h1, h1h2, h2h2 = NMF_Merge.build_tr_det(W2, H2, 1, 2)
    c, p, u = NMF_Merge.solve_remix(W2, H2, 1, 2)
    λ_max = τ/2+sqrt(τ^2/4-δ)
    λ_min = τ/2-sqrt(τ^2/4-δ)
    
    @test abs(λ_max - maximum(F.values))<=1e-12
    @test abs(λ_min - minimum(F.values))<=1e-12

    @test abs(u[1]/u[2] - w[1]/w[2])<1e-3*(u[1]/u[2] + w[1]/w[2])/2
    
    w = [u[1], u[2]]
    @test norm(Q1*w - maximum(F.values)*Q2*w) < 1e-12

    W12, H12, loss = NMF_Merge.mergepair(W2, H2, 1, 2)
    Err(Hm) = sum(abs2, W12*Hm'-W1*H1)
    @test norm(ForwardDiff.gradient(Err, H12)) < 1e-12

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

    Wm, Hm, _, _, mergids = colmerge2to1pq(copy(Wn), copy(Hn), 2)
    Wn1 = [Wn[:, j] for j in axes(Wn, 2)];
    Hn1 = [Hn[i, :] for i in axes(Hn, 1)];
    Ids = [(1,2), (1,3), (2,3)]
    loss1 = NMF_Merge.mergepair(Wn1, Hn1, Ids[1][1], Ids[1][2])[end]
    loss2 = NMF_Merge.mergepair(Wn1, Hn1, Ids[2][1], Ids[2][2])[end]
    loss3 = NMF_Merge.mergepair(Wn1, Hn1, Ids[3][1], Ids[3][2])[end]
    i = findmin([loss1, loss2, loss3])[2]

    @test pop!(copy(mergids)) == Ids[i]
    @test pop!(mergids) == (1,2)

end

