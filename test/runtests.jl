using NMFMerge, NMF, LinearAlgebra, DataStructures, ForwardDiff
using Test

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
    @test sum(abs2, result_1.W - result_2.W) <= 1e-12
    @test sum(abs2, result_1.H - result_2.H) <= 1e-12

    result_1 = nmfmerge(X, 4; alg=:cd)
    result_2 = nmfmerge(X, 5 => 4; alg=:cd)
    @test sum(abs2, result_1.W - result_2.W) <= 1e-12
    @test sum(abs2, result_1.H - result_2.H) <= 1e-12

    result_1 = nmfmerge(X, 8; alg=:cd)
    result_2 = nmfmerge(X, 10 => 8; alg=:cd)
    @test sum(abs2, result_1.W - result_2.W) <= 1e-12
    @test sum(abs2, result_1.H - result_2.H) <= 1e-12
    
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
        c, p, u = NMFMerge.solve_remix(W_v, H_v, 1, 2)
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
    c, p, u = NMFMerge.solve_remix(W2, H2, 1, 2)
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

    W12, H12, loss = NMFMerge.mergepair(W2, H2, 1, 2)
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

    Wm, Hm, mergids = colmerge2to1pq(copy(Wn), copy(Hn), 2)
    Wn1 = [Wn[:, j] for j in axes(Wn, 2)];
    Hn1 = [Hn[i, :] for i in axes(Hn, 1)];
    Ids = [(1,2), (1,3), (2,3)]
    loss1 = NMFMerge.mergepair(Wn1, Hn1, Ids[1][1], Ids[1][2])[end]
    loss2 = NMFMerge.mergepair(Wn1, Hn1, Ids[2][1], Ids[2][2])[end]
    loss3 = NMFMerge.mergepair(Wn1, Hn1, Ids[3][1], Ids[3][2])[end]
    i = findmin([loss1, loss2, loss3])[2]

    @test pop!(copy(mergids)) == Ids[i]
    @test pop!(mergids) == (1,2)

end