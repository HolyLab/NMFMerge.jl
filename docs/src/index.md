```@meta
CurrentModule = NMFMerge
```

# NMFMerge.jl

NMFMerge improves nonnegative matrix factorization (NMF) by *over-factorizing*
and then merging components in pairs. Factorizing a data matrix ``X \approx WH``
with more components than ultimately desired, and then optimally and
sequentially merging the closest component pairs, tends to reach better and more
reproducible optima than factorizing directly into the target number of
components.

The method is described in [An optimal pairwise merge algorithm improves the
quality and consistency of nonnegative matrix
factorization](https://ieeexplore.ieee.org/abstract/document/11071940)
(Guo & Holy, IEEE Trans. Signal Process. 2025,
[doi:10.1109/TSP.2025.3585893](https://doi.org/10.1109/TSP.2025.3585893)).

## Why merging helps

Convergence of NMF becomes poor when the factorization is forced to make
difficult tradeoffs in describing different features of the data matrix.
Performing an initial factorization with an excess of components grants the
opportunity to escape such constraints and describe the full behavior of the
data; redundant or noisy components are then identified and merged together.

![The NMF-Merge concept](assets/ovrsimumerge.png)

Here the data matrix is ``X = WH + N``, where ``W`` and ``H`` are rank 5 and
``N`` is noise. The "Good" and "Bad" factorizations (blue box) are two local
minima reached by rank-5 NMF across 1000 different initializations, while the
factorizations in the green boxes are minima found by NMF with ``r \ge 6``. The
higher-rank solutions can be merged to produce a rank-5 factorization. Thus, by
first computing a higher-rank NMF and then merging to lower rank, you more
reliably identify high-quality solutions.

## Installation

NMFMerge is a registered package. From the Julia REPL, type `]` to enter
package mode and run:

```julia
pkg> add NMFMerge
```

## Quick start: escaping a local minimum

Take a small nonnegative matrix that is exactly rank 3, so it has an *exact*
rank-3 factorization with zero reconstruction error:

```julia
using NMFMerge, NMF, LinearAlgebra

X = [0 4 0 2 0 4 2 2;
     2 2 2 1 4 4 1 1;
     2 6 4 6 4 4 4 2;
     2 1 3 2 4 1 1 0]
```

Even so, rank-3 NMF has a spurious local minimum, and standard NMF (HALS) with
NNDSVD initialization converges to it:

```julia
julia> f = svd(float(X));

julia> result_hals = nnmf(float(X), 3; init=:nndsvd, alg=:cd, initdata=f, maxiter=10^6, tol=1e-8);

julia> result_hals.objvalue / sum(abs2, X)     # relative fitting error
0.0007391570365682461
```

This is not slow convergence. The point is a genuine *strict* local minimum: its
KKT conditions hold and its reduced Hessian is positive definite away from the
trivial column-scaling directions, so further iteration cannot escape it.

NMF-Merge factorizes with one extra component and merges back to three, reaching
the global optimum — the exact factorization:

```julia
julia> result_merge = nmfmerge(float(X), 4 => 3; alg=:cd, maxiter=10^6, tol_final=1e-10);

julia> result_merge.objvalue / sum(abs2, X)    # ≈ machine precision: exact recovery
7.5e-16
```

Standard NMF is stuck with a relative fitting error near ``7 \times 10^{-4}``,
while NMF-Merge recovers the factorization to machine precision. The merge step draws
on a randomized truncated SVD, so the exact value varies slightly from run to
run.

The two solutions are not small perturbations of each other; they are
structurally different factorizations. Matching the components for maximum
overlap (the factorization is invariant to reordering) and unit-normalizing the
columns of ``W``, two of the three components nearly coincide but the third is
badly mismatched:

```
component   HALS (error 0.63)     optimal merge (error ≈ 0)   cosine
    1       0.72 0.47 0.51 0.00   0.67 0.33 0.67 0.00          0.98
    2       0.16 0.00 0.94 0.30   0.00 0.00 0.89 0.45          0.98
    3       0.00 0.57 0.57 0.59   0.00 0.89 0.00 0.45          0.77
```

Each block lists one component's loadings on the four features. Standard NMF
smears feature 3 across components 2 and 3, while the optimal solution keeps it
in component 2 alone; the rows of ``H`` differ in the same way. No reordering or
rescaling reconciles the two — standard NMF has converged to a genuinely
different, worse description of the data, and merging from an over-complete fit
escapes it.

## Lower-level workflow

[`nmfmerge`](@ref) runs the whole pipeline, but the building blocks are exported
so you can drive merging yourself, for example when choosing the number of
components:

1. [`colnormalize`](@ref) rescales the columns of `W` to unit 2-norm (merging
   requires unit-norm columns), pushing the scale into `H` so that `W * H` is
   unchanged.
2. [`merge_pq`](@ref) merges a normalized factorization down to a chosen number
   of components and records the sequence of merges, including the reconstruction
   error each merge incurs. Merging all the way to one component and inspecting
   those errors might provide hints about an appropriate number of components.
3. [`merge_replay`](@ref) replays a prefix of that merge sequence, reproducing
   the factors `merge_pq` would have returned had it stopped at the
   corresponding number of components.

See the [API reference](@ref "API reference") for full signatures and examples.

## Citation

If you use this package, please cite the publication above; see the "Cite this
repository" link in the GitHub "About" bar for format options.
