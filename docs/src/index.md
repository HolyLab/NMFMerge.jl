```@meta
CurrentModule = NMFMerge
```

# NMFMerge

Documentation for [NMFMerge](https://github.com/HolyLab/NMFMerge.jl).

```@index
```

```@autodocs
Modules = [NMFMerge]
```

```@docs
colnormalize(W, H, p::Integer=2)
colmerge2to1pq(S::AbstractArray, T::AbstractArray, n::Integer)
mergecolumns(W::AbstractArray, H::AbstractArray, mergeseq::AbstractArray; tracemerge::Bool = false)
```