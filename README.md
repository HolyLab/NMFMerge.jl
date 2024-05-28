# NMFMerge

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://HolyLab.github.io/NMFMerge.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://HolyLab.github.io/NMFMerge.jl/dev/)
[![Build Status](https://github.com/HolyLab/NMFMerge.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/HolyLab/NMFMerge.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/HolyLab/NMFMerge.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/HolyLab/NMFMerge.jl)

This package includes the code the paper ``
It is used for merging components in non-negative matrix factorization.
All of the examples below assume you've loaded the package with `using FlyThroughPaths`.


## Functions

**colmerge2to1pq**colmerge2to1pq(W, H, n)

This function merges components in ``W`` and ``H`` (columns in ``W`` and rows in ``H``) from original number of components to ``n`` components (``n``columns and rows left in ``W`` and ``H`` respectively) 

