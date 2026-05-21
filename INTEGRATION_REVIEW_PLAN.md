# Integration Review Plan
<!-- Hand-authored in the /review-integration plan schema. Edit freely, but preserve chunk IDs and status values. -->

## Metadata
- **Kind**: `integration`
- **Package**: NMFMerge
- **Source review date**: 2026-05-21
- **Current version**: 1.1.1
- **Issue**: n/a

## Stated values
The goal of this plan is to make NMFMerge compilable into a C-ABI shared
library by `juliac --trim`, so it can be called from Python/Matlab/R (the
`teh/wrappers` branch). Guiding preferences:

- Fail-fast over silent continuation: surfacing errors matters more than
  papering over them. Trimming strips error-message machinery, which is in
  tension with this — flag, do not silently accept, any loss of diagnosability.
- Keep the flexible interactive `nmfmerge` API intact for Julia users; the
  compiled library may use a separate, deliberately monomorphic entrypoint.
- Upstream trim-cleanliness fixes (NMF, TSVD, NonNegLeastSquares) are
  preferred over local vendored patches where the maintainers will accept them.
- Annotate arguments only as specifically as the implementation requires;
  `::Function` and `::Matrix{Float64}` over-restriction is a smell.

## Release strategy
- **Pre-breaking-release**: `n/a (no breaking changes planned)` — the
  recommended approach (CHUNK-003) adds a new internal entrypoint rather than
  altering `nmfmerge`. Revisit if CHUNK-003 chooses to refactor the public API.
- **Inter-cluster releases**: `n/a`

## Background: the `--trim` build

A first `juliac --trim=safe` build was run 2026-05-21 with Julia 1.13.0-rc1
(`julia +rc`; `juliac` at `~/.julia/bin/juliac`). It loaded every dependency,
ran the trim verifier, and emitted a clean `build/nmfmerge_abi.json` — then
failed with **92 verifier errors**, followed by a `juliac` crash in its own
failure path (`UndefVarError(:exit)` in `Base.Compiler`).

Build command (run from the package root):

```bash
juliac --output-lib build/libNMFMerge --export-abi build/nmfmerge_abi.json \
  --project language_wrappers --compile-ccallable --trim=safe --experimental \
  language_wrappers/lib.jl
```

Build harness gotcha: `juliac` copies the `--project` into `/tmp`, so a
relative `[sources]` path-dep (`NMFMerge = {path = ".."}`) cannot be resolved.
`language_wrappers/Manifest.toml` was hand-edited to an absolute path as a
workaround; `language_wrappers/Project.toml` is unchanged (relative). Re-running
the build needs that Manifest edit (or an equivalent fix).

Error taxonomy (the 92):

- **~75% are diagnostic output**, dynamically dead but statically reachable:
  NMF's verbose `Printf` iteration tables (~55), `NonNegLeastSquares.warn`
  (8), TSVD's `biLanczos` `debug` `println` (~4).
- **Constant propagation is lost through `NMF.nnmf`'s keyword interface.**
  NMFMerge calls `nnmf(X, n2; alg, kwargs..., init=:custom, …)`; the
  `kwargs...` splat routed through `Core.kwcall` defeats constprop of `init`
  and `alg`, so the verifier compiles *every* init method (`randinit`,
  `nndsvd`, `spa`) and *every* algorithm (ProjectedALS, ALSPGrad, MultUpdate,
  CoordinateDescent, GreedyCD) — ~6× the necessary surface. Evidence: error #1
  is `randinit` (the `:random` path, never used); errors reference both
  `GreedyCDUpd` and `ProjectedALSUpd`.
- **Genuine type instabilities** (minority): NMF's `zeros(T::DataType,…)` /
  `rand(T::DataType,…)` return `Matrix`, not `Matrix{T}`; GsvdInitialization's
  `Kronecker`+`SparseArrays` `hvcat`/`vcat` and `nonneg_lsq`'s `:fnnls`
  Symbol-keyword dispatch.
- **NMFMerge's own code** contributes a small cluster: `colmerge2to1pq` /
  `pqupdate2to1!` (`src/NMFMerge.jl:127`, `:137`, `:140`), including the
  `::Function` annotation on `pqupdate2to1!`'s `queuepenalty` argument.
- One error is a `juliac` limitation: `Core.invoke_in_world` — "trim
  verification not yet implemented for builtin".

## Decisions
<!-- Answers to `decide` chunks land here, with the chunk ID. -->

## Chunks

### CHUNK-001: preflight
- **Kind**: `preflight`
- **Description**: Establish baseline — `git status` clean, `Pkg.test()` passes, `Test.detect_ambiguities(NMFMerge)` count, current `Project.toml` version. Additionally re-run the `--trim` build (see Background) and record the verifier error count as the trim baseline. Record all results in this chunk's Notes.
- **Status**: `not-started`
- **Notes**:

### CHUNK-002: decide-algorithm-to-pin
- **Kind**: `decide`
- **Description**: The compiled library must commit to a single NMF algorithm (the cause of the constprop blow-up is forwarding `alg` as a runtime value). `lib.jl` currently lets it default to `:greedycd`; NMFMerge's own tests almost all use `:cd` (CoordinateDescent). Decide which algorithm the compiled entrypoint pins. Recommendation: `:cd`, matching the tested path and `runtests.jl:121`.
- **Status**: `not-started`
- **Notes**:

### CHUNK-003: decide-entrypoint-shape
- **Kind**: `decide`
- **Description**: Decide whether to (A) refactor the public `nmfmerge` to drop the `nnmf`/`kwargs...` path, or (B) add a separate internal monomorphic entrypoint (e.g. `_nmfmerge_compiled`) that `lib.jl` calls, leaving `nmfmerge` untouched. Recommendation: B — keeps the flexible API for interactive Julia users and keeps this plan non-breaking. Choosing A makes CHUNK-005 `Breaking: yes` and requires appending a `version-bump` chunk.
- **Status**: `not-started`
- **Notes**:

### CHUNK-004: decide-upstream-vs-local
- **Kind**: `decide`
- **Description**: The trim-cleanliness fixes for NMF, TSVD, NonNegLeastSquares, and GsvdInitialization (CHUNK-010 .. CHUNK-013) live in other repos. Decide per-package: submit an upstream PR, or carry a local patch (dev'd package / fork pinned in `language_wrappers`). Recommendation: upstream PRs for all four — they are mechanical and broadly useful; fall back to a pinned fork only if a maintainer is unresponsive.
- **Status**: `not-started`
- **Notes**:

### CHUNK-005: monomorphic-nmf-path
- **Kind**: `implement`
- **Description**: Give NMFMerge a monomorphic NMF path that does not go through `NMF.nnmf`'s keyword interface. Replace the four `nnmf(...)` calls in `nmfmerge` (`src/NMFMerge.jl:47,52,56` and the initial call) with direct `NMF.solve!(NMF.CoordinateDescent{Float64}(; maxiter, tol, …), X, W, H)` calls (per the pinned algorithm from CHUNK-002), performing `:custom` initialization by hand and dropping the `kwargs...` splat. Shape per CHUNK-003: new internal entrypoint (B) or in-place refactor (A). This is the highest-leverage chunk — it should prune five of six algorithms and the unused init methods. Verification: `Pkg.test()` still passes, and the `--trim` build's error count drops sharply.
- **Status**: `not-started`
- **Notes**:
- **Depends on**: CHUNK-002, CHUNK-003

### CHUNK-006: fix-nmfmerge-own-trim-errors
- **Kind**: `implement`
- **Description**: Fix NMFMerge's own verifier errors in `colmerge2to1pq` / `pqupdate2to1!` (`src/NMFMerge.jl:110-153`). Remove the abstract `::Function` annotation on `pqupdate2to1!`'s `queuepenalty` argument; stabilize the merge-loop result types and the `reduce(hcat, …)` at `:140`. Verification: `Pkg.test()` passes; these errors gone from the trim log.
- **Status**: `not-started`
- **Notes**:

### CHUNK-007: modernize-makefile
- **Kind**: `implement`
- **Description**: `language_wrappers/Makefile` uses the obsolete `juliac` interface (`--output-o`, a `.a` archive, `bindinginfo_*.log`). Rewrite it around the current flow: `juliac --output-lib --export-abi --compile-ccallable --trim=safe --experimental` (see Background for the exact command). Resolve the relative-`[sources]`-path issue robustly rather than relying on the hand-edited Manifest.
- **Status**: `not-started`
- **Notes**:

### CHUNK-008: harden-lib-entrypoint
- **Kind**: `implement`
- **Description**: `language_wrappers/lib.jl`'s `nmfmerge_inplace` validates `Wout.rows == X.rows` and `Hout.cols == X.cols` but not `Wout.cols` / `Hout.rows` against the component count — a mis-sized output throws `DimensionMismatch` across the C ABI. Add the missing checks and decide an error-propagation convention (status field / error code) so failures surface as something a Python/C caller can inspect rather than a process abort. Reference the fail-fast value in Stated values.
- **Status**: `not-started`
- **Notes**:

### CHUNK-009: reinventory-residual-errors
- **Kind**: `investigate`
- **Description**: After the NMFMerge-side fixes, re-run the `--trim` build and produce an accurate inventory of the residual errors, attributed per upstream package. Use it to tighten the scope (and dependency order) of CHUNK-010 .. CHUNK-013 — some may shrink dramatically once only the pinned algorithm's code path is compiled. Record the residual count and per-package breakdown in Notes.
- **Status**: `not-started`
- **Notes**:
- **Depends on**: CHUNK-005, CHUNK-006

### CHUNK-010: trim-clean-nmf
- **Kind**: `implement`
- **Description**: Make NMF.jl trim-clean for the pinned algorithm's path: guard the verbose `Printf` iteration-table output so it is statically eliminable when `verbose=false`, and fix the `zeros(T::DataType,…)` / `rand(T::DataType,…)` type instability in `randinit` (`NMF/src/initialization.jl:5-7`). Delivery per CHUNK-004.
- **Status**: `not-started`
- **Notes**:
- **Depends on**: CHUNK-009, CHUNK-004
- **Cluster**: upstream-trim

### CHUNK-011: trim-clean-tsvd
- **Kind**: `implement`
- **Description**: Make TSVD.jl trim-clean: the `debug` `println` in `biLanczos` (`TSVD/src/svd.jl:177`) is statically reachable. Guard it so it is eliminable when `debug=false`. Delivery per CHUNK-004.
- **Status**: `not-started`
- **Notes**:
- **Depends on**: CHUNK-009, CHUNK-004
- **Cluster**: upstream-trim

### CHUNK-012: trim-clean-nonneglsq
- **Kind**: `implement`
- **Description**: Make NonNegLeastSquares.jl trim-clean: the `warn(...)` calls on unrecognized algorithm variants are statically reachable. Guard or restructure them. Delivery per CHUNK-004.
- **Status**: `not-started`
- **Notes**:
- **Depends on**: CHUNK-009, CHUNK-004
- **Cluster**: upstream-trim

### CHUNK-013: fix-gsvdinit-type-instability
- **Kind**: `implement`
- **Description**: Fix GsvdInitialization's genuine type instabilities surfaced by the verifier: the `Kronecker`+`SparseArrays` `hvcat`/`vcat` path in `gram_sp_C`, and the `nonneg_lsq` `Core.kwcall((alg=:fnnls, gram=true), …)` Symbol-keyword dispatch (`GsvdInitialization/src/GsvdInitialization.jl:~128,199-203`). These are algorithmic, not diagnostic — they need real type-stability work. Delivery per CHUNK-004.
- **Status**: `not-started`
- **Notes**:
- **Depends on**: CHUNK-009, CHUNK-004
- **Cluster**: upstream-trim

### CHUNK-014: report-juliac-bugs
- **Kind**: `investigate`
- **Description**: Report two `juliac`/Julia-1.13-rc findings upstream (JuliaC.jl / julia): (1) the crash in `juliac`'s failure path — `UndefVarError(:exit)` in `Base.Compiler` after the verifier reports errors; (2) `Core.invoke_in_world` — "trim verification not yet implemented for builtin". Record issue links in Notes.
- **Status**: `not-started`
- **Notes**:

### CHUNK-015: end-to-end-verification
- **Kind**: `investigate`
- **Description**: Terminal verification: confirm the `--trim` build completes cleanly and emits `libNMFMerge` + `nmfmerge_abi.json`; run that JSON through JuliaLibWrapping (`write_wrapper` for `CTarget` and `PythonTarget`); smoke-test the generated C header and Python package against the compiled library. Record what works and any remaining friction.
- **Status**: `not-started`
- **Notes**:
- **Depends on**: CHUNK-005, CHUNK-006, CHUNK-007, CHUNK-008, CHUNK-010, CHUNK-011, CHUNK-012, CHUNK-013

## Session ledger
<!-- The implementer appends one line after each session: `- YYYY-MM-DD CHUNK-XXX (name) → next: CHUNK-YYY` -->

## Open Questions
- Coordination: once CHUNK-010 .. CHUNK-013 land as upstream releases, NMFMerge's
  `[compat]` bounds must be bumped to require the trim-clean versions, and
  NMFMerge itself likely needs a release. Sequence this after the
  `upstream-trim` cluster completes.
- The compiled library carries the whole Julia runtime; shipping it to users
  without Julia (the actual Python/Matlab/R distribution story) is out of scope
  for this plan — track it with the JuliaLibWrapping work instead.
