# 0.2.4 (unreleased)
- Add helper function `Context::if_nonzero_else` to build conditionals (using
  the logical operators added in version 0.2.3)
- Add `fidget::context::{Tree, TreeOp}`.  These types allow construction of math
  trees without a parent `Context` (and therefore without deduplication); the
  resulting trees can be loaded into a `Context` using `Context::import`.  This
  replaces the `BoundNode` and `BoundContext` types (previously only available
  for unit tests).
- Remove `Context::remap_xyz` in favor of lazy remapping with `Tree::remap_xyz`.
- Use `Tree` in Rhai scripts, removing the per-script `Context`.  This is an API
  change: previously, shapes in Rhai were thunks evaluated during calls to
  `draw(..)`; now, shapes in Rhai are `Tree` objects, and there's much less
  function wrangling required.
- Remove the distinction between variables and inputs (by removing all
  `Var`-related functions and opcodes).  This was over-engineered; the plan
  going forward will be to support functions with _N_ inputs (currently fixed to
  3, `x`/`y`/`z`).

# 0.2.3
- Fix a possible panic during multithreaded 3D rendering of very small images
- Add `compare` operator (equivalent to `<=>` in C++ or `partial_cmp` in Rust,
  with the difference that unordered results are returned as `NAN`)
- Fix a bug in the x86 JIT evaluator's implementation of interval `abs`
- Add generic `TransformedShape<S>`, representing a shape transformed by a 4x4
  homogeneous matrix.  This replaces `RenderConfig::mat` as the flexible
  strategy for rotation / scale / translation / perspective transforms, e.g. for
  interactive visualization (where you don't want to remap the underlying shape)
- Introduce a new `Bounds` type, representing an X/Y/Z region of interest for
  rendering or meshing.  This overlaps somewhat with `TransformedShape`, but
  it's ergonomic to specify render region instead of having to do the matrix
  math every time.
    - Replaced `RenderConfig::mat` with a new `bounds` member.
    - Added a new `bounds` member to `mesh::Settings`, for octree construction
- Move `Interval` and `Grad` to `fidget::types` module, instead of
  `fidget::eval::types`.
- Fix an edge case in meshing where nearly-planar surfaces could produce
  vertexes far from the desired position.
- Add the `modulo` (Euclidean remainder) operation
- Add logical operations (`and`, `or`, `not`), which can be used to build
  pseudo-conditionals which are simplified by the `TracingEvaluator`.

# 0.2.2
- Added many transcendental functions: `sin`, `cos`, `tan`, `asin`, `acos`,
  `atan`, `exp`, `ln`
- Implemented more rigorous testing of evaluators, fixed a bunch of edge cases
  (mostly differences in `NAN` handling between platforms)
- Tweaks to register allocator, for a small performance improvement

# 0.2.1
- Changed `fidget::eval::Vars` to borrow instead of use an `Arc`
- Properly pre-allocated `mmap` regions based on estimated size
- Bump dependencies, fixing a Dependabot warning about `atty` being unmaintained
- Fixed performance regressions in 3D rendering
- Added `wasm32-unknown-unknown` target to CI checks
- Reordered load/store generation so that register allocation generates linear
  code ("linear" in the type-system sense).
- Small optimizations to register allocation
- Added unit tests for rendering correctness

# 0.2.0: The Big Refactoring
This release includes a significant amount of reorganization and refactoring.
There are two main sets of changes, along with some minor tweaks.

## Completely revamped evaluation traits
Previously, evaluation was tightly coupled to the `Tape` type, and individual
evaluators were tightly bound to a specific tape.  Upon reflection, both of
these decisions were reconsidered:

- We may want to use Fidget's algorithms on implicit models that aren't defined
  as tapes, so the evaluation traits should be more generic
- The previous code put a great deal of effort into reusing evaluator
  workspaces, but it was hard; it turns out that it's much cleaner to have
  evaluators own their workspaces but **not** own their tapes

These changes can be seen in the new `fidget::eval::Shape` trait, which replaces
(but is not equivalent to) the previous `fidget::eval::Family` trait.

## Compiler reorganization
Compiler modules are reorganized to live under `fidget::compiler` instead of
being split between `fidget::ssa` and `fidget::vm`.

## Other small changes
- Improve meshing quality (better vertex placement, etc)
- Add parallelism to meshing implementation, configured by the new
  `fidget::mesh::Settings`.
- Added documentation to `fidget::mesh` modules

# 0.1.4
- Added support for `aarch64-unknown-linux-*` to the JIT compiler; previously,
  `aarch64` was only supported on macOS.
- Added initial meshing support in the `fidget::mesh` module, gated by the
  `mesh` feature (enabled by default).

# 0.1.3
- Added `x86_64` backend for the JIT compiler

# 0.1.0 - 0.1.2
- Initial release to [crates.io](https://crates.io/crates/fidget)
- Several point releases to get [docs.rs](https://docs.rs/fidget/latest/fidget/) working
