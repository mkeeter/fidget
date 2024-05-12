# 0.2.6
- Added `VmShape` serialization (using `serde`), specifically
    - `#[derive(Serialize, Deserialize)}` on `VmData`
    - `impl From<VmData<255>> for VmShape { .. }`
- Fixed stack overflows when handling very deep `Tree` objects
    - Added a non-recursive `Drop` implementation
    - Rewrote `Context::import` to use the heap instead of stack
- Updated `Context::import` to cache the `TreeOp → Node` mapping, which is a
  minor optimization (probably only relevant for unreasonably large `Tree`
  objects)
- Made `fidget::render::RenderHandle` public and documented

# 0.2.5
The highlight of this release is native Windows support (including JIT
compilation).  Everything should work out of the box; please open an issue if
things are misbehaving.

There are a handful of API and feature changes as well:

- Removed `threads` parameter from various `Settings` objects when targeting
  `wasm32`, so that it's harder to accidentally spawn threads (which would
  panic).
- Removed `min_depth` / `max_depth` distinction in meshing; this doesn't
  actually guarantee manifold models
  (see [this scale-invariant adversarial model](https://www.mattkeeter.com/blog/2023-04-23-adversarial/)),
  so the extra complexity isn't worth it.
- Added Windows support (including JIT)
- Removed `write-xor-execute` feature, which was extra cognitive load that no
  one had actually asked for; if you care about it, let me know!

# 0.2.4
The highlight of this release is a refactoring of how shapes are handled in Rhai
scripts: instead of being built from thunks (unevaluated closures) and evaluated
in `draw(..)` and `draw_rgb(..)`, they are built directly using a new `Tree`
object.  This makes writing scripts more ergonomic:

```rust
// Before
fn circle(cx, cy, r) {
   |x, y| {
       sqrt((x - cx) * (x - cx) +
            (y - cy) * (y - cy)) - r
   }
}

// After
fn circle(cx, cy, r) {
    let ax = axes();
    sqrt((ax.x - cx) * (ax.x - cx) +
         (ax.y - cy) * (ax.y - cy)) - r
}
```

The change is even more noticeable on higher-level functions (functions which
operate on shapes):

```rust
// Before
fn intersection(a, b) {
    |x, y| {
        max(a.call(x, y), b.call(x, y))
    }
}

// After
fn intersection(a, b) {
    a.max(b)
}
```

## Detailed changelog
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
- Improved robustness of `viewer` application when editors move files instead of
  writing to them directly.
- Add Rhai bindings for all new opcodes
- Tweak JIT calling convention to take an array of inputs instead of X, Y, Z
  arguments.

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
