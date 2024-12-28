# 0.3.5 (unreleased)
- Added `#[derive(Serialize, Deserialize)]` to `View2` and `View3`
- Make `TranslateHandle` take a `const N: usize` parameter
- Use `TranslateHandle` in `View2` (previously, it was only used in `View3`)
- Make `translate` and `rotate` functions borrow their respective handle,
  instead of taking it by value.
- Fix docstring for `AndRegImm`, `AndRegReg`, `OrRegImm`, and `OrRegReg`

# 0.3.4
- Add `GenericVmFunction::simplify_with` to simultaneously simplify a function
  and pick a new register count for the resulting tape
- Add bidirectional conversions between `JitFunction` and `GenericVmFunction`
  (exposing the inner data member)
- Add a new `TileSizes(Vec<usize>)` object representing tile sizes used during
  rendering.  Unlike the previous `Vec<usize>`, this data type checks our tile
  size invariants at construction.  It is used in the `struct RenderConfig`.
- Rethink rendering and viewport configuration:
    - Add a new `RegionSize<const N: usize>` type (with `ImageSize` and
      `VoxelSize` aliases), representing a render region.  This type is
      responsible for the screen-to-world transform
    - Add `View2` and `View3` types, which stores the world-to-model
      transform (scaling, panning, etc)
    - Image rendering uses both `RegionSize` and `ViewX`; this means that we
      can now render non-square images!
    - Meshing uses just a `View3`, to position the model within the ±1 bounds
    - The previous `fidget::shape::Bounds` type is removed
    - Remove `fidget::render::render2d/3d` from the public API, as they're
      equivalent to the functions on `ImageRenderConfig` / `VoxelRenderConfig`
- Move `RenderHints` into `fidget::render`
- Remove fine-grained features from `fidget` crate, because we aren't actually
  testing the power-set of feature combinations in CI (and some were breaking!).
  The only remaining features are `rhai`, `jit` and `eval-tests`.
- Add new `ShapeVars<T>` type, representing a map from `VarIndex -> T`.  This
  type is used for high-level rendering and meshing of `Shape` objects that
  include supplementary variables
- Add `Octree::build_with_vars` and `Image/VoxelRenderConfig::run_with_vars`
  functions for shapes with supplementary variables
- Change `ShapeBulkEval::eval_v` to take **single** variables (i.e. `x`, `y`,
  `z` vary but each variable has a constant value across the whole slice).  Add
  `ShapeBulkEval::eval_vs` if `x`, `y`, `z` and variables are _all_ changing
  through the slices.
- Add a new `GenericVmTape<N>` type, and use it for VM evaluation.  Previously,
  the `GenericVmFunction<N>` type implemented both `Tape` and `Function`.
- Add `vars()` to `Function` trait, because there are cases where we want to get
  the variable map without building a tape (and it must always be the same).
- Fix soundness bug in `Mmap` (probably not user-visible)
- Add `Send + Sync + Clone` bounds to the `trait Tape`, to make them easily
  shared between threads.  Previously, we used an `Arc<Tape>` to share tapes
  between threads, but tapes were _already_ using an `Arc<..>` under the hood.
- Changed `Tape::recycle` from returning a `Storage` to returning an
  `Option<Storage>`, as tapes may now be shared between threads.
- Use Rayon for 2D and 3D rasterization
    - The `threads` member of `VoxelRenderConfig` and `ImageRenderConfig` is now
      a `Option<ThreadPool>`, which can be `None` (use a single thread),
      `Some(ThreadPool::Global)` (use the global Rayon pool), or
      `Some(ThreadPool::Custom(..))` (use a user-provided pool)
    - This is a step towards WebAssembly multithreading, using
      `wasm-bindgen-rayon`.
    - `ThreadCount` is moved to `fidget::mesh`, because that's the only place
      it's now used
        - The plan is to switch to Rayon for meshing as well, eventually
- Tweak `View2` and `View3` APIs to make them more useful as camera types

# 0.3.3
- `Function` and evaluator types now produce multiple outputs
    - `MathFunction::new` now takes a slice of nodes, instead of a single node
    - All of the intermediate tape formats (`SsaTape`, etc) are aware of
      multiple output nodes
    - Evaluation now returns a slice of outputs, one for each root node (ordered
      based on order in the `&[Node]` slice passed  to `MathFunction::new`)
- `RegisterAllocator` no longer binds SSA register 0 to physical register 0 by
  default. If you don't know what this means, don't worry about it.

# 0.3.2
- Added `impl IntoNode for Var`, to make handling `Var` values in a context
  easier.
- Added `impl From<TreeOp> for Tree` for convenience
- Added `Context::export(&self, n: Node) -> Tree` to make a freestanding `Tree`
  given a context-specific `Node`.
- Fix possible corruption of `x24` during AArch64 float slice JIT evaluation,
  due to incorrect stack alignment.
- Added `Context::deriv` and `Tree::deriv` to do symbolic differentiation of
  math expressions.

# 0.3.1
The highlight of this release is the `fidget::solver` module, which implements
the Levenberg-Marquardt algorithm to minimize a system of equations (represented
as `fidget::eval::Function` objects).  It's our first official case of using
Fidget's types and traits for things _other than_ pure implicit surfaces!

- Fixed a bug in the x86 JIT which could corrupt registers during gradient
  (`grad_slice`) evaluation
- Renamed `Context::const_value` to `Context::get_const` and tweaked its return
  type to match `Context::get_var`.
- Added `impl From<i32> for Tree` to make writing tree expressions easier
- Removed `Error::ReservedName` and `Error::DuplicateName`, which were unused
- Add the `fidget::solver` module, which contains a simple solver for systems of
  equations.  The solver requires the equations to implement Fidget's `Function`
  trait.  It uses both point-wise and gradient evaluation to solve for a set of
  `Var` values, using the Levenberg-Marquardt algorithm.
- Add `Tree::var()` and `impl TryFrom<Tree> for Var`

# 0.3.0
- Major refactoring of core evaluation traits
    - The lowest-level "thing that can be evaluated" trait has changed from
      `Shape` (taking `(x, y, z)` inputs) to `Function` (taking an arbitrary
      number of variables).
    - `Shape` is now a wrapper around a `F: Function` instead of a trait.
    - Shape evaluators are now wrappers around `E: BulkEvaluator` or `E:
      TracingEvaluator`, which convert `(x, y, z)` arguments into
      list-of-variables arguments.
    - Using the `VmShape` or `JitShape` types should be mostly the same as
      before; changes are most noticeable if you're writing things that are
      generic across `S: Shape`.
- Major refactoring of how variables are handled
    - Removed `VarNode`; the canonical variable type is `Var`, which is its own
      unique index.
    - Removed named variables, to make `Var` trivially `Copy + Clone`.
    - Added `vars()` method to `Function` trait, allowing users to look up the
      mapping from variable to evaluation index.  A `Var` now represents a
      persistent identity from `Tree` to `Context` to `Function` evaluation.
    - Move `Var` and `VarMap` into `fidget::vars` module, because they're no
      longer specific to a `Context`.
    - `Op::Input` now takes a `u32` argument, instead of a `u8`
- Fixed a bug in the AArch64 JIT would which could corrupt certain registers
  during interval evaluation.

# 0.2.7
This release brings us to opcode parity with `libfive`'s operators, adding
`atan2` and various rounding operations.  In addition, there are a few new APIs
and rearrangements in preparation for larger refactoring to come.

- Changed to 2D rendering API to support render modes which use linear
  interpolation to process full / empty regions
    - Specifically, `RenderMode::interval` now returns an `IntervalAction`,
      which can be `Fill(..)`, `Recurse`, or `Interpolate`.
    - Modify `SdfRenderMode` use this interpolation; the previous pixel-perfect
      behavior is renamed to `SdfPixelRenderModel`
    - Make `RenderMode` trait methods static, because they weren't using `&self`
    - Change signature of `fidget::render::render2d` to pass the mode only as a
      generic parameter, instead of an argument
- Add new operations: `floor`, `ceil`, `round`, `atan2`
- Changed `BulkEvaluator::eval` signature to take x, y, z arguments as `&[T]`
  instead of `&[f32]`.  This is more flexible for gradient evaluation, because
  it allows the caller to specify up to three gradients, without pinning them to
  specific argument.
- Moved `tile_sizes_2d`, `tile_sizes_3d`, and `simplify_tree_during_meshing`
  into a new `fidget::shape::RenderHits` trait.  This is a building block for
  generic (_n_-variable) function evaluation.

# 0.2.6
This is a relatively small release; there are a few features to improve the
WebAssembly demo, bug fixes and improvements for very deep `Tree` objects, and
one more public API.

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
  vertices far from the desired position.
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
