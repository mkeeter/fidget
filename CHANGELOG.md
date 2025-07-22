# 0.3.9 (unreleased)
- Fix panic in tape construction if a multi-output expression has a constant as
  an output.
- Add `Function::can_simplify` to check whether a `Function` can _ever_ be
  simplified (if not, then interval evaluation isn't as useful).
- Fix invalid access when doing bulk evaluation on a `Function` with more than
  one output.  This caused a panic using the VM evaluator, and a segfault
  (typically) using the JIT evaluator.
- Make `PartialEq` for `Tree` objects do deep comparisons, instead of shallow
  (pointer) equality
    - This is more expensive, but matches typical data structures in Rust
    - `Tree::ptr_eq` can be used to perform shallow equality checks
- Changed 3D image rendering to saturate when voxels are touching the camera.
  Saturated voxels snap to the image's depth and have a normal of `[0, 0, 1]`.
- Replaced `view` in `ImageRenderConfig` and `VoxelRenderConfig` with a generic
  `world_to_model` matrix, for more flexibility when rendering.
- Add mathematical constants for Rhai scripts (`PI`, `E`, `TAU`...)

# 0.3.8
- Bug fix: `Image::height()` was returning width instead!
- Add `#[derive(PartialEq)]` to `View2` and `View3`
- Improve rendering at small images sizes
    - Previously, we rendered at least one tile of size `cfg.tile_sizes[0]`
    - Now, we pick the smallest possible tile size for the root tiles; if we're
      rendering a 32×32 image with tile sizes of `[128, 32, 8]`, then we'll
      render a single 32×32 tile (instead of 128×128)
    - As part of this change, a few functions were removed from the `TileSizes`
      public API; they're now attached to an internal `struct TileSizesRef`.
- Add a new `trait ShapeVisitor` and `pub fn visit_shapes(..)` for introspection
  into shapes and transforms defined in the `shapes` module.  This is expected
  to be useful for automatically generated scripted bindings (indeed,
  `fidget::rhai` now uses it).
- Added more document to `fidget::shapes`
- Made `fidget::rhai::shapes::register_shape` private
- Significant changes to 2D rendering API:
    - Changed `fidget::render::render2d` to always return an
      `Image<DistancePixel>`, which encodes either a distance sample or details
      on fill at that point.
    - Removed the `trait RenderMode`, which was previously used to generate
      pixel samples from a distance field.
    - Added `ImageRenderConfig::pixel_perfect`, which forces sampling down to
      individual pixels.
    - Added `fidget::render::effects::{to_rgba_bitmap, to_debug_bitmap,
      to_rgba_distance}` to post-process an `Image<DistancePixel>`.  These
      replace the previous `RenderMode` objects for generating specific flavors
      of image.
    - Note that **approximate SDFs are entirely removed**; they were not
      substantially faster, added complexity, and looked bad.
- Add `#[facet(default = ...)]` annotations to relevant fields in
  `fidget::shapes`.
    - These annotations are used in the Rhai bindings to build shapes without
      specifying every field, e.g. `circle(1)` leaves the center as a default
      value.
- Fix issue where `Shape::eval_*` functions would return an empty slice if there
  were no active variables in the shape; it now returns a slice that's the
  correct size (i.e. matching the input slices).
- In Rhai bindings, `FromDynamic` now takes the field's default value as a hint,
  which may be used when promoting other types.  For example, if you pass a
  `vec2` (instead of a `vec3`) to both `Move` and `Scale`, the `z` component
  will be set to 0 for `Move` and 1 for `Scale`.
- Add helper functions to destructure and rebuild `Canvas2`, `Canvas3`, `View2`,
  and `View3` into their component values.
- Rename `fidget::{shapes, rhai}::vec` to `fidget::{shapes, rhai}::types`
- Add more common types used in shapes and Rhai bindings:
    - `Axis` is a unit `Vec3`
    - `Plane` is an `Axis` and offset
- Add `Reflect` shape, as well as `ReflectX/Y/Z`
- **Removed** `fidget::rhai::Engine` and `fidget::rhai::Axes`
    - Previously, `fidget::rhai::Engine` wrapped a Rhai engine and
      Fidget-specific functionality
    - The internals of `Engine` were tightly coupled to `fidget-viewer`, in ways
      that weren't useful for other GUIs.
    - Now, `fidget::rhai::engine()` returns a `rhai::Engine` object with Fidget
      types pre-installed, but without the wrapper type.
    - The Rhai `draw(..)` and `draw_rgb(..)` functions are moved into
      `fidget-viewer`, because they're specific to that one GUI (`fidget-cli` is
      also aware of `draw`, but now has its own simpler implementation).
- Move `Vec2/3/4`, `Axis`, `Plane` into `fidget::shapes::types` instead of
  re-exporting them in `fidget::shapes`.
- Make `fidget::shapes::types::{Value, Type}` public; these represent types
  which can be used in shapes (with ergonomic Rhai bindings).
- Fix missing local optimizations in `Context::import` (e.g. `x * 0 => 0`)
- Rename `remap_xyz(..)` to just `remap(..)` in Rhai bindings; add a
  two-argument version which leaves `z` unchanged

# 0.3.7
- Small release to fix an issue with 0.3.6 being published with invalid local
  changes (I thought `cargo` prevented this, dunno how it happened)
- Mark functions on `Interval` and `Grad` as `#[inline]`, to improve performance
  when those types are used outside of the Fidget crate itself.
- Make `fidget::rhai` submodules visible (`vec`, `shape`, `tree`) for
  finer-grained usage outside of Fidget.
- Update to Rust 2024 edition, set minimum `rustc` version to 1.87
- Update dependencies; remove some that have become unused

# 0.3.6
- Change `Option<ThreadPool<'a>>` to `Option<&'a ThreadPool>` throughout the
  codebase; moving the reference out of the `ThreadPool` eliminates the need for
  a separate `rayon::ThreadPool` object on the stack.
- Significant rewrite of meshing!  It now uses the same `Option<&ThreadPool>`
  type and is multithreaded using Rayon, meaning it can work in WebAssembly.
- Changed 3D rendering and effects functions to use a new `GeometryBuffer` type,
  which combines depth and normal data into a single image.
- Add `fidget::gui` module, which defines `Canvas2` and `Canvas3`. The canvas
  types are stateful abstractions around a GUI canvas, with support for cursor
  interactions.
- Change `ImageSize::transform_point` and `VoxelSize::transform_point` to take
  a point with `i32` coordinates (instead of `f32`).  This helps us distinguish
  between screen (pixel) and world (floating-point) coordinates at the type
  level.
- Add `Tree::remap_affine` (and `TreeOp::RemapAffine`) to perform affine
  transformations on math expressions.  These transformations are composable;
  two affine transforms will be combined into a single transform if stacked
  together.
- Major updates to the Rhai standard library and default bindings:
    - Add `vec2`, `vec3`, `vec4` types
    - Shapes are now constructed with object maps
    - Added documentation in `fidget::rhai::docs` module

# 0.3.5
- Added `#[derive(Serialize, Deserialize)]` to `View2` and `View3`
- Make `TranslateHandle` take a `const N: usize` parameter
- Use `TranslateHandle` in `View2` (previously, it was only used in `View3`)
- Make `translate` and `rotate` functions borrow their respective handle,
  instead of taking it by value.
- Fix docstring for `AndRegImm`, `AndRegReg`, `OrRegImm`, and `OrRegReg`
- Add `cancel: CancelToken` to 2D and 3D rendering configuration objects; this
  is a shared `Arc<AtomicBool>` which can be used to stop rendering.  The
  returned type is now an `Option<...>`, where `None` indicates that rendering
  was cancelled.
- Fix inconsistency between JIT and VM evaluator when performing interval
  evaluation of `not([NAN, NAN])`.
- Propagate `NAN` values through `and` and `or` operations on intervals.
- Add a new `Image<P>` generic image type (wrapping a `Vec<P>`, `width`, and
  `height`).
    - Define `DepthImage`, `NormalImage`, and `ColorImage` specializations
    - Use these types in 2D and 3D rendering
- Remove `Grad::to_rgb` in favor of handling it at the image level
- Add `fidget::render::effects` module for post-processing rendered images:
    - Combining depth and normal images into a shaded image
    - Denoising normals to fix back-facing samples
    - Computing and applying screen-space ambient occlusion
- Optimize implementation of interval `modulo` for cases where the right-hand
  argument is a positive constant value (which is the most common when using it
  for domain repetition)
- Update many dependencies to their latest versions

## WebAssembly building notes
Due to [`getrandom#504`](https://github.com/rust-random/getrandom/pull/504),
crates which use Fidget as a library **and** compile to WebAssembly must
select a `getrandom` backend.  This can be done either on the command line
(`RUSTFLAGS='--cfg getrandom_backend="wasm_js"'`) or in a `.cargo/config.toml`
configuration file (e.g. [this file](https://github.com/mkeeter/fidget/tree/main/.cargo/config.toml)
in Fidget itself).

See
[the `getrandom` docs](https://docs.rs/getrandom/latest/getrandom/#webassembly-support)
for more details on why this is necessary.

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
