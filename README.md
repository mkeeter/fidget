# Fidget
Fidget is experimental infrastructure for complex closed-form implicit surfaces.

Right now, it includes a few fundamental building blocks:

- Manipulation and deduplication of math expressions
- Conversion from graphs into straight-line code ("tapes") for evaluation
- Tape simplification, based on interval evaluation results
- A _very fast_ JIT compiler, with hand-written AArch64 routines for
    - Point-wise evaluation (`f32`)
    - Interval evaluation (`[lower, upper]`)
    - SIMD evaluation (`f32 x 4`)
    - Gradient evaluation (partial derivatives with respect to x, y, and z)

These building blocks are used in an implementation of bitmap rendering.

## Crate organization
The project is split into multiple crates to improve compile times:

- `fidget-core`: Construction and evaluation of expressions
- `fidget-jit`: JIT compilation (AArch64 only)
- `fidget-render`: 2D and 3D rendering
- `fidget-rhai`: Rhai bindings

All of these `crates` are re-exported by the `fidget` omnibus crate.
