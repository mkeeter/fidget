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

These building blocks are used in an implementation of bitmap rendering.
