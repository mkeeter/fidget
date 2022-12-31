# Fidget
![Build & Test Github Actions badge](https://github.com/mkeeter/fidget/actions/workflows/rust.yml/badge.svg)

Fidget is experimental infrastructure for complex closed-form implicit surfaces.

It is **not ready for public use**, but is published to
[crates.io](https://crates.io) to claim the package name.

(As such, I'd appreciate if people didn't share it to news aggregators or post
about it on Twitter.  If you feel an overwhelming urge to talk about it, feel
free to [reach out directly](https://mattkeeter.com/about))

That being said, it already includes a bunch of functionality:

- Manipulation and deduplication of math expressions
- Conversion from graphs into straight-line code ("tapes") for evaluation
- Tape simplification, based on interval evaluation results
- A _very fast_ JIT compiler, with hand-written AArch64 routines for
    - Point-wise evaluation (`f32`)
    - Interval evaluation (`[lower, upper]`)
    - SIMD evaluation (`f32 x 4`)
    - Gradient evaluation (partial derivatives with respect to x, y, and z)

These building blocks are used in an implementation of bitmap rendering.

If this all sounds oddly familiar, it's because you've read
[Massively Parallel Rendering of Complex Closed-Form Implicit Surfaces](https://www.mattkeeter.com/research/mpr/).
Fidget includes all of the building blocks from that paper, but with an emphasis
on (native) evaluation on the CPU, rather than (interpreted) evaluation on the
GPU.

## Crate features
The project is based on the `fidget` crate, with three relevant features

- `render` builds `fidget::render`, which includes functions to render 2D and
  3D images.
- `rhai` builds [Rhai](https://rhai.rs/) bindings
- `jit` builds the JIT compiler

By default, all of these features are enabled.

In the [repository on Github](https://github.com/mkeeter/fidget), there are
two demo applications:

- `demo` does bitmap rendering and unscientific benchmarking from the command
  line
- `gui` is a minimal GUI for interactive exploration

These are deliberately not published to [https://crates.io](crates.io), because
they're demo applications and not complete end-user tools.

## Platforms
At the moment, only macOS (AArch64) is fully supported.

Disabling the `jit` feature should allow for cross-platform rendering
(albeit without the JIT compiler), but this is untested.
