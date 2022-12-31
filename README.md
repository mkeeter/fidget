# Fidget
![Build & Test Github Actions badge](https://github.com/mkeeter/fidget/actions/workflows/rust.yml/badge.svg)
[![docs.rs badge](https://img.shields.io/docsrs/fidget?label=docs.rs)](https://docs.rs/fidget/)
[![crates.io badge](https://img.shields.io/crates/v/fidget)](https://crates.io/crates/fidget)

Fidget is experimental infrastructure for complex closed-form implicit surfaces:

- Manipulation and deduplication of math expressions
- Conversion from graphs into straight-line code ("tapes") for evaluation
- Tape simplification, based on interval evaluation results
- A _very fast_ JIT compiler, with hand-written AArch64 routines for
    - Point-wise evaluation (`f32`)
    - Interval evaluation (`[lower, upper]`)
    - SIMD evaluation (`f32 x 4`)
    - Gradient evaluation (partial derivatives with respect to x, y, and z)
- Bitmap rendering of implicit surfaces in 2D (with a variety of rendering
  modes) and 3D (producing heightmaps and normals).

If this all sounds oddly familiar, it's because you've read
[Massively Parallel Rendering of Complex Closed-Form Implicit Surfaces](https://www.mattkeeter.com/research/mpr/).
Fidget includes all of the building blocks from that paper, but with an emphasis
on (native) evaluation on the CPU, rather than (interpreted) evaluation on the
GPU.

The library has [extensive documentation](https://docs.rs/fidget/latest/fidget/),
including a high-level overview of the APIs in the crate-level docs; this is a
great place to get started!

At the moment, it has strong Lego-kit-without-a-manual energy: there are lots of
functions that are individually documented, but putting them together into
something useful is left as an exercise to the reader.  There may also be some
missing pieces, and the API seams may not be in the right places; if you're
doing serious work with the library, expect to fork it and make local
modifications.

Issues and PRs are welcome, although I'm unlikely to merge anything which adds
substantial maintenance burden.  This is a personal-scale experimental project,
so adjust your expectations accordingly.

## Demo applications
In the [repository on Github](https://github.com/mkeeter/fidget), there are
two demo applications:

- `demo` does bitmap rendering and unscientific benchmarking from the command
  line
- `gui` is a minimal GUI for interactive exploration

These are deliberately not published to [https://crates.io](crates.io), because
they're demo applications and not complete end-user tools.

## Platforms
At the moment, the JIT only supports macOS + AArch64.

Disabling the `jit` feature allows for cross-platform rendering, using an
interpreter rather than JIT compilation.

If you want to write some x86 assembly (and who doesn't?!), adding that backend
would be a fun project and highly welcome!

## License
© 2022-2023 Matthew Keeter  
Released under the [Mozilla Public License 2.0](https://github.com/mkeeter/fidget/blob/main/LICENSE.txt)
