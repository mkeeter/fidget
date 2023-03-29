# Fidget
[![» Crate](https://badgen.net/crates/v/fidget)](https://crates.io/crates/fidget)
[![» Docs](https://badgen.net/badge/api/docs.rs/df3600)](https://docs.rs/fidget/)
[![» CI](https://badgen.net/github/checks/mkeeter/fidget/main)](https://github.com/mkeeter/fidget/actions/workflows/rust.yml)
[![» MPL-2.0](https://badgen.net/github/license/mkeeter/fidget)](LICENSE.txt)

Fidget is experimental infrastructure for complex closed-form implicit surfaces.

At the moment, it is **quietly public**: it's available on Github and published
to [crates.io](https://crates.io/crates.fidget), but I'd appreciate if you
refrain from posting it to Hacker News / Twitter / etc; I'm planning to write an
overview blog post and put together a few demo applications before making a
larger announcement. If you have an overwhelming urge to talk about it,
[feel free to reach out directly](https://mattkeeter.com/about)!

The library contains a variety of data structures and algorithms, e.g.

- Manipulation and deduplication of math expressions
- Conversion from graphs into straight-line code ("tapes") for evaluation
- Tape simplification, based on interval evaluation results
- A _very fast_ JIT compiler, with hand-written `aarch64` and `x86_64` routines
  for
    - Point-wise evaluation (`f32`)
    - Interval evaluation (`[lower, upper]`)
    - SIMD evaluation (`f32 x 4` on ARM, `f32 x 8` on x86)
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

- `demo` does bitmap rendering from the command line
- `viewer` is a minimal GUI for interactive exploration

These are deliberately not published to [https://crates.io](crates.io), because
they're demo applications and not complete end-user tools.

## Platforms
At the moment, the JIT supports three platforms:

- `aarch64-apple-darwin`
- `aarch64-unknown-linux-*`
- `x86_64-unknown-linux-*`

`aarch64` platforms require NEON instructions and `x86_64` platforms require
AVX2 support; both of these extensions are nearly a decade old and should be
widespread.

Disabling the `jit` feature allows for cross-platform rendering, using an
interpreter rather than JIT compilation.

`x86_64-pc-windows-*` and `aarch64-pc-windows-*` _may_ be close to working (with
only minor tweaks required); the author does not have a Windows machine on which
to test.

## Similar projects
Fidget overlaps with various projects in the implicit modeling space:

- [Antimony: CAD from a parallel universe](https://mattkeeter.com/projects/antimony)*
- [`libfive`: Infrastructure for solid modeling](https://libfive.com)*
- [Massively Parallel Rendering of Complex Closed-Form Implicit Surfaces (MPR)](https://github.com/mkeeter/mpr)*
- [ImplicitCAD: Powerful, Open-Source, Programmatic CAD](https://implicitcad.org/)
- [Ruckus: Procedural CAD For Weirdos](https://docs.racket-lang.org/ruckus/index.html)
- [Curv: a language for making art using mathematics](https://github.com/curv3d/curv)
- [sdf: Simple SDF mesh generation in Python](https://github.com/fogleman/sdf)
- [Forged Thoughts: A Modeling & Rendering Language in Rust](https://forgedthoughts.com/)
- Probably more; PRs welcome!

*written by the same author

(the MPR paper also cites
[many references](https://dl.acm.org/doi/10.1145/3386569.3392429#sec-ref)
to related academic work)

Compared to these projects, Fidget is unique in having a native JIT **and**
using that JIT while performing tape simplification.  Situating it among
projects by the same author – which all use roughly the same rendering
strategies – it looks something like this:

|                 | CPU               | GPU
|-----------------|-------------------|------
| **Interpreter** | `libfive`, Fidget | MPR
| **JIT**         | Fidget            | (please give me APIs to do this)

Fidget's native JIT makes it _blazing fast_.
For example, here are rough benchmarks rasterizing [this model](https://www.mattkeeter.com/projects/siggraph/depth_norm@2x.png)
across three different implementations:

Size  | `libfive` | MPR     | Fidget (VM) | Fidget (JIT)
------|-----------|---------|-------------|---------------
1024³ | 66.8 ms   | 22.6 ms | 61.7 ms     | 23.6 ms
1536³ | 127 ms    | 39.3 ms | 112 ms      | 45.4 ms
2048³ | 211 ms    | 60.6 ms | 184 ms      | 77.4 ms

`libfive` and Fidget are running on an M1 Max CPU; MPR is running on a GTX 1080
Ti GPU.  We see that Fidget's interpreter is slightly better than `libfive`, and
Fidget's JIT is _nearly_ competitive with the GPU-based MPR.

Fidget is missing a bunch of features that are found in more mature projects.
For example, it only includes a debug GUI, and its meshing is much less
battle-tested than `libfive`.

## License
© 2022-2023 Matthew Keeter  
Released under the [Mozilla Public License 2.0](https://github.com/mkeeter/fidget/blob/main/LICENSE.txt)
