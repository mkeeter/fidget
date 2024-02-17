# 0.2.0: The Big Refactoring (unreleased)
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
