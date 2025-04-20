//! Documentation for writing Rhai scripts with Fidget bindings
//!
//! # Trees
//! The basic type for math expressions is a `Tree`, which is equivalent to
//! [`fidget::context::Tree`](crate::context::Tree).  Trees are typically built
//! from `(x, y, z)` primitives, which can be constructed with the `axes()`
//! function:
//!
//! ```
//! # use fidget::rhai::Engine;
//! # let mut e = Engine::new();
//! # e.run("
//! let xyz = axes();
//! xyz.x + xyz.y
//! # ").unwrap();
//! ```
//!
//! `x, y, z` variables are also automatically injected into the `Engine`'s
//! context before evaluation.
//!
//! # Vector types
//! The Rhai context includes `vec2`, `vec3`, and `vec4` types, which are
//! roughly analogous to their GLSL equivalents (for floating-point only).
//! Many (but not all) functions are implemented and overloaded for these types;
//! if you encounter something missing, feel free to open an issue.
//!
//! # Shapes
//! In Rhai scripts, shapes can be constructed using object map notation:
//! ```
//! # use fidget::rhai::Engine;
//! # let mut e = Engine::new();
//! # e.run("
//! circle(#{ center: vec2(1.0, 2.0), radius: 3.0 })
//! # ").unwrap();
//! ```
//!
//! This works for any object type; in addition, there are a bunch of ergonomic
//! improvements on top of this low-level syntax.
//!
//! ## Type coercions
//! Shapes are built from a set of Rust primitives, with generous conversions
//! from Rhai's native types:
//!
//! - Scalar values (`f64`)
//!     - Both floating-point and integer Rhai values will be accepted
//! - Vectors (`vec2` and `vec3`)
//!     - These may be explicitly constructed with `vec2(x, y)` and
//!       `vec3(x, y, z)`
//!     - Appropriately-sized arrays of numbers will be automatically converted
//!     - A `vec2` (or something convertible into a `vec2`) will be converted
//!       into a `vec3` with a `z` value of 0.
//! ```
//! # use fidget::rhai::Engine;
//! # let mut e = Engine::new();
//! # e.run("
//! // array -> vec2
//! let c = circle(#{ center: [1, 2], radius: 3 });
//!
//! // array -> vec3
//! let s = sphere(#{ center: [1, 2, 4], radius: 3 });
//!
//! // array -> vec2 -> vec3
//! move(#{ shape: c, offset: [1, 1] });
//! # ").unwrap();
//! ```
//!
//! ## Uniquely typed functions
//! Any shape with unique arguments may skip the object map and pass arguments
//! directly; order doesn't matter, because the type is unambiguous.
//!
//! ```
//! # use fidget::rhai::Engine;
//! # let mut e = Engine::new();
//! # e.run("
//! // array -> vec2
//! let c1 = circle([1, 2], 3);
//! let c2 = circle(3, [1, 2]); // order doesn't matter!
//! # ").unwrap();
//! ```
//!
//! Note that some kinds of type coercion will not work in this regime, e.g.
//! `vec2 -> vec3`:
//!
//! ```should_panic
//! # use fidget::rhai::Engine;
//! # let mut e = Engine::new();
//! # e.run("
//! // array -> vec2
//! let c1 = sphere([1, 1], 4); // vec2 -> vec3 conversion
//! # ").unwrap();
//! ```
//!
//! ## Function chaining
//! Shapes with a single initial `Tree` member are typically transforms (e.g.
//! `move` from above).  These functions may be called with the tree as their
//! first (unnamed) argument, followed by an object map of remaining parameters.
//!
//! ```
//! # use fidget::rhai::Engine;
//! # let mut e = Engine::new();
//! # e.run("
//! let c = circle(#{ center: [1, 2], radius: 3 });
//! move(c, #{ offset: [1, 1] });
//! # ").unwrap();
//! ```
//!
//! Given Rhai's dispatch strategy, this can also be written as a function
//! chain, which is more ergonomic for a string of transforms:
//!
//! ```
//! # use fidget::rhai::Engine;
//! # let mut e = Engine::new();
//! # e.run("
//! circle(#{ center: [1, 2], radius: 3 })
//!     .move(#{ offset: [1, 1] });
//! # ").unwrap();
//! ```
//!
//! A transform which only take a single argument may skip the object map:
//!
//! ```
//! # use fidget::rhai::Engine;
//! # let mut e = Engine::new();
//! # e.run("
//! circle(#{ center: [1, 2], radius: 3 })
//!     .move([1, 1]);
//! # ").unwrap();
//! ```
//!
//! ## Functions of two trees
//! Shapes which take two trees can be called with two (unnamed) arguments:
//!
//! ```
//! # use fidget::rhai::Engine;
//! # let mut e = Engine::new();
//! # e.run("
//! let a = circle(#{ center: [0, 0], radius: 1 });
//! let b = circle(#{ center: [1, 0], radius: 0.5 });
//! difference(a, b);
//! # ").unwrap();
//! ```
//!
//! ## Tree reduction functions
//! Any function which takes a single `Vec<Tree>` will accept both an array of
//! trees or individual tree arguments (up to an 8-tuple).
//!
//! ```
//! # use fidget::rhai::Engine;
//! # let mut e = Engine::new();
//! # e.run("
//! let a = circle(#{ center: [1, 1], radius: 3 });
//! let b = circle(#{ center: [2, 2], radius: 3 });
//! let c = circle(#{ center: [3, 3], radius: 3 });
//! union([a, b, c]);
//! union(a, b, c);
//! union(a, b, c, a, b, c, a, b);
//! # ").unwrap();
//! ```
//!
//! ## Automatic tree reduction
//! Any shape which takes a `Tree` will also accept an array of trees, which are
//! automatically reduced with a union operation.
//!
//! ```
//! # use fidget::rhai::Engine;
//! # let mut e = Engine::new();
//! # e.run("
//! [
//!     circle(#{ center: [0, 0], radius: 3 }),
//!     circle(#{ center: [2, 2], radius: 3 }),
//! ]
//! .move(#{ offset: [1, 1] });
//! # ").unwrap();
//! ```
