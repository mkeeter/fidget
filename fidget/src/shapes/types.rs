//! Types used in shape construction
//!
//! This module includes both GLSL-style `Vec` types and higher-level
//! representations of modeling concepts (e.g. [`Axis`]).
//!
//! We use dedicated types (instead of `nalgebra` types) because we must derive
//! `Facet` on them, so are limited by the orphan rule.
use facet::{ConstTypeId, Facet};
use strum::IntoDiscriminant;

use crate::context::Tree;

/// Error type for type construction
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// Vector is too short to convert to an axis
    #[error("vector is too short to convert to an axis (length: {0})")]
    TooShort(f64),

    /// Vector is too long to convert to an axis
    #[error("vector is too long to convert to an axis (length: {0})")]
    TooLong(f64),

    /// Could not normalize vector due to an invalid length
    #[error("could not normalize vector due to an invalid length")]
    BadLength,

    /// Wrong type
    #[error("wrong type; expected {expected}, got {actual}")]
    WrongType {
        /// Expected type
        expected: Type,
        /// Actual type
        actual: Type,
    },
}

/// 2D position
#[derive(Copy, Clone, Debug, PartialEq, Facet)]
#[allow(missing_docs)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl From<nalgebra::Vector2<f64>> for Vec2 {
    fn from(value: nalgebra::Vector2<f64>) -> Self {
        Self {
            x: value.x,
            y: value.y,
        }
    }
}

impl From<Vec2> for nalgebra::Vector2<f64> {
    fn from(value: Vec2) -> Self {
        Self::new(value.x, value.y)
    }
}

impl From<f64> for Vec2 {
    fn from(value: f64) -> Self {
        Self { x: value, y: value }
    }
}

impl Vec2 {
    /// Builds a new `Vec2` from `x, y` coordinates
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
    /// Returns the L2-norm
    pub fn norm(&self) -> f64 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }
    fn combine<F: Fn(f64, f64) -> f64>(self, rhs: Self, f: F) -> Self {
        Self {
            x: f(self.x, rhs.x),
            y: f(self.y, rhs.y),
        }
    }
    fn map<F: Fn(f64) -> f64>(self, f: F) -> Self {
        Self {
            x: f(self.x),
            y: f(self.y),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

/// 3D position
#[derive(Copy, Clone, Debug, PartialEq, Facet)]
#[allow(missing_docs)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl From<nalgebra::Vector3<f64>> for Vec3 {
    fn from(value: nalgebra::Vector3<f64>) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}

impl From<Vec3> for nalgebra::Vector3<f64> {
    fn from(value: Vec3) -> Self {
        Self::new(value.x, value.y, value.z)
    }
}

impl From<f64> for Vec3 {
    fn from(value: f64) -> Self {
        Self {
            x: value,
            y: value,
            z: value,
        }
    }
}

impl Vec3 {
    /// Builds a new `Vec3` from `x, y, z` coordinates
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    /// Returns the L2-norm
    pub fn norm(&self) -> f64 {
        (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
    }
    fn combine<F: Fn(f64, f64) -> f64>(self, rhs: Self, f: F) -> Self {
        Self {
            x: f(self.x, rhs.x),
            y: f(self.y, rhs.y),
            z: f(self.z, rhs.z),
        }
    }
    fn map<F: Fn(f64) -> f64>(self, f: F) -> Self {
        Self {
            x: f(self.x),
            y: f(self.y),
            z: f(self.z),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

/// 4D position (`xyzw`)
#[derive(Copy, Clone, Debug, PartialEq, Facet)]
#[allow(missing_docs)]
pub struct Vec4 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl From<nalgebra::Vector4<f64>> for Vec4 {
    fn from(value: nalgebra::Vector4<f64>) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z: value.z,
            w: value.w,
        }
    }
}

impl From<Vec4> for nalgebra::Vector4<f64> {
    fn from(value: Vec4) -> Self {
        Self::new(value.x, value.y, value.z, value.w)
    }
}

impl From<f64> for Vec4 {
    fn from(value: f64) -> Self {
        Self {
            x: value,
            y: value,
            z: value,
            w: value,
        }
    }
}

impl Vec4 {
    fn combine<F: Fn(f64, f64) -> f64>(self, rhs: Self, f: F) -> Self {
        Self {
            x: f(self.x, rhs.x),
            y: f(self.y, rhs.y),
            z: f(self.z, rhs.z),
            w: f(self.w, rhs.w),
        }
    }
    fn map<F: Fn(f64) -> f64>(self, f: F) -> Self {
        Self {
            x: f(self.x),
            y: f(self.y),
            z: f(self.z),
            w: f(self.w),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

macro_rules! impl_binary {
    ($ty:ident, $op:ident, $base_fn:ident) => {
        impl std::ops::$op<$ty> for $ty {
            type Output = $ty;

            fn $base_fn(self, rhs: $ty) -> Self {
                self.combine(rhs, |a, b| a.$base_fn(b))
            }
        }
        impl std::ops::$op<$ty> for f64 {
            type Output = $ty;
            fn $base_fn(self, rhs: $ty) -> $ty {
                $ty::from(self).$base_fn(rhs)
            }
        }
        impl std::ops::$op<f64> for $ty {
            type Output = $ty;
            fn $base_fn(self, rhs: f64) -> $ty {
                self.$base_fn($ty::from(rhs))
            }
        }
    };
    ($ty:ident, $base_fn:ident, $f:expr) => {
        pub fn $base_fn<R>(self, rhs: R) -> Self
        where
            $ty: From<R>,
        {
            self.combine($ty::from(rhs), $f)
        }
    };
    ($ty:ident, $base_fn:ident) => {
        impl_binary!($ty, $base_fn, |a, b| a.$base_fn(b));
    };
}

macro_rules! impl_unary {
    ($ty:ident, $base_fn:ident, $f:expr) => {
        pub fn $base_fn(self) -> Self {
            self.map($f)
        }
    };
    ($ty:ident, $base_fn:ident) => {
        impl_unary!($ty, $base_fn, |a| a.$base_fn());
    };
}

macro_rules! impl_all {
    ($ty:ident) => {
        impl_binary!($ty, Add, add);
        impl_binary!($ty, Mul, mul);
        impl_binary!($ty, Sub, sub);
        impl_binary!($ty, Div, div);

        #[allow(missing_docs)]
        impl $ty {
            impl_binary!($ty, min);
            impl_binary!($ty, max);
            impl_unary!($ty, sqrt);
            impl_unary!($ty, abs);
        }
    };
}

impl_all!(Vec2);
impl_all!(Vec3);
impl_all!(Vec4);

////////////////////////////////////////////////////////////////////////////////

/// Normalized 3D axis (of length 1)
#[derive(Copy, Clone, Debug, PartialEq, Facet)]
pub struct Axis(Vec3);

impl TryFrom<Vec3> for Axis {
    type Error = Error;
    fn try_from(value: Vec3) -> Result<Self, Self::Error> {
        let norm = value.norm();
        if norm.is_nan() {
            Err(Error::BadLength)
        } else if norm < 1e-8 {
            Err(Error::TooShort(norm))
        } else if norm > 1e8 {
            Err(Error::TooLong(norm))
        } else {
            Ok(Self(value / norm))
        }
    }
}

impl Axis {
    /// Returns the axis vector
    pub fn vec(&self) -> &Vec3 {
        &self.0
    }
    /// The X axis
    pub const X: Self = Axis(Vec3 {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    });
    /// The Y axis
    pub const Y: Self = Axis(Vec3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    });
    /// The Z axis
    pub const Z: Self = Axis(Vec3 {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    });
}

/// Unoriented plane in 3D space, specified as an axis + offset
#[derive(Copy, Clone, Debug, PartialEq, Facet)]
pub struct Plane {
    /// Axis orthogonal to the plane
    pub axis: Axis,
    /// Offset relative to the origin
    pub offset: f64,
}

impl Plane {
    /// The XY plane
    pub const XY: Self = Plane {
        axis: Axis::Y,
        offset: 0.0,
    };
    /// The YZ plane
    pub const YZ: Self = Plane {
        axis: Axis::X,
        offset: 0.0,
    };
    /// The ZX plane
    pub const ZX: Self = Plane {
        axis: Axis::Y,
        offset: 0.0,
    };
}

////////////////////////////////////////////////////////////////////////////////

/// Enumeration representing all types that can be used in shapes
#[derive(Debug, strum::EnumDiscriminants)]
#[strum_discriminants(name(Type), derive(enum_map::Enum), allow(missing_docs))]
#[allow(missing_docs)]
pub enum Value {
    Float(f64),
    Vec2(Vec2),
    Vec3(Vec3),
    Vec4(Vec4),
    Axis(Axis),
    Plane(Plane),
    Tree(Tree),
    VecTree(Vec<Tree>),
}

impl Value {
    /// Puts the type into an in-progress builder at a particular index
    ///
    /// # Panics
    /// If the currently-selected builder field does not match our type
    pub(crate) fn put<'facet, 'shape>(
        self,
        builder: &mut facet::Partial<'facet, 'shape>,
        i: usize,
    ) {
        match self {
            Value::Float(v) => builder.set_nth_field(i, v),
            Value::Vec2(v) => builder.set_nth_field(i, v),
            Value::Vec3(v) => builder.set_nth_field(i, v),
            Value::Vec4(v) => builder.set_nth_field(i, v),
            Value::Axis(v) => builder.set_nth_field(i, v),
            Value::Plane(v) => builder.set_nth_field(i, v),
            Value::Tree(v) => builder.set_nth_field(i, v),
            Value::VecTree(v) => builder.set_nth_field(i, v),
        }
        .unwrap();
    }
}

macro_rules! try_from_type {
    ($ty:ty, $name:ident) => {
        impl<'a> TryFrom<&'a Value> for &'a $ty {
            type Error = $crate::shapes::types::Error;
            fn try_from(v: &'a Value) -> Result<&'a $ty, Self::Error> {
                if let Value::$name(f) = v {
                    Ok(f)
                } else {
                    Err(Self::Error::WrongType {
                        expected: Type::$name,
                        actual: v.discriminant(),
                    })
                }
            }
        }
    };
    ($ty:ident) => {
        try_from_type!($ty, $ty);
    };
}

try_from_type!(f64, Float);
try_from_type!(Vec2);
try_from_type!(Vec3);
try_from_type!(Vec4);
try_from_type!(Tree);
try_from_type!(Plane);
try_from_type!(Axis);
try_from_type!(Vec<Tree>, VecTree);

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Type::Float => "f64",
            Type::Vec2 => "Vec2",
            Type::Vec3 => "Vec3",
            Type::Vec4 => "Vec4",
            Type::Axis => "Axis",
            Type::Plane => "Plane",
            Type::Tree => "Tree",
            Type::VecTree => "Vec<Tree>",
        };
        write!(f, "{}", s)
    }
}

/// Convert from a Facet type id to a tag
impl TryFrom<facet::ConstTypeId> for Type {
    type Error = facet::ConstTypeId;
    fn try_from(t: facet::ConstTypeId) -> Result<Self, Self::Error> {
        if t == ConstTypeId::of::<f64>() {
            Ok(Self::Float)
        } else if t == ConstTypeId::of::<Vec2>() {
            Ok(Self::Vec2)
        } else if t == ConstTypeId::of::<Vec3>() {
            Ok(Self::Vec3)
        } else if t == ConstTypeId::of::<Vec4>() {
            Ok(Self::Vec4)
        } else if t == ConstTypeId::of::<Axis>() {
            Ok(Self::Axis)
        } else if t == ConstTypeId::of::<Plane>() {
            Ok(Self::Plane)
        } else if t == ConstTypeId::of::<Tree>() {
            Ok(Self::Tree)
        } else if t == ConstTypeId::of::<Vec<Tree>>() {
            Ok(Self::VecTree)
        } else {
            Err(t)
        }
    }
}

impl Type {
    /// Executes a default builder function for the given type
    ///
    /// # Safety
    /// `f` must be a builder for the type associated with this tag
    pub unsafe fn build_from_default_fn(
        &self,
        f: unsafe fn(facet::PtrUninit) -> facet::PtrMut,
    ) -> Value {
        unsafe {
            match self {
                Type::Float => Value::Float(eval_default_fn(f)),
                Type::Vec2 => Value::Vec2(eval_default_fn(f)),
                Type::Vec3 => Value::Vec3(eval_default_fn(f)),
                Type::Vec4 => Value::Vec4(eval_default_fn(f)),
                Type::Axis => Value::Axis(eval_default_fn(f)),
                Type::Plane => Value::Plane(eval_default_fn(f)),
                Type::Tree => Value::Tree(eval_default_fn(f)),
                Type::VecTree => Value::VecTree(eval_default_fn(f)),
            }
        }
    }
}

/// Evaluates a default builder function, returning a value
///
/// # Safety
/// `f` must be a builder for type `T`
pub unsafe fn eval_default_fn<T>(
    f: unsafe fn(facet::PtrUninit) -> facet::PtrMut,
) -> T {
    let mut v = std::mem::MaybeUninit::<T>::uninit();
    let ptr = facet::PtrUninit::new(&mut v);
    // SAFETY: `f` must be a builder for type `T`
    unsafe { f(ptr) };
    // SAFETY: `v` is initialized by `f`
    unsafe { v.assume_init() }
}

/// Checks whether `T`'s fields are all [`Type`]-compatible.
pub(crate) fn validate<T: Facet<'static>>() -> facet::StructType<'static> {
    let facet::Type::User(facet::UserType::Struct(s)) = T::SHAPE.ty else {
        panic!("must be a struct-shaped type");
    };
    for f in s.fields {
        if Type::try_from(f.shape().id).is_err() {
            panic!("unknown type: {}", f.shape());
        }
    }
    s
}
