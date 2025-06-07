//! GLSL-style vector types
//!
//! We use dedicated types (instead of `nalgebra` types) because we must derive
//! `Facet` on them, so are limited by the orphan rule.
use facet::Facet;

/// 2D position
#[derive(Copy, Clone, Facet)]
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
#[derive(Copy, Clone, Facet)]
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
#[derive(Copy, Clone, Facet)]
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
