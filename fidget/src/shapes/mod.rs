use crate::context::Tree;
use std::sync::Arc;

#[cfg(feature = "rhai")]
use rhai::{CustomType, TypeBuilder};

/// Represents a point in 2D space
///
/// The members are unnamed because it's not necessarily `XY` space; it may be
/// remapped by a set of `Axes`.
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "rhai", derive(CustomType), rhai_type(extra = Self::build_extra))]
pub struct Vec2(pub f64, pub f64);

#[cfg(feature = "rhai")]
impl Vec2 {
    fn build_extra(builder: &mut TypeBuilder<Self>) {
        builder.with_fn("vec2", Self);
    }
}

#[cfg(feature = "rhai")]
impl FromDynamic for Vec2 {
    fn from_dynamic(d: rhai::Dynamic) -> Option<Self> {
        d.clone().try_cast::<Vec2>().or_else(|| {
            let array = d.into_array().ok()?;
            if array.len() == 2 {
                let x = f64::from_dynamic(array[0].clone())?;
                let y = f64::from_dynamic(array[1].clone())?;
                Some(Vec2(x, y))
            } else {
                None
            }
        })
    }
}

/// Represents a point in 3D space
///
/// The members are unnamed because it's not necessarily `XYZ` space; it may be
/// remapped by a set of `Axes`.
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "rhai", derive(CustomType), rhai_type(extra = Self::build_extra))]
pub struct Vec3(pub f64, pub f64, pub f64);

#[cfg(feature = "rhai")]
impl Vec3 {
    fn build_extra(builder: &mut TypeBuilder<Self>) {
        builder.with_fn("vec3", Self);
    }
}

#[cfg(feature = "rhai")]
impl FromDynamic for Vec3 {
    fn from_dynamic(d: rhai::Dynamic) -> Option<Self> {
        d.clone().try_cast::<Vec3>().or_else(|| {
            let array = d.into_array().ok()?;
            if array.len() == 2 {
                let x = f64::from_dynamic(array[0].clone())?;
                let y = f64::from_dynamic(array[1].clone())?;
                let z = f64::from_dynamic(array[2].clone())?;
                Some(Vec3(x, y, z))
            } else {
                None
            }
        })
    }
}

macro_rules! define_shape {
    (
        $(#[$meta:meta])*
        pub struct $StructName:ident {
            $(
                $shape_name:ident: Arc<dyn ShapeBuilder>,
            )?
            $(
                $(#[$field_meta:meta])*
                pub $field_name:ident: $field_type:ty = $default:expr,
            )*

            fn build(&$self:ident) -> Tree {
                $($build:stmt);*
            }
        }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Debug)]
        #[cfg_attr(feature = "rhai", derive(CustomType), rhai_type(extra = Self::build_extra))]
        pub struct $StructName {
            $(
                /// Shape to be transformed
                pub $shape_name: Arc<dyn ShapeBuilder>,
            )?
            $(
                $(#[$field_meta])*
                pub $field_name: $field_type,
            )*
        }

        impl ShapeBuilder for $StructName {
            fn build(&$self) -> Tree {
                $($build)*
            }
            fn to_arc(&self) -> Arc<dyn ShapeBuilder> {
                Arc::new(self.clone())
            }
        }

        // Rhai-specific bindings to make scripting more ergonomic
        #[cfg(feature = "rhai")]
        #[allow(clippy::self_named_constructors)]
        impl $StructName {
            $(
                fn $field_name(
                    ctx: rhai::NativeCallContext,
                    mut this: Self,
                    value: rhai::Dynamic
                ) -> Result<Self, Box<rhai::EvalAltResult>> {
                    this.$field_name = <$field_type>::from_dynamic(
                        value.clone()
                    ).ok_or_else(||
                        Box::new(rhai::EvalAltResult::ErrorMismatchDataType(
                            stringify!($field_type).to_string(),
                            value.type_name().to_string(),
                            ctx.position())
                        ))?;
                    Ok(this)
                }
            )*

            fn build_extra(builder: &mut TypeBuilder<Self>) {
                // Register constructor function (lower-case)
                builder.with_fn(
                    paste::paste! { stringify!( [< $StructName:snake >] ) },
                    |$(
                        ctx: rhai::NativeCallContext,
                        $shape_name: rhai::Dynamic
                    )?| -> Result<Self, Box<rhai::EvalAltResult>> {
                        $(
                            let $shape_name =
                                ctx.call_native_fn("to_arc", ($shape_name,))?;
                        )?
                        Ok(Self {
                            $( $shape_name, )?
                            $( $field_name: $default, )*
                        })
                    }
                );
            }

            pub(crate) fn register(engine: &mut rhai::Engine) {
                engine.build_type::<$StructName>();
                engine.register_fn("to_arc", |c: $StructName| c.to_arc());
                // Register all of the builders as functions as well
                $(
                    engine.register_fn(
                        stringify!($field_name),
                        $StructName::$field_name
                    );
                )*
            }
        }
    };
}

define_shape! {
    /// A 2D circle
    #[derive(Copy)]
    pub struct Circle {
        /// Radius of the circle
        pub radius: f64 = 1.0,
        /// Center of the circle
        pub center: Vec2 = Vec2(0.0, 0.0),

        fn build(&self) -> Tree {
            let (x, y, _z) = Tree::axes();
            ((x - self.center.0).square()
            + (y - self.center.1).square())
            .sqrt() - self.radius
        }
    }
}

/// Shapes used in modeling
pub trait ShapeBuilder: Send + Sync + std::fmt::Debug {
    /// Bind a shape to a particular set of axes
    fn build(&self) -> Tree;

    /// Converts yourself to an `Arc<dyn ShapeBuilder>`
    ///
    /// This is used in bindings to Rhai scripts
    fn to_arc(&self) -> Arc<dyn ShapeBuilder>;
}

#[cfg(feature = "rhai")]
trait FromDynamic
where
    Self: Sized,
{
    fn from_dynamic(d: rhai::Dynamic) -> Option<Self>;
}

#[cfg(feature = "rhai")]
impl FromDynamic for f64 {
    fn from_dynamic(d: rhai::Dynamic) -> Option<Self> {
        d.clone()
            .try_cast::<f64>()
            .or_else(|| d.try_cast::<i64>().map(|f| f as f64))
    }
}

define_shape! {
    /// Uniform scaling
    pub struct Scale {
        shape: Arc<dyn ShapeBuilder>,
        /// Center about which we are scaling
        pub center: Vec3 = Vec3(0.0, 0.0, 0.0),
        /// Scale factor to apply
        pub scale: f64 = 1.0,

        fn build(&self) -> Tree {
            let (x, y, z) = Tree::axes();
            self.shape.build().remap_xyz(
                (x - self.center.0) * self.scale + self.center.0,
                (y - self.center.1) * self.scale + self.center.1,
                (z - self.center.2) * self.scale + self.center.2)
        }
    }
}

define_shape! {
    /// Uniform scaling
    pub struct Move {
        shape: Arc<dyn ShapeBuilder>,
        /// X translation
        pub dx: f64 = 0.0,
        /// Y translation
        pub dy: f64 = 0.0,
        /// Z translation
        pub dz: f64 = 0.0,

        fn build(&self) -> Tree {
            let (x, y, z) = Tree::axes();
            self.shape.build().remap_xyz(
                x - self.dx,
                y - self.dy,
                z - self.dz,
            )
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "rhai")]
pub(crate) fn register_types(engine: &mut rhai::Engine) {
    engine.build_type::<Vec2>();
    engine.build_type::<Vec3>();

    Circle::register(engine);
    Scale::register(engine);
    Move::register(engine);
}
