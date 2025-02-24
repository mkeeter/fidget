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

/// A 3D coordinate space
pub struct Axes(pub Tree, pub Tree, pub Tree);

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

            fn build(&$self:ident, $axes:ident: &Axes) -> Tree {
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
            fn build(&$self, $axes: &Axes) -> Tree {
                $($build)*
            }
            fn to_arc(&self) -> Arc<dyn ShapeBuilder> {
                Arc::new(self.clone())
            }
        }

        // Rhai-specific bindings to make scripting more ergonomic
        #[cfg(feature = "rhai")]
        impl $StructName {
            $(
                fn $field_name(mut self, value: $field_type) -> Self {
                    self.$field_name = value;
                    self
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
    /// A circle in 2D space
    #[derive(Copy)]
    pub struct Circle {
        /// Radius of the circle
        pub radius: f64 = 1.0,
        /// Center of the circle
        pub center: Vec2 = Vec2(0.0, 0.0),

        fn build(&self, axes: &Axes) -> Tree {
            ((axes.0.clone() - self.center.0).square()
                + (axes.1.clone() - self.center.1).square())
            .sqrt()
                - self.radius
        }
    }
}

/// Shapes used in modeling
pub trait ShapeBuilder: Send + Sync + std::fmt::Debug {
    /// Bind a shape to a particular set of axes
    fn build(&self, axes: &Axes) -> Tree;

    /// Converts yourself to an `Arc<dyn ShapeBuilder>`
    ///
    /// This is used in bindings to Rhai scripts
    fn to_arc(&self) -> Arc<dyn ShapeBuilder>;
}

define_shape! {
    /// Uniform scaling
    pub struct Scale3d {
        shape: Arc<dyn ShapeBuilder>,
        /// Center about which we are scaling
        pub center: Vec3 = Vec3(0.0, 0.0, 0.0),
        /// Scale factor to apply
        pub scale: f64 = 1.0,

        fn build(&self, axes: &Axes) -> Tree {
            let a =
                (axes.0.clone() - self.center.0) * self.scale + self.center.0;
            let b =
                (axes.1.clone() - self.center.1) * self.scale + self.center.1;
            let c =
                (axes.2.clone() - self.center.2) * self.scale + self.center.2;

            self.shape.build(&Axes(a, b, c))
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "rhai")]
pub(crate) fn register_types(engine: &mut rhai::Engine) {
    engine.build_type::<Vec2>();
    engine.build_type::<Vec3>();

    Circle::register(engine);
    Scale3d::register(engine);
}
