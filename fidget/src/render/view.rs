use nalgebra::{
    geometry::{Similarity2, Similarity3},
    Matrix3, Matrix4, Point2, Point3, Vector2, Vector3,
};

/// Object providing a world-to-model transform in 2D
///
/// Rendering and meshing happen in the ±1 square or cube; these are referred to
/// as _world_ coordinates.  A `View` generates a homogeneous transform matrix
/// that maps from positions in world coordinates to _model_ coordinates, which
/// can be whatever you want.
///
/// For example, the world-to-model transform could map the ±1 region onto the
/// ±0.5 region, which would be a zoom transform.
///
/// Here's an example of using a `View2` to focus on the region `[4, 6]`:
///
/// ```
/// # use nalgebra::{Vector2, Point2};
/// # use fidget::render::{View2};
/// let view = View2::from_center_and_scale(Vector2::new(5.0, 5.0), 1.0);
///
/// //   -------d-------
/// //   |             |
/// //   |             |
/// //   c      a      b
/// //   |             |
/// //   |             |
/// //   -------e-------
/// let a = view.transform_point(&Point2::new(0.0, 0.0));
/// assert_eq!(a, Point2::new(5.0, 5.0));
///
/// let b = view.transform_point(&Point2::new(1.0, 0.0));
/// assert_eq!(b, Point2::new(6.0, 5.0));
///
/// let c = view.transform_point(&Point2::new(-1.0, 0.0));
/// assert_eq!(c, Point2::new(4.0, 5.0));
///
/// let d = view.transform_point(&Point2::new(0.0, 1.0));
/// assert_eq!(d, Point2::new(5.0, 6.0));
///
/// let e = view.transform_point(&Point2::new(0.0, -1.0));
/// assert_eq!(e, Point2::new(5.0, 4.0));
/// ```
///
/// See also
/// [`RegionSize::screen_to_world`](crate::render::RegionSize::screen_to_world),
/// which converts from screen to world coordinates.
#[derive(Copy, Clone, Debug)]
pub struct View2 {
    mat: Similarity2<f32>,
}

impl Default for View2 {
    fn default() -> Self {
        Self {
            mat: Similarity2::identity(),
        }
    }
}

impl View2 {
    /// Builds a camera from a center (in world coordinates) and a scale
    ///
    /// The resulting camera will point at the center, and the viewport will be
    /// ± `scale` in size.
    pub fn from_center_and_scale(center: Vector2<f32>, scale: f32) -> Self {
        let mat =
            Similarity2::from_parts(center.into(), Default::default(), scale);
        Self { mat }
    }

    /// Returns the world-to-model transform matrix
    pub fn world_to_model(&self) -> Matrix3<f32> {
        self.mat.into()
    }
    /// Transform a point from world to model space
    pub fn transform_point(&self, p: &Point2<f32>) -> Point2<f32> {
        self.mat.transform_point(p)
    }

    /// Applies a translation (in model units) to the current camera position
    pub fn translate(&mut self, dt: Vector2<f32>) {
        self.mat.append_translation_mut(&dt.into());
    }

    /// Zooms the camera about a particular position (in model space)
    pub fn zoom(&mut self, amount: f32, pos: Option<Point2<f32>>) {
        match pos {
            Some(before) => {
                // Convert to world space before scaling
                let p = self.mat.inverse_transform_point(&before);
                self.mat.append_scaling_mut(amount);
                let pos_after = self.transform_point(&p);
                self.mat
                    .append_translation_mut(&(before - pos_after).into());
            }
            None => {
                self.mat.append_scaling_mut(amount);
            }
        }
    }
}

/// Object providing a view-to-model transform in 2D
#[derive(Copy, Clone, Debug)]
pub struct View3 {
    mat: Similarity3<f32>,
}

impl Default for View3 {
    fn default() -> Self {
        Self {
            mat: Similarity3::identity(),
        }
    }
}

/// Object providing a world-to-model transform in 3D
///
/// See [`View2`] for a diagram of coordinate spaces
impl View3 {
    /// Builds a camera from a center (in world coordinates) and a scale
    ///
    /// The resulting camera will point at the center, and the viewport will be
    /// ± `scale` in size.
    pub fn from_center_and_scale(center: Vector3<f32>, scale: f32) -> Self {
        let mat =
            Similarity3::from_parts(center.into(), Default::default(), scale);
        Self { mat }
    }

    /// Returns the world-to-model transform matrix
    pub fn world_to_model(&self) -> Matrix4<f32> {
        self.mat.into()
    }

    /// Transform a point from world to model space
    pub fn transform_point(&self, p: &Point3<f32>) -> Point3<f32> {
        self.mat.transform_point(p)
    }

    /// Applies a translation (in model units) to the current camera position
    pub fn translate(&mut self, dt: Vector3<f32>) {
        self.mat.append_translation_mut(&dt.into());
    }

    /// Zooms the camera about a particular position (in model space)
    pub fn zoom(&mut self, amount: f32, pos: Option<Point3<f32>>) {
        match pos {
            Some(before) => {
                // Convert to world space before scaling
                let p = self.mat.inverse_transform_point(&before);
                self.mat.append_scaling_mut(amount);
                let pos_after = self.transform_point(&p);
                self.mat
                    .append_translation_mut(&(before - pos_after).into());
            }
            None => {
                self.mat.append_scaling_mut(amount);
            }
        }
    }
}
