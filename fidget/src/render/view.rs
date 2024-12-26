use nalgebra::{
    Const, DimNameAdd, Matrix3, Matrix4, OMatrix,
    OPoint, OVector, Point2, Point3, Vector2, Vector3,
    U1, DefaultAllocator, allocator::Allocator, DimNameSub, DimNameSum
};
use serde::{Deserialize, Serialize};

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
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct View2 {
    center: Vector2<f32>,
    scale: f32,
}

impl Default for View2 {
    fn default() -> Self {
        Self {
            scale: 1.0,
            center: Vector2::new(0.0, 0.0),
        }
    }
}

impl View2 {
    /// Builds a camera from a center (in world coordinates) and a scale
    ///
    /// The resulting camera will point at the center, and the viewport will be
    /// ± `scale` in size.
    pub fn from_center_and_scale(center: Vector2<f32>, scale: f32) -> Self {
        Self { center, scale }
    }

    /// Returns the scaling matrix for this view
    fn scale_mat(&self) -> Matrix3<f32> {
        Matrix3::new_scaling(self.scale)
    }

    /// Returns the translation matrix for this view
    fn translation_mat(&self) -> Matrix3<f32> {
        Matrix3::new_translation(&self.center)
    }

    /// Returns the world-to-model transform matrix
    pub fn world_to_model(&self) -> Matrix3<f32> {
        self.translation_mat() * self.scale_mat()
    }

    /// Transform a point from world to model space
    pub fn transform_point(&self, p: &Point2<f32>) -> Point2<f32> {
        self.world_to_model().transform_point(p)
    }

    /// Begins a translation operation, given a point in world space
    pub fn begin_translate(&self, start: Point2<f32>) -> TranslateHandle<2> {
        let initial_mat = self.world_to_model();
        TranslateHandle {
            start: initial_mat.transform_point(&start),
            initial_mat,
            initial_center: self.center,
        }
    }

    /// Applies a translation (in world units) to the current camera position
    pub fn translate(
        &mut self,
        h: &TranslateHandle<2>,
        pos: Point2<f32>,
    ) -> bool {
        let next_center = h.center(pos);
        let changed = next_center != self.center;
        self.center = next_center;
        changed
    }

    /// Zooms the camera about a particular position (in world space)
    ///
    /// Returns `true` if the view has changed, `false` otherwise
    pub fn zoom(&mut self, amount: f32, pos: Option<Point2<f32>>) -> bool {
        match pos {
            Some(before) => {
                let pos_before = self.transform_point(&before);
                self.scale *= amount;
                let pos_after = self.transform_point(&before);
                self.center += pos_before - pos_after;
            }
            None => {
                self.scale *= amount;
            }
        }
        amount != 1.0
    }
}

/// Object providing a view-to-model transform in 2D
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct View3 {
    center: Vector3<f32>,
    scale: f32,
    yaw: f32,
    pitch: f32,
}

impl Default for View3 {
    fn default() -> Self {
        Self {
            center: Vector3::new(0.0, 0.0, 0.0),
            scale: 1.0,
            yaw: 0.0,
            pitch: 0.0,
        }
    }
}

/// Object providing a world-to-model transform in 3D
///
/// This is implemented as a uniform scaling operation, followed by rotation
/// (pitch / yaw, i.e. turntable rotation), followed by translation.
///
/// See [`View2`] for a diagram of coordinate spaces
impl View3 {
    /// Builds a camera from a center (in world coordinates) and a scale
    ///
    /// The resulting camera will point at the center along the `-Z` axis, and
    /// the viewport will be ± `scale` in size.
    pub fn from_center_and_scale(center: Vector3<f32>, scale: f32) -> Self {
        Self {
            center,
            scale,
            yaw: 0.0,
            pitch: 0.0,
        }
    }

    /// Returns the world-to-model transform matrix
    pub fn world_to_model(&self) -> Matrix4<f32> {
        self.translation_mat() * self.rot_mat() * self.scale_mat()
    }

    /// Transform a point from world to model space
    pub fn transform_point(&self, p: &Point3<f32>) -> Point3<f32> {
        self.world_to_model().transform_point(p)
    }

    /// Begins a translation operation, given a point in world space
    pub fn begin_translate(&self, start: Point3<f32>) -> TranslateHandle<3> {
        let initial_mat = self.world_to_model();
        TranslateHandle {
            start: initial_mat.transform_point(&start),
            initial_mat,
            initial_center: self.center,
        }
    }

    /// Returns the scaling matrix for this view
    fn scale_mat(&self) -> Matrix4<f32> {
        Matrix4::new_scaling(self.scale)
    }

    /// Returns the rotation matrix for this view
    fn rot_mat(&self) -> Matrix4<f32> {
        Matrix4::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
            self.yaw,
        ) * Matrix4::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(1.0, 0.0, 0.0)),
            self.pitch,
        )
    }

    /// Returns the translation matrix for this view
    fn translation_mat(&self) -> Matrix4<f32> {
        Matrix4::new_translation(&self.center)
    }

    /// Applies a translation (in world units) to the current camera position
    pub fn translate(
        &mut self,
        h: &TranslateHandle<3>,
        pos: Point3<f32>,
    ) -> bool {
        let next_center = h.center(pos);
        let changed = next_center != self.center;
        self.center = next_center;
        changed
    }

    /// Zooms the camera about a particular position (in world space)
    ///
    /// Returns `true` if the view has changed, `false` otherwise
    pub fn zoom(&mut self, amount: f32, pos: Option<Point3<f32>>) -> bool {
        match pos {
            Some(before) => {
                let pos_before = self.transform_point(&before);
                self.scale *= amount;
                let pos_after = self.transform_point(&before);
                self.center += pos_before - pos_after;
            }
            None => {
                self.scale *= amount;
            }
        }
        amount != 1.0
    }

    /// Begins a rotation operation, given a point in world space
    pub fn begin_rotate(&self, start: Point3<f32>) -> RotateHandle {
        RotateHandle {
            start,
            initial_yaw: self.yaw,
            initial_pitch: self.pitch,
        }
    }

    /// Rotates the camera, given a cursor end position in world space
    ///
    /// Returns `true` if the view has changed, `false` otherwise
    pub fn rotate(&mut self, h: &RotateHandle, pos: Point3<f32>) -> bool {
        let next_yaw = h.yaw(pos.x);
        let next_pitch = h.pitch(pos.y);
        let changed = (next_yaw != self.yaw) || (next_pitch != self.pitch);
        self.yaw = next_yaw;
        self.pitch = next_pitch;
        changed
    }
}

/// Handle to perform rotations on a [`View3`]
#[derive(Copy, Clone)]
pub struct RotateHandle {
    /// Position of the initial click in world space
    start: Point3<f32>,
    initial_yaw: f32,
    initial_pitch: f32,
}

/// Eyeballed for pleasant UI
const ROTATE_SPEED: f32 = 2.0;

impl RotateHandle {
    fn yaw(&self, x: f32) -> f32 {
        (self.initial_yaw + (self.start.x - x) * ROTATE_SPEED)
            % std::f32::consts::TAU
    }
    fn pitch(&self, y: f32) -> f32 {
        (self.initial_pitch + (y - self.start.y) * ROTATE_SPEED)
            .clamp(0.0, std::f32::consts::PI)
    }
}

/// Handle to perform translation on a [`View2`] or [`View3`]
#[derive(Copy, Clone)]
pub struct TranslateHandle<const N: usize> 
where
    Const<N>: DimNameAdd<U1>,
    DefaultAllocator: Allocator<DimNameSum<Const<N>, U1>, DimNameSum<Const<N>, U1>>,
    DefaultAllocator: Allocator<<<Const<N> as DimNameAdd<Const<1>>>::Output as DimNameSub<Const<1>>>::Output>,
    <Const<N> as DimNameAdd<Const<1>>>::Output: DimNameSub<Const<1>>,
    <DefaultAllocator as nalgebra::allocator::Allocator<<<Const<N> as DimNameAdd<Const<1>>>::Output as DimNameSub<Const<1>>>::Output>>::Buffer<u32>: std::marker::Copy,
    OMatrix<
        f32,
        <Const<N> as DimNameAdd<Const<1>>>::Output,
        <Const<N> as DimNameAdd<Const<1>>>::Output,
    >: Copy,

{
    /// Position of the initial click, in model space
    start: OPoint<f32, Const<N>>,
    /// Initial world-to-model transform matrix
    initial_mat: OMatrix<
        f32,
        <Const<N> as DimNameAdd<Const<1>>>::Output,
        <Const<N> as DimNameAdd<Const<1>>>::Output,
    >,
    /// Initial value of [`View2::center`] or [`View3::center`]
    initial_center: OVector<f32, Const<N>>,
}

impl TranslateHandle<2> 
{
    fn center(&self, pos: Point2<f32>) -> Vector2<f32> {
        let pos_model = self.initial_mat.transform_point(&pos);
        self.initial_center - (pos_model - self.start)
    }
}

impl TranslateHandle<3> 
{
    fn center(&self, pos: Point3<f32>) -> Vector3<f32> {
        let pos_model = self.initial_mat.transform_point(&pos);
        self.initial_center - (pos_model - self.start)
    }
}
