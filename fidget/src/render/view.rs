use nalgebra::{
    geometry::Similarity2, Matrix3, Matrix4, Point2, Point3, Vector2, Vector3,
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
        // TODO make this world space for consistency?
        self.mat.append_translation_mut(&dt.into());
    }

    /// Zooms the camera about a particular position (in world space)
    pub fn zoom(&mut self, amount: f32, pos: Option<Point2<f32>>) {
        match pos {
            Some(before) => {
                let pos_before = self.transform_point(&before);
                self.mat.append_scaling_mut(amount);
                let pos_after = self.transform_point(&before);
                self.mat
                    .append_translation_mut(&(pos_before - pos_after).into());
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
/// See [`View2`] for a diagram of coordinate spaces
impl View3 {
    /// Builds a camera from a center (in world coordinates) and a scale
    ///
    /// The resulting camera will point at the center, and the viewport will be
    /// ± `scale` in size.
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
        let scale = Matrix4::new_scaling(self.scale);
        let rot = Matrix4::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
            self.yaw,
        ) * Matrix4::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(1.0, 0.0, 0.0)),
            self.pitch,
        );
        let translation = Matrix4::new_translation(&self.center);

        translation * rot * scale
    }

    /// Transform a point from world to model space
    pub fn transform_point(&self, p: &Point3<f32>) -> Point3<f32> {
        self.world_to_model().transform_point(p)
    }

    /// Applies a translation (in model units) to the current camera position
    pub fn translate(&mut self, dt: Vector3<f32>) {
        // TODO for consistency, make this screen units?
        self.center += dt;
    }

    /// Zooms the camera about a particular position (in screen space)
    pub fn zoom(&mut self, amount: f32, pos: Option<Point3<f32>>) {
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
    }

    /// Begins a rotation operation, given a point in world space
    pub fn begin_rotate(&self, start: Point2<f32>) -> RotateHandle {
        RotateHandle {
            start,
            initial_yaw: self.yaw,
            initial_pitch: self.pitch,
        }
    }

    /// Rotates the camera, given a cursor end position in world space
    ///
    /// Returns `true` if the view has changed, `false` otherwise
    pub fn rotate(&mut self, h: RotateHandle, pos: Point2<f32>) -> bool {
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
    start: Point2<f32>,
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
