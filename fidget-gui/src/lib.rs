//! Platform-independent GUI abstractions
use fidget_core::render::{ImageSize, VoxelSize};
use nalgebra::{
    Const, DefaultAllocator, DimNameAdd, DimNameSum, Matrix3, Matrix4, OMatrix,
    OPoint, OVector, Point2, Point3, U1, Vector2, Vector3,
    allocator::Allocator,
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
/// # use fidget_gui::{View2};
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
/// [`RegionSize::screen_to_world`](fidget_core::render::RegionSize::screen_to_world),
/// which converts from screen to world coordinates.
#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq)]
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

    /// Returns a `(center, scale)` tuple
    pub fn components(&self) -> (Vector2<f32>, f32) {
        (self.center, self.scale)
    }

    /// Builds a view from its components
    ///
    /// This function is identical to
    /// [`from_center_and_scale`](Self::from_center_and_scale)
    pub fn from_components(center: Vector2<f32>, scale: f32) -> Self {
        Self::from_center_and_scale(center, scale)
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
#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq)]
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

    /// Returns a `(center, scale, yaw, pitch)` tuple
    pub fn components(&self) -> (Vector3<f32>, f32, f32, f32) {
        (self.center, self.scale, self.yaw, self.pitch)
    }

    /// Builds the view from its components
    pub fn from_components(
        center: Vector3<f32>,
        scale: f32,
        yaw: f32,
        pitch: f32,
    ) -> Self {
        Self {
            center,
            scale,
            yaw,
            pitch,
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
    DefaultAllocator:
        Allocator<DimNameSum<Const<N>, U1>, DimNameSum<Const<N>, U1>>,
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

impl TranslateHandle<2> {
    /// Returns the new value for [`View2::center`]
    fn center(&self, pos: Point2<f32>) -> Vector2<f32> {
        let pos_model = self.initial_mat.transform_point(&pos);
        self.initial_center - (pos_model - self.start)
    }
}

impl TranslateHandle<3> {
    /// Returns the new value for [`View3::center`]
    fn center(&self, pos: Point3<f32>) -> Vector3<f32> {
        let pos_model = self.initial_mat.transform_point(&pos);
        self.initial_center - (pos_model - self.start)
    }
}

/// Stateful abstraction for a 2D canvas supporting drag and zoom
///
/// The canvas may be used in either immediate mode or callback mode.
///
/// In **immediate mode**, the user sends the complete cursor state to
/// [`Canvas2::interact`].
///
/// In **callback mode**, lower-level functions should be invoked as callbacks:
/// - [`Canvas2::resize`]
/// - [`Canvas2::begin_drag`]
/// - [`Canvas2::drag`]
/// - [`Canvas2::end_drag`]
/// - [`Canvas2::zoom`]
#[derive(Copy, Clone)]
pub struct Canvas2 {
    view: View2,
    image_size: ImageSize,
    drag_start: Option<TranslateHandle<2>>,
}

/// On-screen cursor state
#[derive(Copy, Clone, Debug)]
pub struct CursorState<D> {
    /// Position within the canvas, in screen coordinates
    pub screen_pos: Point2<i32>,

    /// Current drag state
    ///
    /// This is generic, because it varies between 2D and 3D
    pub drag: D,
}

impl Canvas2 {
    /// Builds a new canvas with the given image size and default view
    pub fn new(image_size: ImageSize) -> Self {
        Self {
            view: View2::default(),
            image_size,
            drag_start: None,
        }
    }

    /// Destructures the canvas into its components
    pub fn components(&self) -> (View2, ImageSize) {
        (self.view, self.image_size)
    }

    /// Builds the canvas from components
    pub fn from_components(view: View2, image_size: ImageSize) -> Self {
        Self {
            view,
            image_size,
            drag_start: None,
        }
    }

    /// Stateful interaction with the canvas
    ///
    /// The `cursor_state` indicates whether cursor is on-screen, and if so,
    /// whether the mouse button is down.
    ///
    /// Returns a boolean value indicating whether the view has changed
    #[must_use]
    pub fn interact(
        &mut self,
        image_size: ImageSize,
        cursor_state: Option<CursorState<bool>>,
        scroll: f32,
    ) -> bool {
        self.image_size = image_size;
        let mut changed = false;
        let pos_screen = match cursor_state {
            Some(cs) => {
                if cs.drag {
                    self.begin_drag(cs.screen_pos); // idempotent
                    changed |= self.drag(cs.screen_pos);
                } else {
                    self.end_drag();
                }
                Some(cs.screen_pos)
            }
            _ => {
                self.end_drag();
                None
            }
        };
        changed |= self.zoom(scroll, pos_screen);
        changed
    }

    /// Returns the current view
    pub fn view(&self) -> View2 {
        self.view
    }

    /// Returns the current image size
    pub fn image_size(&self) -> ImageSize {
        self.image_size
    }

    /// Callback when the canvas is resized
    pub fn resize(&mut self, image_size: ImageSize) {
        self.image_size = image_size;
    }

    /// Begins a new drag with the mouse at the given screen position
    ///
    /// If a drag is already in progress, this function does nothing.
    pub fn begin_drag(&mut self, pos_screen: Point2<i32>) {
        if self.drag_start.is_none() {
            let pos_world = self.image_size.transform_point(pos_screen);
            self.drag_start = Some(self.view.begin_translate(pos_world));
        }
    }

    /// Callback when the cursor is moved
    ///
    /// If a drag is in progress, then the drag continues and this function
    /// returns a boolean value indicating whether the view has changed;
    /// otherwise, it returns `false`
    #[must_use]
    pub fn drag(&mut self, pos_screen: Point2<i32>) -> bool {
        if let Some(prev) = &self.drag_start {
            let pos_world = self.image_size.transform_point(pos_screen);
            self.view.translate(prev, pos_world)
        } else {
            false
        }
    }

    /// Callback when the mouse button is released
    pub fn end_drag(&mut self) {
        self.drag_start = None;
    }

    /// Callback when the user scrolls
    ///
    /// `amount` should be a linear amount (either positive or negative); this
    /// function converts it into an scaled exponential.
    ///
    /// Returns a boolean value indicating whether the view has changed
    #[must_use]
    pub fn zoom(
        &mut self,
        amount: f32,
        pos_screen: Option<Point2<i32>>,
    ) -> bool {
        let pos_world = pos_screen.map(|p| self.image_size.transform_point(p));
        self.view.zoom((amount / 100.0).exp2(), pos_world)
    }
}

////////////////////////////////////////////////////////////////////////////////

/// Stateful abstraction for a 3D canvas supporting pan, zoom, and rotate
#[derive(Copy, Clone)]
pub struct Canvas3 {
    view: View3,
    image_size: VoxelSize,
    drag_start: Option<Drag3>,
}

/// 3D drag mode
#[derive(Copy, Clone)]
pub enum DragMode {
    /// Translate the view in local XY coordinates
    Pan,
    /// Rotate the view about the current center
    Rotate,
}

#[derive(Copy, Clone)]
enum Drag3 {
    Pan(TranslateHandle<3>),
    Rotate(RotateHandle),
}

/// Operations have the same semantics as [`Canvas2`]; see those docstrings for
/// details.
#[allow(missing_docs)]
impl Canvas3 {
    pub fn new(image_size: VoxelSize) -> Self {
        Self {
            view: View3::default(),
            image_size,
            drag_start: None,
        }
    }

    pub fn components(&self) -> (View3, VoxelSize) {
        (self.view, self.image_size)
    }

    pub fn from_components(view: View3, image_size: VoxelSize) -> Self {
        Self {
            view,
            image_size,
            drag_start: None,
        }
    }

    pub fn image_size(&self) -> VoxelSize {
        self.image_size
    }

    #[must_use]
    pub fn interact(
        &mut self,
        image_size: VoxelSize,
        cursor_state: Option<CursorState<Option<DragMode>>>,
        scroll: f32,
    ) -> bool {
        let mut changed = false;
        self.image_size = image_size;
        let pos_screen = match cursor_state {
            Some(cs) => {
                if let Some(drag_mode) = cs.drag {
                    self.begin_drag(cs.screen_pos, drag_mode); // idempotent
                    changed |= self.drag(cs.screen_pos);
                } else {
                    self.end_drag();
                }
                Some(cs.screen_pos)
            }
            _ => {
                self.end_drag();
                None
            }
        };
        changed |= self.zoom(scroll, pos_screen);
        changed
    }

    pub fn view(&self) -> View3 {
        self.view
    }

    pub fn begin_drag(&mut self, pos_screen: Point2<i32>, drag_mode: DragMode) {
        if self.drag_start.is_none() {
            let pos_world = self.screen_to_world(pos_screen);
            self.drag_start = Some(match drag_mode {
                DragMode::Pan => {
                    Drag3::Pan(self.view.begin_translate(pos_world))
                }
                DragMode::Rotate => {
                    Drag3::Rotate(self.view.begin_rotate(pos_world))
                }
            });
        }
    }

    fn screen_to_world(&self, pos_screen: Point2<i32>) -> Point3<f32> {
        self.image_size.transform_point(Point3::new(
            pos_screen.x,
            pos_screen.y,
            0,
        ))
    }

    #[must_use]
    pub fn drag(&mut self, pos_screen: Point2<i32>) -> bool {
        let pos_world = self.screen_to_world(pos_screen);
        match &self.drag_start {
            Some(Drag3::Pan(prev)) => self.view.translate(prev, pos_world),
            Some(Drag3::Rotate(prev)) => self.view.rotate(prev, pos_world),
            None => false,
        }
    }

    pub fn end_drag(&mut self) {
        self.drag_start = None;
    }

    #[must_use]
    pub fn zoom(
        &mut self,
        amount: f32,
        pos_screen: Option<Point2<i32>>,
    ) -> bool {
        let pos_world = pos_screen.map(|p| self.screen_to_world(p));
        self.view.zoom((amount / 100.0).exp2(), pos_world)
    }
}
