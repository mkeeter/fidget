//! Platform-independent GUI abstractions
use crate::render::{
    ImageSize, RotateHandle, TranslateHandle, View2, View3, VoxelSize,
};
use nalgebra::{Point2, Point3};

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
