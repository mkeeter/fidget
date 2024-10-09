use nalgebra::Vector2;

#[derive(Copy, Clone, Debug)]
pub struct Camera2D {
    /// Scale to translate between screen and world units
    scale: f32,

    /// Offset in world units
    offset: Vector2<f32>,

    /// Starting position for a drag, in world units
    drag_start: Option<Vector2<f32>>,

    /// Size of the viewport in screen units
    viewport: Rect,
}

#[derive(Copy, Clone, Debug)]
pub struct Rect {
    pub min: Vector2<f32>,
    pub max: Vector2<f32>,
}

impl Default for Camera2D {
    fn default() -> Self {
        Camera2D {
            drag_start: None,
            scale: 1.0,
            offset: Vector2::zeros(),
            viewport: Rect {
                min: Vector2::new(f32::NAN, f32::NAN),
                max: Vector2::new(f32::NAN, f32::NAN),
            },
        }
    }
}

impl Camera2D {
    /// Builds a new camera with the given viewport
    pub fn new() -> Self {
        Self::default()
    }

    pub fn offset(&self) -> Vector2<f32> {
        self.offset
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn viewport(&self) -> Rect {
        self.viewport
    }

    /// Returns UV coordinates that cover the window
    pub fn uv(&self) -> Rect {
        let size = self.viewport.max - self.viewport.min;
        if size.x > size.y {
            let r = (1.0 - (size.y / size.x)) / 2.0;
            Rect {
                min: Vector2::new(0.0, r),
                max: Vector2::new(1.0, 1.0 - r),
            }
        } else {
            let r = (1.0 - (size.x / size.y)) / 2.0;
            Rect {
                min: Vector2::new(r, 0.0),
                max: Vector2::new(1.0 - r, 1.0),
            }
        }
    }

    /// Updates the screen and world viewport sizes
    pub fn set_viewport(&mut self, viewport: Rect) {
        self.viewport = viewport;
    }

    /// Converts from mouse position to a UV position within the render window
    fn screen_to_world(&self, p: Vector2<f32>) -> Vector2<f32> {
        let size = self.viewport.max - self.viewport.min;
        let out = (p - (self.viewport.min + self.viewport.max) / 2.0)
            * self.scale
            / size.x.max(size.y);
        self.offset + out.component_mul(&Vector2::new(2.0, -2.0))
    }

    /// Performs a new drag operation
    ///
    /// Returns `true` if the camera has changed
    pub fn drag(&mut self, pos: Vector2<f32>) -> bool {
        if let Some(start) = self.drag_start {
            let prev_offset = self.offset;
            self.offset = Vector2::zeros();
            let pos = self.screen_to_world(pos);
            let new_offset = start - pos;
            let changed = prev_offset != new_offset;
            self.offset = new_offset;
            changed
        } else {
            let pos = self.screen_to_world(pos);
            self.drag_start = Some(pos);
            false
        }
    }

    /// Releases the drag
    pub fn release(&mut self) {
        self.drag_start = None
    }

    /// Called when the mouse is scrolled
    pub fn scroll(&mut self, pos: Option<Vector2<f32>>, scroll: f32) -> bool {
        if scroll != 0.0 {
            let pos_before = pos.map(|p| self.screen_to_world(p));
            self.scale /= (scroll / 100.0).exp2();
            if let Some(pos_before) = pos_before {
                let pos_after = self.screen_to_world(pos.unwrap());
                self.offset += pos_before - pos_after;
            }
            true
        } else {
            false
        }
    }
}
