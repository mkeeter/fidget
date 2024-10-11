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
    /// Builds a new camera with an empty viewport
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the camera's current offset, in world units
    pub fn offset(&self) -> Vector2<f32> {
        self.offset
    }

    /// Returns the camera's current scale, mapping from screen to world units
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Returns the current viewport, in screen units
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

    /// Updates the screen viewport size
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

    /// Updates the camera position when the mouse is held and dragged
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

    /// Updates the camera zoom when the mouse is scrolled
    ///
    /// If the mouse cursor position is provided (in screen units), then the
    /// camera offset is updated to keep the same point under the cursor.
    ///
    /// Returns `true` if the camera has changed
    pub fn scroll(&mut self, pos: Option<Vector2<f32>>, scroll: f32) -> bool {
        if scroll != 0.0 {
            let new_scale = self.scale / (scroll / 100.0).exp2();
            match pos {
                Some(p) => {
                    let pos_before = self.screen_to_world(p);
                    self.scale = new_scale;
                    let pos_after = self.screen_to_world(p);
                    self.offset += pos_before - pos_after;
                }
                None => {
                    self.scale = new_scale;
                }
            }
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_drag() {
        let mut c = Camera2D::new();
        c.set_viewport(Rect {
            min: Vector2::new(0.0, 0.0),
            max: Vector2::new(100.0, 100.0),
        });
        let p = c.screen_to_world(Vector2::new(0.0, 0.0));
        assert_eq!(p, Vector2::new(-1.0, 1.0));
        let p = c.screen_to_world(Vector2::new(100.0, 0.0));
        assert_eq!(p, Vector2::new(1.0, 1.0));
        let p = c.screen_to_world(Vector2::new(0.0, 100.0));
        assert_eq!(p, Vector2::new(-1.0, -1.0));
        let p = c.screen_to_world(Vector2::new(100.0, 100.0));
        assert_eq!(p, Vector2::new(1.0, -1.0));

        c.drag(Vector2::new(50.0, 50.0));
        c.drag(Vector2::new(100.0, 50.0));
        c.release();

        let p = c.screen_to_world(Vector2::new(50.0, 50.0));
        assert_eq!(p, Vector2::new(-1.0, 0.0));

        c.drag(Vector2::new(50.0, 50.0));
        c.drag(Vector2::new(50.0, 100.0));
        c.release();

        let p = c.screen_to_world(Vector2::new(50.0, 50.0));
        assert_eq!(p, Vector2::new(-1.0, 1.0));
    }

    #[test]
    fn test_zoom() {
        let mut c = Camera2D::new();
        c.set_viewport(Rect {
            min: Vector2::new(0.0, 0.0),
            max: Vector2::new(100.0, 100.0),
        });

        let points = [
            (75.0, 75.0),
            (75.0, 25.0),
            (25.0, 75.0),
            (25.0, 25.0),
            (50.0, 50.0),
        ]
        .map(|(x, y)| Vector2::new(x, y));
        for &p in &points {
            let corner = Vector2::zeros();
            let prev_p = c.screen_to_world(p);
            let prev_corner = c.screen_to_world(corner);
            c.scroll(Some(p), 100.0);
            let next_p = c.screen_to_world(p);
            let next_corner = c.screen_to_world(corner);

            assert_eq!(prev_p, next_p);
            assert_ne!(prev_corner, next_corner);
        }

        // Undo the zooming to put us back at the origin
        assert_ne!(c.offset, Vector2::new(0.0, 0.0));
        for &p in points.iter().rev() {
            c.scroll(Some(p), -100.0);
        }
        assert_eq!(c.offset, Vector2::new(0.0, 0.0));
    }
}
