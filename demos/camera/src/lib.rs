use nalgebra::{Point2, Vector2};

#[derive(Copy, Clone, Debug)]
pub struct Camera2D {
    scale: f32,

    /// Offset in world units
    offset: Vector2<f32>,

    /// Position in screen units
    drag_start: Option<Vector2<f32>>,

    /// Size of the viewport in screen units
    rect: Rect,

    /// Size of the viewport in world units
    uv: Rect,
}

#[derive(Copy, Clone, Debug)]
pub struct Rect {
    pub min: Vector2<f32>,
    pub max: Vector2<f32>,
}

impl Camera2D {
    /// Builds a new camera with the given viewport
    pub fn new(rect: Rect, uv: Rect) -> Self {
        Camera2D {
            drag_start: None,
            scale: 1.0,
            offset: Vector2::zeros(),
            rect,
            uv,
        }
    }

    pub fn offset(&self) -> Vector2<f32> {
        self.offset
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Updates the screen and world viewport sizes
    pub fn set_viewport(&mut self, rect: Rect, uv: Rect) {
        self.rect = rect;
        self.uv = uv;
    }

    /// Converts from mouse position to a UV position within the render window
    fn mouse_to_uv(&self, p: Point2<f32>) -> Vector2<f32> {
        let r = (p - self.rect.min)
            .coords
            .component_div(&(self.rect.max - self.rect.min));
        const ONE: Vector2<f32> = Vector2::new(1.0, 1.0);
        let pos = self.uv.min.component_mul(&(ONE - r))
            + self.uv.max.component_mul(&r);
        let out = ((pos * 2.0) - ONE) * self.scale;
        println!("{p:?} -> {:?}", out + self.offset);
        out + self.offset
    }

    /// Performs a new drag operation
    ///
    /// Returns `true` if the camera has changed
    pub fn drag(&mut self, pos: Point2<f32>) -> bool {
        if let Some(start) = self.drag_start {
            self.offset = Vector2::zeros();
            let pos = self.mouse_to_uv(pos);
            println!("{start:?} -> {pos:?}");
            self.offset = start - pos;
            true
        } else {
            let pos = self.mouse_to_uv(pos);
            self.drag_start = Some(pos);
            false
        }
    }

    /// Releases the drag
    pub fn release(&mut self) {
        self.drag_start = None
    }

    /// Called when the mouse is scrolled
    pub fn scroll(&mut self, pos: Option<Point2<f32>>, scroll: f32) -> bool {
        if scroll != 0.0 {
            let pos_before = pos.map(|p| self.mouse_to_uv(p));
            self.scale /= (scroll / 100.0).exp2();
            if let Some(pos_before) = pos_before {
                let pos_after = self.mouse_to_uv(pos.unwrap());
                self.offset += pos_before - pos_after;
            }
            true
        } else {
            false
        }
    }
}
