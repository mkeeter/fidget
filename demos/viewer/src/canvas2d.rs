use fidget::render::{TranslateHandle, View2};
use nalgebra::Point2;

#[derive(Copy, Clone, Default)]
pub(crate) struct Canvas2D {
    view: View2,
    drag_start: Option<TranslateHandle<2>>,
}

impl Canvas2D {
    pub fn drag(&mut self, pos_world: Point2<f32>) -> bool {
        if let Some(prev) = &self.drag_start {
            self.view.translate(prev, pos_world)
        } else {
            self.drag_start = Some(self.view.begin_translate(pos_world));
            false
        }
    }

    pub fn end_drag(&mut self) {
        self.drag_start = None;
    }

    pub fn zoom(
        &mut self,
        amount: f32,
        pos_world: Option<Point2<f32>>,
    ) -> bool {
        self.view.zoom((amount / 100.0).exp2(), pos_world)
    }

    pub fn view(&self) -> View2 {
        self.view
    }
}
