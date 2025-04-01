use fidget::render::{RotateHandle, TranslateHandle, View3};
use nalgebra::Point3;

#[derive(Copy, Clone, Default)]
pub(crate) struct Canvas3D {
    view: View3,
    drag_start: Option<Drag3D>,
}

#[derive(Copy, Clone)]
pub(crate) enum DragMode {
    Pan,
    Rotate,
}

#[derive(Copy, Clone)]
enum Drag3D {
    Pan(TranslateHandle<3>),
    Rotate(RotateHandle),
}

impl Canvas3D {
    pub fn view(&self) -> View3 {
        self.view
    }

    pub fn drag(
        &mut self,
        pos_world: Point3<f32>,
        drag_mode: DragMode,
    ) -> bool {
        match &self.drag_start {
            Some(Drag3D::Pan(prev)) => self.view.translate(prev, pos_world),
            Some(Drag3D::Rotate(prev)) => self.view.rotate(prev, pos_world),
            None => {
                self.drag_start = Some(match drag_mode {
                    DragMode::Pan => {
                        Drag3D::Pan(self.view.begin_translate(pos_world))
                    }
                    DragMode::Rotate => {
                        Drag3D::Rotate(self.view.begin_rotate(pos_world))
                    }
                });
                false
            }
        }
    }

    pub fn end_drag(&mut self) {
        self.drag_start = None;
    }

    pub fn zoom(
        &mut self,
        amount: f32,
        pos_world: Option<Point3<f32>>,
    ) -> bool {
        self.view.zoom((amount / 100.0).exp2(), pos_world)
    }
}
