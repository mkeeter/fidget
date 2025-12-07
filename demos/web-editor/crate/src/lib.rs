use fidget::{
    context::{Context, Tree},
    gui::{Canvas2, Canvas3, DragMode, View2, View3},
    raster::{GeometryBuffer, ImageRenderConfig, VoxelRenderConfig},
    render::{CancelToken, ImageSize, ThreadPool, TileSizes, VoxelSize},
    var::Var,
    vm::{VmData, VmShape},
};
use nalgebra::Point2;

use wasm_bindgen::prelude::*;
pub use wasm_bindgen_rayon::init_thread_pool;

#[derive(Clone)]
#[wasm_bindgen]
pub struct JsTree(Tree);

#[derive(Clone)]
#[wasm_bindgen]
pub struct JsVmShape(VmShape);

#[wasm_bindgen]
pub fn eval_script(s: &str) -> Result<JsTree, String> {
    let engine = fidget::rhai::engine();
    let out = engine.eval(s);
    out.map(JsTree).map_err(|e| format!("{e}"))
}

/// Serializes a `JsTree` into a `bincode`-packed `VmData`
#[wasm_bindgen]
pub fn serialize_into_tape(t: JsTree) -> Result<Vec<u8>, String> {
    let mut ctx = Context::new();
    let root = ctx.import(&t.0);
    let shape = VmShape::new(&ctx, root).map_err(|e| format!("{e}"))?;
    let vm_data = shape.inner().data();
    let axes = shape.axes();
    bincode::serialize(&(vm_data, axes)).map_err(|e| format!("{e}"))
}

/// Deserialize a `bincode`-packed `VmData` into a `VmShape`
#[wasm_bindgen]
pub fn deserialize_tape(data: Vec<u8>) -> Result<JsVmShape, String> {
    let (d, axes): (VmData<255>, [Var; 3]) =
        bincode::deserialize(&data).map_err(|e| format!("{e}"))?;
    Ok(JsVmShape(VmShape::new_raw(d.into(), axes)))
}

/// Renders the image in 2D
#[wasm_bindgen]
pub fn render_2d(
    shape: JsVmShape,
    image_size: usize,
    camera: JsCamera2,
    cancel: JsCancelToken,
) -> Result<Vec<u8>, String> {
    fn inner(
        shape: VmShape,
        image_size: usize,
        view: View2,
        cancel: CancelToken,
    ) -> Option<Vec<u8>> {
        let cfg = ImageRenderConfig {
            image_size: ImageSize::from(image_size as u32),
            threads: Some(&ThreadPool::Global),
            tile_sizes: TileSizes::new(&[64, 16, 8]).unwrap(),
            pixel_perfect: false,
            world_to_model: view.world_to_model(),
            cancel,
        };

        let tmp = cfg.run(shape)?;
        let out =
            fidget::raster::effects::to_rgba_bitmap(tmp, false, cfg.threads);
        Some(out.into_iter().flatten().collect())
    }
    inner(shape.0, image_size, camera.0, cancel.0)
        .ok_or_else(|| "cancelled".to_owned())
}

/// Renders a heightmap image
#[wasm_bindgen]
pub fn render_heightmap(
    shape: JsVmShape,
    image_size: usize,
    camera: JsCamera3,
    cancel: JsCancelToken,
) -> Result<Vec<u8>, String> {
    let image = render_3d_inner(shape.0, image_size, camera.0, cancel.0)
        .ok_or_else(|| "cancelled".to_string())?;

    // Convert into an image
    Ok(image
        .into_iter()
        .flat_map(|v| {
            let d = (v.depth as usize * 255 / image_size) as u8;
            [d, d, d, 255]
        })
        .collect())
}

/// Renders a shaded image
#[wasm_bindgen]
pub fn render_normals(
    shape: JsVmShape,
    image_size: usize,
    camera: JsCamera3,
    cancel: JsCancelToken,
) -> Result<Vec<u8>, String> {
    let image = render_3d_inner(shape.0, image_size, camera.0, cancel.0)
        .ok_or_else(|| "cancelled".to_string())?;

    // Convert into an image
    Ok(image
        .into_iter()
        .flat_map(|p| {
            let [r, g, b] = if p.depth > 0.0 { p.to_color() } else { [0; 3] };
            [r, g, b, 255]
        })
        .collect())
}

fn render_3d_inner(
    shape: VmShape,
    image_size: usize,
    view: View3,
    cancel: CancelToken,
) -> Option<GeometryBuffer> {
    let cfg = VoxelRenderConfig {
        image_size: VoxelSize::from(image_size as u32),
        threads: Some(&ThreadPool::Global),
        tile_sizes: TileSizes::new(&[128, 64, 32, 16, 8]).unwrap(),
        world_to_model: view.world_to_model(),
        cancel,
    };
    cfg.run(shape.clone())
}

////////////////////////////////////////////////////////////////////////////////

#[wasm_bindgen]
pub struct JsCamera3(View3);

#[wasm_bindgen]
impl JsCamera3 {
    #[wasm_bindgen]
    pub fn deserialize(data: &[u8]) -> Self {
        Self(bincode::deserialize::<View3>(data).unwrap())
    }
}

#[wasm_bindgen]
pub struct JsCanvas3(Canvas3);

#[wasm_bindgen]
impl JsCanvas3 {
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32) -> Self {
        Self(Canvas3::new(VoxelSize::new(
            width,
            height,
            width.max(height),
        )))
    }

    #[wasm_bindgen]
    pub fn serialize_view(&self) -> Vec<u8> {
        bincode::serialize(&self.0.view()).unwrap()
    }

    #[wasm_bindgen]
    pub fn begin_drag(&mut self, x: i32, y: i32, button: bool) {
        self.0.begin_drag(
            Point2::new(x, y),
            if button {
                DragMode::Pan
            } else {
                DragMode::Rotate
            },
        )
    }

    #[wasm_bindgen]
    pub fn drag(&mut self, x: i32, y: i32) -> bool {
        self.0.drag(Point2::new(x, y))
    }

    #[wasm_bindgen]
    pub fn end_drag(&mut self) {
        self.0.end_drag()
    }

    #[wasm_bindgen]
    pub fn zoom_about(&mut self, amount: f32, x: i32, y: i32) -> bool {
        self.0.zoom(amount, Some(Point2::new(x, y)))
    }
}

////////////////////////////////////////////////////////////////////////////////

#[wasm_bindgen]
pub struct JsCamera2(View2);

#[wasm_bindgen]
pub struct JsCanvas2(Canvas2);

#[wasm_bindgen]
impl JsCanvas2 {
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32) -> Self {
        Self(Canvas2::new(ImageSize::new(width, height)))
    }

    #[wasm_bindgen]
    pub fn serialize_view(&self) -> Vec<u8> {
        bincode::serialize(&self.0.view()).unwrap()
    }

    #[wasm_bindgen]
    pub fn begin_drag(&mut self, x: i32, y: i32) {
        self.0.begin_drag(Point2::new(x, y))
    }

    #[wasm_bindgen]
    pub fn drag(&mut self, x: i32, y: i32) -> bool {
        self.0.drag(Point2::new(x, y))
    }

    #[wasm_bindgen]
    pub fn end_drag(&mut self) {
        self.0.end_drag()
    }

    #[wasm_bindgen]
    pub fn zoom_about(&mut self, amount: f32, x: i32, y: i32) -> bool {
        self.0.zoom(amount, Some(Point2::new(x, y)))
    }
}

#[wasm_bindgen]
impl JsCamera2 {
    #[wasm_bindgen]
    pub fn deserialize(data: &[u8]) -> Self {
        Self(bincode::deserialize::<View2>(data).unwrap())
    }
}

////////////////////////////////////////////////////////////////////////////////

#[wasm_bindgen]
pub struct JsCancelToken(CancelToken);

#[wasm_bindgen]
impl JsCancelToken {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self(CancelToken::new())
    }

    #[wasm_bindgen]
    pub fn cancel(&self) {
        self.0.cancel()
    }

    #[wasm_bindgen]
    pub fn get_ptr(&self) -> *const std::sync::atomic::AtomicBool {
        self.0.clone().into_raw()
    }

    #[wasm_bindgen]
    pub unsafe fn from_ptr(ptr: *const std::sync::atomic::AtomicBool) -> Self {
        let token = unsafe { CancelToken::from_raw(ptr) };
        Self(token)
    }
}

////////////////////////////////////////////////////////////////////////////////

#[wasm_bindgen]
pub fn get_module() -> wasm_bindgen::JsValue {
    wasm_bindgen::module()
}

#[wasm_bindgen]
pub fn get_memory() -> wasm_bindgen::JsValue {
    wasm_bindgen::memory()
}
