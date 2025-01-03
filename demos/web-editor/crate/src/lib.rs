use fidget::{
    context::{Context, Tree},
    render::{
        BitRenderMode, CancelToken, ImageRenderConfig, ImageSize, RotateHandle,
        ThreadPool, TileSizes, TranslateHandle, View2, View3,
        VoxelRenderConfig, VoxelSize,
    },
    var::Var,
    vm::{VmData, VmShape},
};
use nalgebra::{Point2, Point3};

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
    let mut engine = fidget::rhai::Engine::new();
    engine.set_limit(50_000); // ¯\_(ツ)_/¯
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
            threads: Some(ThreadPool::Global),
            tile_sizes: TileSizes::new(&[64, 16, 8]).unwrap(),
            view,
            cancel,
        };

        let out = cfg.run::<_, BitRenderMode>(shape)?;
        Some(
            out.into_iter()
                .flat_map(|b| {
                    let b = b as u8 * u8::MAX;
                    [b, b, b, 255]
                })
                .collect(),
        )
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
    let (depth, _norm) =
        render_3d_inner(shape.0, image_size, camera.0, cancel.0)
            .ok_or_else(|| "cancelled".to_string())?;

    // Convert into an image
    Ok(depth
        .into_iter()
        .flat_map(|v| {
            let d = (v as usize * 255 / image_size) as u8;
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
    let (_depth, norm) =
        render_3d_inner(shape.0, image_size, camera.0, cancel.0)
            .ok_or_else(|| "cancelled".to_string())?;

    // Convert into an image
    Ok(norm
        .into_iter()
        .flat_map(|[r, g, b]| [r, g, b, 255])
        .collect())
}

fn render_3d_inner(
    shape: VmShape,
    image_size: usize,
    view: View3,
    cancel: CancelToken,
) -> Option<(Vec<u32>, Vec<[u8; 3]>)> {
    let cfg = VoxelRenderConfig {
        image_size: VoxelSize::from(image_size as u32),
        threads: Some(ThreadPool::Global),
        tile_sizes: TileSizes::new(&[64, 32, 16, 8]).unwrap(),
        view,
        cancel,
    };
    cfg.run(shape.clone())
}

////////////////////////////////////////////////////////////////////////////////

#[wasm_bindgen]
pub struct JsCamera3(View3);

#[wasm_bindgen]
impl JsCamera3 {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self(View3::default())
    }

    #[wasm_bindgen]
    pub fn serialize(&self) -> Vec<u8> {
        bincode::serialize(&self.0).unwrap()
    }

    #[wasm_bindgen]
    pub fn deserialize(data: &[u8]) -> Self {
        Self(bincode::deserialize::<View3>(data).unwrap())
    }

    #[wasm_bindgen]
    pub fn begin_translate(&self, x: f32, y: f32) -> JsTranslateHandle3 {
        JsTranslateHandle3(self.0.begin_translate(Point3::new(x, y, 0.0)))
    }

    #[wasm_bindgen]
    pub fn translate(
        &mut self,
        h: &JsTranslateHandle3,
        x: f32,
        y: f32,
    ) -> bool {
        self.0.translate(&h.0, Point3::new(x, y, 0.0))
    }

    #[wasm_bindgen]
    pub fn begin_rotate(&self, x: f32, y: f32) -> JsRotateHandle {
        JsRotateHandle(self.0.begin_rotate(Point3::new(x, y, 0.0)))
    }

    #[wasm_bindgen]
    pub fn rotate(&mut self, h: &JsRotateHandle, x: f32, y: f32) -> bool {
        self.0.rotate(&h.0, Point3::new(x, y, 0.0))
    }

    #[wasm_bindgen]
    pub fn zoom_about(&mut self, amount: f32, x: f32, y: f32) -> bool {
        self.0.zoom(amount, Some(Point3::new(x, y, 0.0)))
    }
}

#[wasm_bindgen]
pub struct JsRotateHandle(RotateHandle);

#[wasm_bindgen]
pub struct JsTranslateHandle3(TranslateHandle<3>);

////////////////////////////////////////////////////////////////////////////////

#[wasm_bindgen]
pub struct JsCamera2(View2);

#[wasm_bindgen]
impl JsCamera2 {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self(View2::default())
    }

    #[wasm_bindgen]
    pub fn serialize(&self) -> Vec<u8> {
        bincode::serialize(&self.0).unwrap()
    }

    #[wasm_bindgen]
    pub fn deserialize(data: &[u8]) -> Self {
        Self(bincode::deserialize::<View2>(data).unwrap())
    }

    #[wasm_bindgen]
    pub fn begin_translate(&self, x: f32, y: f32) -> JsTranslateHandle2 {
        JsTranslateHandle2(self.0.begin_translate(Point2::new(x, y)))
    }

    #[wasm_bindgen]
    pub fn translate(
        &mut self,
        h: &JsTranslateHandle2,
        x: f32,
        y: f32,
    ) -> bool {
        self.0.translate(&h.0, Point2::new(x, y))
    }

    #[wasm_bindgen]
    pub fn zoom_about(&mut self, amount: f32, x: f32, y: f32) -> bool {
        self.0.zoom(amount, Some(Point2::new(x, y)))
    }
}

#[wasm_bindgen]
pub struct JsTranslateHandle2(TranslateHandle<2>);

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
        Self(CancelToken::from_raw(ptr))
    }

    #[wasm_bindgen]
    pub fn is_cancelled(&self) -> bool {
        self.0.is_cancelled()
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
