use fidget::{
    context::{Context, Tree},
    eval::MathShape,
    render::{BitRenderMode, RenderConfig},
    shape::Bounds,
    vm::{VmData, VmShape},
    Error,
};
use wasm_bindgen::prelude::*;

#[derive(Clone)]
#[wasm_bindgen]
pub struct JsTree(Tree);

#[derive(Clone)]
#[wasm_bindgen]
pub struct JsVmShape(VmShape);

#[wasm_bindgen]
pub fn eval_script(s: &str) -> Result<JsTree, String> {
    let mut engine = fidget::rhai::Engine::new();
    let out = engine.eval(s);
    out.map(JsTree).map_err(|e| format!("{e}"))
}

/// Serializes a `JsTree` into a `bincode`-packed `VmData`
#[wasm_bindgen]
pub fn serialize_into_tape(t: JsTree) -> Result<Vec<u8>, String> {
    let mut ctx = Context::new();
    let root = ctx.import(&t.0);
    let shape = VmShape::new(&ctx, root).map_err(|e| format!("{e}"))?;
    bincode::serialize(shape.data()).map_err(|e| format!("{e}"))
}

/// Deserialize a `bincode`-packed `VmData` into a `VmShape`
#[wasm_bindgen]
pub fn deserialize_tape(data: Vec<u8>) -> Result<JsVmShape, String> {
    let d: VmData<255> =
        bincode::deserialize(&data).map_err(|e| format!("{e}"))?;
    Ok(JsVmShape(VmShape::from(d)))
}

/// Renders a subregion of an image, for webworker-based multithreading
///
/// The image has a total size of `image_size` (on each side) and is divided
/// into `0 <= pos < workers_per_side^2` tiles.
#[wasm_bindgen]
pub fn render_region(
    shape: JsVmShape,
    image_size: usize,
    index: usize,
    workers_per_side: usize,
) -> Result<Vec<u8>, String> {
    if index >= workers_per_side.pow(2) {
        return Err("invalid index".to_owned());
    }
    if image_size % workers_per_side != 0 {
        return Err(
            "image_size must be divisible by workers_per_side".to_owned()
        );
    }
    fn inner(
        shape: VmShape,
        image_size: usize,
        index: usize,
        workers_per_side: usize,
    ) -> Result<Vec<u8>, Error> {
        // Corner position in [0, workers_per_side] coordinates
        let mut corner = nalgebra::Vector2::new(
            index / workers_per_side,
            index % workers_per_side,
        )
        .cast::<f32>();
        // Corner position in [-1, 1] coordinates
        corner = (corner * 2.0 / workers_per_side as f32).add_scalar(-1.0);

        // Scale of each tile
        let scale = 2.0 / workers_per_side as f32;

        // Tile center
        let center = corner.add_scalar(scale / 2.0);

        let cfg = RenderConfig::<2> {
            image_size: image_size / workers_per_side,
            bounds: Bounds {
                center,
                size: scale / 2.0,
            },
            ..RenderConfig::default()
        };

        let out = cfg.run(shape, &BitRenderMode)?;
        Ok(out
            .into_iter()
            .flat_map(|b| {
                let b = b as u8 * u8::MAX;
                [b, b, b, 255]
            })
            .collect())
    }
    inner(shape.0, image_size, index, workers_per_side)
        .map_err(|e| format!("{e}"))
}
