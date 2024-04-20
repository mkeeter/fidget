use fidget::{
    context::{Context, Tree},
    eval::MathShape,
    render::{BitRenderMode, RenderConfig},
    shape::Bounds,
    vm::VmShape,
    Error,
};
use wasm_bindgen::prelude::*;

#[derive(Clone)]
#[wasm_bindgen]
pub struct JsTree(Tree);

#[wasm_bindgen]
pub fn eval_script(s: &str) -> Result<JsTree, String> {
    let mut engine = fidget::rhai::Engine::new();
    let out = engine.eval(s);
    out.map(JsTree).map_err(|e| format!("{e}"))
}

#[wasm_bindgen]
pub fn render(t: JsTree, image_size: usize) -> Result<Vec<u8>, String> {
    fn inner(t: Tree, image_size: usize) -> Result<Vec<u8>, Error> {
        let mut ctx = Context::new();
        let root = ctx.import(&t);

        let cfg = RenderConfig::<2> {
            image_size,
            ..RenderConfig::default()
        };

        let shape = VmShape::new(&ctx, root)?;
        let out = cfg.run(shape, &BitRenderMode)?;
        Ok(out
            .into_iter()
            .flat_map(|b| {
                let b = b as u8 * u8::MAX;
                [b, b, b, 255]
            })
            .collect())
    }
    inner(t.0, image_size).map_err(|e| format!("{e}"))
}

/// Renders a subregion of an image, for webworker-based multithreading
///
/// The image has a total size of `image_size` (on each side) and is divided
/// into `0 <= pos < workers_per_side^2` tiles.
#[wasm_bindgen]
pub fn render_region(
    t: JsTree,
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
        t: Tree,
        image_size: usize,
        index: usize,
        workers_per_side: usize,
    ) -> Result<Vec<u8>, Error> {
        let mut ctx = Context::new();
        let root = ctx.import(&t);

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

        let shape = VmShape::new(&ctx, root)?;
        let out = cfg.run(shape, &BitRenderMode)?;
        Ok(out
            .into_iter()
            .flat_map(|b| {
                let b = b as u8 * u8::MAX;
                [b, b, b, 255]
            })
            .collect())
    }
    inner(t.0, image_size, index, workers_per_side).map_err(|e| format!("{e}"))
}
