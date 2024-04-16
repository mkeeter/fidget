use fidget::{
    context::{Context, Tree},
    eval::MathShape,
    render::{BitRenderMode, RenderConfig},
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
pub fn render(t: JsTree) -> Result<Vec<u8>, String> {
    render_inner(t.0).map_err(|e| format!("{e}"))
}

fn render_inner(t: Tree) -> Result<Vec<u8>, Error> {
    let mut ctx = Context::new();
    let root = ctx.import(&t);

    let cfg = RenderConfig::<2> {
        image_size: 256,
        ..RenderConfig::default()
    };

    let shape = VmShape::new(&ctx, root)?;
    let out = cfg.run(shape, &BitRenderMode)?;
    Ok(out.into_iter().map(|b| b as u8).collect())
}
