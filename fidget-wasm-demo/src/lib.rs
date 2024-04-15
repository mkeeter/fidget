use fidget::context::Tree;
use wasm_bindgen::prelude::*;

#[derive(Clone)]
#[wasm_bindgen]
pub struct JsTree(Tree);

#[wasm_bindgen]
pub fn eval_script(s: &str) -> Result<JsTree, String> {
    let mut engine = fidget::rhai::Engine::new();
    let out = engine.eval(s);
    out.map(JsTree).map_err(|e| format!("{e:?}"))
}
