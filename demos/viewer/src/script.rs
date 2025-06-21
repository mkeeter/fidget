use anyhow::Result;
use crossbeam_channel::{Receiver, Sender};
use fidget::{context::Tree, rhai::FromDynamic};
use log::debug;
use std::sync::{Arc, Mutex};

/// Receives scripts and executes them with Fidget
pub(crate) fn rhai_script_thread(
    rx: Receiver<String>,
    tx: Sender<Result<ScriptContext, String>>,
) -> Result<()> {
    let mut engine = Engine::new();

    loop {
        let script = rx.recv()?;
        debug!("rhai script thread received script");
        let r = engine.run(&script).map_err(|e| e.to_string());
        debug!("rhai script thread is sending result to render thread");
        tx.send(r)?;
    }
}

/// Engine for evaluating a Rhai script with Fidget-specific bindings
pub struct Engine {
    engine: rhai::Engine,
    context: Arc<Mutex<ScriptContext>>,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    pub fn new() -> Self {
        let mut engine = fidget::rhai::engine();

        engine.register_fn("draw", draw);
        engine.register_fn("draw_rgb", draw_rgb);

        let context = Arc::new(Mutex::new(ScriptContext::new()));
        engine.set_default_tag(rhai::Dynamic::from(context.clone()));
        engine.set_max_expr_depths(64, 32);

        Self { engine, context }
    }

    /// Executes a full script
    pub fn run(
        &mut self,
        script: &str,
    ) -> Result<ScriptContext, Box<rhai::EvalAltResult>> {
        self.context.lock().unwrap().clear();

        self.engine.run(script)?;

        // Steal the ScriptContext's contents
        let mut lock = self.context.lock().unwrap();
        Ok(std::mem::take(&mut lock))
    }
}

/// Shape to render
///
/// Populated by calls to `draw(...)` or `draw_rgb(...)` in a Rhai script
pub struct DrawShape {
    /// Tree to render
    pub tree: Tree,
    /// Color to use when drawing the shape
    pub color_rgb: [u8; 3],
}

/// Context for shape evaluation
///
/// This object stores a set of shapes, which is populated by calls to `draw` or
/// `draw_rgb` during script evaluation.
pub struct ScriptContext {
    /// List of shapes populated since the last call to [`clear`](Self::clear)
    pub shapes: Vec<DrawShape>,
}

impl Default for ScriptContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ScriptContext {
    /// Builds a new empty script context
    pub fn new() -> Self {
        Self { shapes: vec![] }
    }
    /// Resets the script context
    pub fn clear(&mut self) {
        self.shapes.clear();
    }
}

fn draw(
    ctx: rhai::NativeCallContext,
    tree: rhai::Dynamic,
) -> Result<(), Box<rhai::EvalAltResult>> {
    let tree = Tree::from_dynamic(&ctx, tree, None)?;
    let ctx = ctx.tag().unwrap().clone_cast::<Arc<Mutex<ScriptContext>>>();
    ctx.lock().unwrap().shapes.push(DrawShape {
        tree,
        color_rgb: [u8::MAX; 3],
    });
    Ok(())
}

fn draw_rgb(
    ctx: rhai::NativeCallContext,
    tree: rhai::Dynamic,
    r: f64,
    g: f64,
    b: f64,
) -> Result<(), Box<rhai::EvalAltResult>> {
    let tree = Tree::from_dynamic(&ctx, tree, None)?;
    let ctx = ctx.tag().unwrap().clone_cast::<Arc<Mutex<ScriptContext>>>();
    let f = |a| {
        if a < 0.0 {
            0
        } else if a > 1.0 {
            255
        } else {
            (a * 255.0) as u8
        }
    };
    ctx.lock().unwrap().shapes.push(DrawShape {
        tree,
        color_rgb: [f(r), f(g), f(b)],
    });
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use fidget::context::{BinaryOpcode, Context, Op};

    #[test]
    fn test_simple_script() {
        let mut engine = Engine::new();
        let out = engine
            .run(
                "
                let s = circle(#{ center: [0, 0], radius: 2 });
                draw(s.move([1, 3]));
                ",
            )
            .unwrap();
        assert_eq!(out.shapes.len(), 1);
        let mut ctx = Context::new();
        let sum = ctx.import(&out.shapes[0].tree);
        assert_eq!(ctx.eval_xyz(sum, 1.0, 3.0, 0.0).unwrap(), -2.0);
    }

    #[test]
    fn test_gyroid_sphere() {
        let mut engine = Engine::new();
        let s = include_str!("../../../models/gyroid-sphere.rhai");
        let out = engine.run(s).unwrap();
        assert_eq!(out.shapes.len(), 1);
        let mut ctx = Context::new();
        let sphere = ctx.import(&out.shapes[0].tree);
        assert!(matches!(
            ctx.get_op(sphere).unwrap(),
            Op::Binary(BinaryOpcode::Max, _, _)
        ));
    }
}
