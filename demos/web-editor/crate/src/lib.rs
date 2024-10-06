use fidget::{
    context::{Context, Tree},
    render::{BitRenderMode, RenderConfig},
    shape::Bounds,
    var::Var,
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

/// Renders a subregion of an image, for webworker-based multithreading
///
/// The image has a total size of `image_size` (on each side) and is divided
/// into `0 <= pos < workers_per_side^2` tiles.
#[wasm_bindgen]
pub fn render_region_2d(
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

        let out = cfg.run::<_, BitRenderMode>(shape)?;
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

/// Renders a subregion of a heightmap, for webworker-based multithreading
///
/// The image has a total size of `image_size` (on each side) and is divided
/// into `0 <= pos < workers_per_side^2` tiles.
#[wasm_bindgen]
pub fn render_region_heightmap(
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
    let (depth, _norm) =
        render_3d_inner(shape.0, image_size, index, workers_per_side)
            .map_err(|e| format!("{e}"))?;

    // Convert into an image
    Ok(depth
        .into_iter()
        .flat_map(|v| {
            let d = (v as usize * 255 / image_size) as u8;
            [d, d, d, 255]
        })
        .collect())
}

/// Renders a subregion with normals, for webworker-based multithreading
///
/// The image has a total size of `image_size` (on each side) and is divided
/// into `0 <= pos < workers_per_side^2` tiles.
#[wasm_bindgen]
pub fn render_region_normals(
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
    let (_depth, norm) =
        render_3d_inner(shape.0, image_size, index, workers_per_side)
            .map_err(|e| format!("{e}"))?;

    // Convert into an image
    Ok(norm
        .into_iter()
        .flat_map(|[r, g, b]| [r, g, b, 255])
        .collect())
}

fn render_3d_inner(
    shape: VmShape,
    image_size: usize,
    index: usize,
    workers_per_side: usize,
) -> Result<(Vec<u32>, Vec<[u8; 3]>), Error> {
    let mut current_depth = vec![];
    let mut current_norm = vec![];

    // Work from front to back, so we can bail out early if the image is full
    for z in (0..workers_per_side).rev() {
        // Corner position in [0, workers_per_side] coordinates
        let mut corner = nalgebra::Vector3::new(
            index / workers_per_side,
            index % workers_per_side,
            z,
        )
        .cast::<f32>();
        // Corner position in [-1, 1] coordinates
        corner = (corner * 2.0 / workers_per_side as f32).add_scalar(-1.0);

        // Scale of each tile
        let scale = 2.0 / workers_per_side as f32;

        // Tile center
        let center = corner.add_scalar(scale / 2.0);

        let cfg = RenderConfig::<3> {
            image_size: image_size / workers_per_side,
            bounds: Bounds {
                center,
                size: scale / 2.0,
            },
            ..RenderConfig::default()
        };

        // Special case for the first tile, which can be copied over
        let (mut depth, norm) = cfg.run(shape.clone())?;
        for d in &mut depth {
            if *d > 0 {
                *d += (z * image_size / workers_per_side) as u32;
            }
        }
        if current_depth.is_empty() {
            current_depth = depth;
            current_norm = norm;
        } else {
            let mut all = true;
            for i in 0..depth.len() {
                if depth[i] > current_depth[i] {
                    current_depth[i] = depth[i];
                    current_norm[i] = norm[i];
                }
                all &= current_depth[i] == 0;
            }
            if all {
                break;
            }
        }
    }
    Ok((current_depth, current_norm))
}
