use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::time::Instant;
use std::{
    io::Read,
    sync::{Arc, Mutex},
};

use anyhow::{Context as _, Result, bail};
use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use env_logger::Env;
use log::info;

use fidget::{
    context::{Context, Node},
    rhai::FromDynamic,
};

/// Simple test program
#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(subcommand)]
    cmd: Command,
}

#[derive(Subcommand)]
enum Command {
    Render2d {
        #[clap(flatten)]
        settings: ImageSettings,

        /// Render mode
        #[clap(long, value_enum, default_value_t)]
        mode: RenderMode2D,

        /// Center point of the render
        #[clap(
            long, default_value = "0,0", allow_hyphen_values = true,
            value_parser = parse_vec2
        )]
        center: [f32; 2],
    },

    Render3d {
        #[clap(flatten)]
        settings: ImageSettings,

        /// Render mode
        #[clap(long, value_enum, default_value_t)]
        mode: RenderMode3DArg,

        /// Rotation about the Y axis (in degrees)
        #[clap(long, default_value_t = 0.0, allow_hyphen_values = true)]
        pitch: f32,

        /// Rotation about the X axis (in degrees)
        #[clap(long, default_value_t = 0.0, allow_hyphen_values = true)]
        yaw: f32,

        /// Rotation about the Z axis (in degrees)
        #[clap(long, default_value_t = 0.0, allow_hyphen_values = true)]
        roll: f32,

        /// Center point of the render
        #[clap(
            long, default_value = "0,0,0", allow_hyphen_values = true,
            value_parser = parse_vec3
        )]
        center: [f32; 3],

        /// Flatten values on the Z axis to prevent screen clipping
        #[clap(long, default_value_t = 1.0)]
        zflatten: f32,

        /// Render using an isometric perspective
        #[clap(long, conflicts_with = "perspective")]
        isometric: bool,

        /// Apply SSAO to a shaded image
        ///
        /// Only compatible with `--mode=shaded`
        #[clap(long)]
        ssao: bool,

        /// Skip denoising of normals
        ///
        /// Incompatible with `--mode=heightmap`
        #[clap(long)]
        no_denoise: bool,

        /// Strength of perspective transform
        #[clap(long)]
        perspective: Option<f32>,
    },
    Mesh {
        #[clap(flatten)]
        settings: MeshSettings,
    },
    /// Print generated shader code
    Shader {
        #[clap(long, value_enum)]
        name: ShaderName,
    },
}

#[derive(ValueEnum, Clone)]
enum ShaderName {
    IntervalTiles,
    VoxelRay,
    Backfill,
}

#[derive(ValueEnum, Default, Clone)]
enum EvalMode {
    #[default]
    Vm,

    #[cfg(feature = "jit")]
    Jit,
}

#[derive(strum::EnumDiscriminants, Clone, Debug)]
#[strum_discriminants(name(RenderMode3DArg), derive(ValueEnum))]
enum RenderMode3D {
    /// Pixels are colored based on height
    Heightmap,
    /// Pixels are colored based on normals
    Normals { denoise: bool },
    /// Pixels are shaded
    Shaded { denoise: bool, ssao: bool },
    /// Raw (unblurred) SSAO occlusion, for debugging
    RawOcclusion { denoise: bool },
    /// Blurred SSAO occlusion, for debugging
    BlurredOcclusion { denoise: bool },
}

impl Default for RenderMode3DArg {
    fn default() -> Self {
        Self::Heightmap
    }
}

#[derive(ValueEnum, Default, Clone)]
enum RenderMode2D {
    /// Pixels are colored based on interval results
    #[default]
    Debug,
    /// Monochrome rendering (white-on-black)
    Mono,
    /// Signed distance field visualization
    Sdf,
    /// Brute-force (pixel-by-pixel) evaluation
    Brute,
}

#[derive(Parser)]
struct ImageSettings {
    #[clap(flatten)]
    script: ScriptSettings,

    /// Name of a `.png` file to write
    #[clap(short, long)]
    out: Option<PathBuf>,

    /// Evaluator flavor
    #[clap(short, long, value_enum, default_value_t)]
    eval: EvalMode,

    /// Perform per-pixel evaluation using WGPU
    ///
    /// (interval evaluation still uses the mode specified by `--eval`)
    #[clap(long)]
    wgpu: bool,

    /// Number of threads to use
    #[clap(short, long)]
    threads: Option<NonZeroUsize>,

    /// Number of times to render (for benchmarking)
    #[clap(short = 'N', default_value_t = 1)]
    n: usize,

    /// Image size
    #[clap(short, long, default_value_t = 128)]
    size: u32,

    /// Scale applied to the model before rendering
    #[clap(long, default_value_t = 1.0)]
    scale: f32,
}

impl ImageSettings {
    fn threads(&self) -> Option<fidget::render::ThreadPool> {
        match self.threads {
            Some(n) if n.get() == 1 => None,
            Some(n) => Some(fidget::render::ThreadPool::Custom(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n.get())
                    .build()
                    .unwrap(),
            )),
            None => Some(fidget::render::ThreadPool::Global),
        }
    }
}

#[derive(Parser)]
struct MeshSettings {
    #[clap(flatten)]
    script: ScriptSettings,

    /// Octree depth
    #[clap(short, long)]
    depth: u8,

    /// Name of a `.stl` file to write
    #[clap(short, long)]
    out: Option<PathBuf>,

    /// Evaluator flavor
    #[clap(short, long, value_enum, default_value_t)]
    eval: EvalMode,

    /// Number of threads to use
    #[clap(short, long)]
    threads: Option<NonZeroUsize>,

    /// Number of times to render (for benchmarking)
    #[clap(short = 'N', default_value_t = 1)]
    n: usize,

    /// Scale applied to the model before rendering
    #[clap(long, default_value_t = 1.0)]
    scale: f32,

    /// Center point of the render
    #[clap(
        long, default_value = "0,0,0", allow_hyphen_values = true,
        value_parser = parse_vec3
    )]
    center: [f32; 3],
}

#[derive(Parser)]
struct ScriptSettings {
    /// Input file
    #[clap(short, long)]
    input: PathBuf,

    /// Input file type
    #[clap(long, default_value_t, value_enum)]
    r#type: ScriptType,
}

#[derive(clap::ValueEnum, Copy, Clone, Default, Debug)]
enum ScriptType {
    /// Select based on file extension
    #[default]
    Auto,
    /// Rhai script
    Rhai,
    /// Raw VM instructions
    Vm,
}

fn parse_vec3(s: &str) -> Result<[f32; 3]> {
    let parts: Vec<f32> = s
        .split(',')
        .map(str::trim)
        .map(|num| num.parse::<f32>())
        .collect::<Result<Vec<f32>, _>>()
        .with_context(|| format!("failed to parse vec3 from '{s}'"))?;

    if parts.len() == 3 {
        Ok([parts[0], parts[1], parts[2]])
    } else {
        bail!(
            "expected 3 comma-separated floating point numbers, got {}",
            parts.len()
        )
    }
}

fn parse_vec2(s: &str) -> Result<[f32; 2]> {
    let parts: Vec<f32> = s
        .split(',')
        .map(str::trim)
        .map(|num| num.parse::<f32>())
        .collect::<Result<Vec<f32>, _>>()
        .with_context(|| format!("failed to parse vec2 from '{s}'"))?;

    if parts.len() == 2 {
        Ok([parts[0], parts[1]])
    } else {
        bail!(
            "expected 2 comma-separated floating point numbers, got {}",
            parts.len()
        )
    }
}

////////////////////////////////////////////////////////////////////////////////
fn run3d<F: fidget::eval::Function + fidget::render::RenderHints>(
    shape: fidget::shape::Shape<F>,
    settings: &ImageSettings,
    mode: RenderMode3D,
) -> Vec<u8> {
    let threads = settings.threads();
    let threads = threads.as_ref();
    let cfg = fidget::render::VoxelRenderConfig {
        image_size: fidget::render::VoxelSize::from(settings.size),
        tile_sizes: F::tile_sizes_3d(),
        threads,
        ..Default::default()
    };

    let mut image = Default::default();

    let start = std::time::Instant::now();
    for _ in 0..settings.n {
        image = cfg.run(shape.clone()).unwrap();
    }
    info!(
        "Rendered {}x at {:?} ms/frame",
        settings.n,
        start.elapsed().as_micros() as f64 / 1000.0 / (settings.n as f64)
    );

    let start = std::time::Instant::now();
    let out = match mode {
        RenderMode3D::Normals { denoise } => {
            let image = if denoise {
                fidget::render::effects::denoise_normals(&image, threads)
            } else {
                image
            };
            image
                .into_iter()
                .flat_map(|p| {
                    if p.depth > 0 {
                        let c = p.to_color();
                        [c[0], c[1], c[2], 255]
                    } else {
                        [0, 0, 0, 0]
                    }
                })
                .collect()
        }
        RenderMode3D::Shaded { ssao, denoise } => {
            let image = if denoise {
                fidget::render::effects::denoise_normals(&image, threads)
            } else {
                image
            };
            let color =
                fidget::render::effects::apply_shading(&image, ssao, threads);
            image
                .into_iter()
                .zip(color)
                .flat_map(|(p, c)| {
                    if p.depth > 0 {
                        [c[0], c[1], c[2], 255]
                    } else {
                        [0, 0, 0, 0]
                    }
                })
                .collect()
        }
        RenderMode3D::RawOcclusion { denoise } => {
            let image = if denoise {
                fidget::render::effects::denoise_normals(&image, threads)
            } else {
                image
            };
            let ssao = fidget::render::effects::compute_ssao(&image, threads);
            ssao.into_iter()
                .flat_map(|p| {
                    if p.is_nan() {
                        [255; 4]
                    } else {
                        let v = (p * 255.0).min(255.0) as u8;
                        [v, v, v, 255]
                    }
                })
                .collect()
        }
        RenderMode3D::BlurredOcclusion { denoise } => {
            let image = if denoise {
                fidget::render::effects::denoise_normals(&image, threads)
            } else {
                image
            };
            let ssao = fidget::render::effects::compute_ssao(&image, threads);
            let blurred = fidget::render::effects::blur_ssao(&ssao, threads);
            blurred
                .into_iter()
                .flat_map(|p| {
                    if p.is_nan() {
                        [255; 4]
                    } else {
                        let v = (p * 255.0).min(255.0) as u8;
                        [v, v, v, 255]
                    }
                })
                .collect()
        }
        RenderMode3D::Heightmap => {
            let z_max = image.iter().map(|p| p.depth).max().unwrap_or(1);
            image
                .into_iter()
                .flat_map(|p| {
                    if p.depth > 0 {
                        let z = (p.depth * 255 / z_max) as u8;
                        [z, z, z, 255]
                    } else {
                        [0, 0, 0, 0]
                    }
                })
                .collect()
        }
    };
    info!("Post-processed image in {:?}", start.elapsed());

    out
}

////////////////////////////////////////////////////////////////////////////////

fn run2d<F: fidget::eval::Function + fidget::render::RenderHints>(
    shape: fidget::shape::Shape<F>,
    settings: &ImageSettings,
    mode: RenderMode2D,
) -> Vec<u8> {
    if matches!(mode, RenderMode2D::Brute) {
        let tape = shape.float_slice_tape(Default::default());
        let mut eval = fidget::shape::Shape::<F>::new_float_slice_eval();
        let mut out: Vec<bool> = vec![];
        for _ in 0..settings.n {
            let mut xs = vec![];
            let mut ys = vec![];
            let div = (settings.size - 1) as f64;
            for i in 0..settings.size {
                let y = -(-1.0 + 2.0 * (i as f64) / div);
                for j in 0..settings.size {
                    let x = -1.0 + 2.0 * (j as f64) / div;
                    xs.push(x as f32);
                    ys.push(y as f32);
                }
            }
            let zs = vec![0.0; xs.len()];
            let values = eval.eval(&tape, &xs, &ys, &zs).unwrap();
            out = values.iter().map(|v| *v <= 0.0).collect();
        }
        // Convert from Vec<bool> to an image
        out.into_iter()
            .map(|b| if b { [u8::MAX; 4] } else { [0, 0, 0, 255] })
            .flat_map(|i| i.into_iter())
            .collect()
    } else {
        let threads = settings.threads();
        let cfg = fidget::render::ImageRenderConfig {
            image_size: fidget::render::ImageSize::from(settings.size),
            tile_sizes: F::tile_sizes_2d(),
            threads: threads.as_ref(),
            pixel_perfect: matches!(mode, RenderMode2D::Sdf),
            ..Default::default()
        };
        let mut image = fidget::render::Image::default();
        match mode {
            RenderMode2D::Mono => {
                for _ in 0..settings.n {
                    let tmp = cfg.run::<_>(shape.clone()).unwrap();
                    image = fidget::render::effects::to_rgba_bitmap(
                        tmp,
                        false,
                        cfg.threads,
                    );
                }
            }
            RenderMode2D::Sdf => {
                for _ in 0..settings.n {
                    let tmp = cfg.run(shape.clone()).unwrap();
                    image = fidget::render::effects::to_rgba_distance(
                        tmp,
                        cfg.threads,
                    );
                }
            }
            RenderMode2D::Debug => {
                for _ in 0..settings.n {
                    let tmp = cfg.run::<_>(shape.clone()).unwrap();
                    image = fidget::render::effects::to_debug_bitmap(
                        tmp,
                        cfg.threads,
                    );
                }
            }
            RenderMode2D::Brute => unreachable!(),
        }
        image.into_iter().flatten().collect()
    }
}

fn run_wgpu_3d<F: fidget::eval::MathFunction + fidget::render::RenderHints>(
    shape: fidget::shape::Shape<F>,
    settings: &ImageSettings,
) -> Vec<u8> {
    let start = Instant::now();
    let image_size = fidget::render::VoxelSize::from(settings.size);
    let mut image = vec![];
    let threads = settings.threads();

    let mut ctx = fidget::wgpu::VoxelRayContext::new().unwrap();

    // Send over our image pixels
    for _i in 0..settings.n {
        // Note that this copies the bytecode each time
        let s = std::time::Instant::now();
        image = ctx
            .run_3d(
                shape.clone(),
                fidget::render::VoxelRenderConfig {
                    image_size,
                    tile_sizes: fidget::render::TileSizes::new(&[128, 64])
                        .unwrap(),
                    threads: threads.as_ref(),
                    ..Default::default()
                },
            )
            .unwrap()
            .unwrap();
        println!(" => {:?}", s.elapsed());
    }
    info!(
        "Rendered {}x at {:?} ms/frame",
        settings.n,
        start.elapsed().as_micros() as f64 / 1000.0 / (settings.n as f64)
    );

    image.into_iter().flat_map(|b| b.to_le_bytes()).collect()
}

////////////////////////////////////////////////////////////////////////////////

fn run_mesh<F: fidget::eval::Function + fidget::render::RenderHints>(
    shape: fidget::shape::Shape<F>,
    settings: &MeshSettings,
) -> (fidget::mesh::Mesh, std::time::Duration, std::time::Duration) {
    let mut mesh = fidget::mesh::Mesh::new();

    // Transform the shape based on our render settings
    let s = 1.0 / settings.scale;
    let scale = nalgebra::Scale3::new(s, s, s);
    let center = nalgebra::Translation3::new(
        -settings.center[0],
        -settings.center[1],
        -settings.center[2],
    );
    let t = center.to_homogeneous() * scale.to_homogeneous();
    let shape = shape.apply_transform(t);

    let threads = match settings.threads {
        Some(n) if n.get() == 1 => None,
        Some(n) => Some(fidget::render::ThreadPool::Custom(
            rayon::ThreadPoolBuilder::new()
                .num_threads(n.get())
                .build()
                .unwrap(),
        )),
        None => Some(fidget::render::ThreadPool::Global),
    };
    let threads = threads.as_ref();

    let mut octree_time = std::time::Duration::ZERO;
    let mut mesh_time = std::time::Duration::ZERO;
    for _ in 0..settings.n {
        let settings = fidget::mesh::Settings {
            depth: settings.depth,
            threads,
            ..Default::default()
        };
        let start = std::time::Instant::now();
        let octree = fidget::mesh::Octree::build(&shape, settings);
        octree_time += start.elapsed();

        let start = std::time::Instant::now();
        mesh = octree.walk_dual(settings);
        mesh_time += start.elapsed();
    }
    (mesh, octree_time, mesh_time)
}

fn load_script(settings: &ScriptSettings) -> Result<(Context, Node)> {
    let now = Instant::now();
    let mut file = std::fs::File::open(&settings.input)?;
    let ty = match settings.r#type {
        ScriptType::Auto => {
            let ext = settings
                .input
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| s.to_ascii_lowercase());
            match ext.as_deref() {
                Some("rhai") => ScriptType::Rhai,
                Some("vm") => ScriptType::Vm,
                Some(s) => {
                    bail!(
                        "Unknown extension '{s}', should be '.rhai' or '.vm' \
                         (or specify --type)"
                    )
                }
                None => bail!("cannot detect script type without extension"),
            }
        }
        s => s,
    };
    let (ctx, root) = match ty {
        ScriptType::Vm => Context::from_text(&mut file)?,
        ScriptType::Rhai => {
            let mut engine = fidget::rhai::engine();
            let mut script = String::new();
            file.read_to_string(&mut script)
                .context("failed to read script to string")?;
            let out = Arc::new(Mutex::new(None));
            let out_ = out.clone();
            engine.register_fn(
                "draw",
                move |ctx: rhai::NativeCallContext,
                      d: rhai::Dynamic|
                      -> Result<(), Box<rhai::EvalAltResult>> {
                    let t = fidget::context::Tree::from_dynamic(&ctx, d, None)?;
                    let mut out = out_.lock().unwrap();
                    if out.is_some() {
                        return Err("can only draw one shape".into());
                    }
                    *out = Some(t);
                    Ok(())
                },
            );
            engine.run(&script)?;
            if let Some(tree) = out.lock().unwrap().take() {
                let mut ctx = Context::new();
                let node = ctx.import(&tree);
                (ctx, node)
            } else {
                bail!("script must include a draw(tree) call");
            }
        }
        ScriptType::Auto => unreachable!(),
    };
    info!("Loaded file in {:?}", now.elapsed());
    Ok((ctx, root))
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .init();

    let args = Args::parse();

    match args.cmd {
        Command::Render2d {
            settings,
            mode,
            center,
        } => {
            let (ctx, root) = load_script(&settings.script)?;
            let start = Instant::now();
            let s = 1.0 / settings.scale;
            let scale = nalgebra::Scale3::new(s, s, s);
            let center =
                nalgebra::Translation3::new(-center[0], -center[1], 0.0);
            let t = center.to_homogeneous() * scale.to_homogeneous();
            let buffer = if settings.wgpu {
                bail!("cannot use wgpu for 2D rendering");
            } else {
                match settings.eval {
                    #[cfg(feature = "jit")]
                    EvalMode::Jit => {
                        let shape = fidget::jit::JitShape::new(&ctx, root)?
                            .apply_transform(t);

                        info!("Built shape in {:?}", start.elapsed());
                        run2d(shape, &settings, mode)
                    }
                    EvalMode::Vm => {
                        let shape = fidget::vm::VmShape::new(&ctx, root)?
                            .apply_transform(t);
                        info!("Built shape in {:?}", start.elapsed());
                        run2d(shape, &settings, mode)
                    }
                }
            };

            info!(
                "Rendered {}x at {:?} ms/frame",
                settings.n,
                start.elapsed().as_micros() as f64
                    / 1000.0
                    / (settings.n as f64)
            );
            if let Some(out) = settings.out {
                image::save_buffer(
                    out,
                    &buffer,
                    settings.size,
                    settings.size,
                    image::ColorType::Rgba8,
                )?;
            }
        }
        Command::Render3d {
            settings,
            mode,
            isometric,
            pitch,
            zflatten,
            yaw,
            roll,
            center,
            perspective,
            ssao,
            no_denoise,
        } => {
            // Manual checking of arguments
            // TODO this isn't printed in color
            if ssao && !matches!(mode, RenderMode3DArg::Shaded) {
                let mut cmd = Args::command();
                let sub = cmd.find_subcommand_mut("render3d").unwrap();
                let error = sub.error(
                    clap::error::ErrorKind::ArgumentConflict,
                    "`--ssao` is only allowed when `--mode=shaded`",
                );
                error.exit();
            }
            if no_denoise && matches!(mode, RenderMode3DArg::Heightmap) {
                let mut cmd = Args::command();
                let sub = cmd.find_subcommand_mut("render3d").unwrap();
                let error = sub.error(
                    clap::error::ErrorKind::ArgumentConflict,
                    "`--no-denoise` is not allowed when `--mode=heightmap`",
                );
                error.exit();
            }
            let denoise = !no_denoise;
            let mode = match mode {
                RenderMode3DArg::Shaded => {
                    RenderMode3D::Shaded { ssao, denoise }
                }
                RenderMode3DArg::Heightmap => RenderMode3D::Heightmap,
                RenderMode3DArg::BlurredOcclusion => {
                    RenderMode3D::BlurredOcclusion { denoise }
                }
                RenderMode3DArg::RawOcclusion => {
                    RenderMode3D::RawOcclusion { denoise }
                }
                RenderMode3DArg::Normals => RenderMode3D::Normals { denoise },
            };
            let (ctx, root) = load_script(&settings.script)?;
            let start = Instant::now();
            let s = 1.0 / settings.scale;
            let scale = nalgebra::Scale3::new(s, s, s);
            let pitch = nalgebra::Rotation3::new(
                nalgebra::Vector3::x() * pitch * std::f32::consts::PI / 180.0,
            );
            let yaw = nalgebra::Rotation3::new(
                nalgebra::Vector3::y() * yaw * std::f32::consts::PI / 180.0,
            );
            let roll = nalgebra::Rotation3::new(
                nalgebra::Vector3::z() * roll * std::f32::consts::PI / 180.0,
            );
            let center =
                nalgebra::Translation3::new(-center[0], -center[1], -center[2]);

            let perspective =
                perspective.unwrap_or(if isometric { 0.0 } else { 0.3 });
            let mut camera = nalgebra::Transform3::identity();
            *camera.matrix_mut().get_mut((3, 2)).unwrap() = perspective;

            let zflatten = nalgebra::Scale3::new(1.0, 1.0, zflatten);

            let t = center.to_homogeneous()
                * yaw.to_homogeneous()
                * roll.to_homogeneous()
                * pitch.to_homogeneous()
                * scale.to_homogeneous()
                * camera.to_homogeneous()
                * zflatten.to_homogeneous();

            let buffer = if settings.wgpu {
                if !matches!(mode, RenderMode3D::Heightmap) {
                    bail!("cannot use WGPU rendering in mode {mode:?}");
                }
                match settings.eval {
                    #[cfg(feature = "jit")]
                    EvalMode::Jit => {
                        let shape = fidget::jit::JitShape::new(&ctx, root)?
                            .apply_transform(t);
                        info!("Built shape in {:?}", start.elapsed());
                        run_wgpu_3d(shape, &settings)
                    }
                    EvalMode::Vm => {
                        let shape = fidget::vm::VmShape::new(&ctx, root)?
                            .apply_transform(t);
                        info!("Built shape in {:?}", start.elapsed());
                        run_wgpu_3d(shape, &settings)
                    }
                }
            } else {
                match settings.eval {
                    #[cfg(feature = "jit")]
                    EvalMode::Jit => {
                        let shape = fidget::jit::JitShape::new(&ctx, root)?
                            .apply_transform(t);
                        info!("Built shape in {:?}", start.elapsed());
                        run3d(shape, &settings, mode)
                    }
                    EvalMode::Vm => {
                        let shape = fidget::vm::VmShape::new(&ctx, root)?
                            .apply_transform(t);
                        info!("Built shape in {:?}", start.elapsed());
                        run3d(shape, &settings, mode)
                    }
                }
            };

            if let Some(out) = settings.out {
                info!("Writing image to {out:?}");
                image::save_buffer(
                    out,
                    &buffer,
                    settings.size,
                    settings.size,
                    image::ColorType::Rgba8,
                )?;
            }
        }
        Command::Mesh { settings } => {
            let (ctx, root) = load_script(&settings.script)?;
            let start = Instant::now();
            let (mesh, octree_time, mesh_time) = match settings.eval {
                #[cfg(feature = "jit")]
                EvalMode::Jit => {
                    let shape = fidget::jit::JitShape::new(&ctx, root)?;
                    info!("Built shape in {:?}", start.elapsed());
                    run_mesh(shape, &settings)
                }
                EvalMode::Vm => {
                    let shape = fidget::vm::VmShape::new(&ctx, root)?;
                    info!("Built shape in {:?}", start.elapsed());
                    run_mesh(shape, &settings)
                }
            };
            info!(
                "Rendered {}x at {:?} ms/iter",
                settings.n,
                start.elapsed().as_micros() as f64
                    / 1000.0
                    / (settings.n as f64)
            );
            info!(
                "  Octree construction: {:?} ms/iter",
                octree_time.as_micros() as f64 / 1000.0 / (settings.n as f64)
            );
            info!(
                "  Mesh construction: {:?} ms/iter",
                mesh_time.as_micros() as f64 / 1000.0 / (settings.n as f64)
            );
            if let Some(out) = settings.out {
                info!("Writing STL to {out:?}");
                mesh.write_stl(&mut std::fs::File::create(out)?)?;
            }
        }
        Command::Shader { name } => match name {
            ShaderName::VoxelRay => {
                println!("{}", fidget::wgpu::voxel_ray_shader())
            }
            ShaderName::IntervalTiles => {
                println!("{}", fidget::wgpu::interval_tiles_shader())
            }
            ShaderName::Backfill => {
                println!("{}", fidget::wgpu::backfill_shader())
            }
        },
    }

    Ok(())
}
