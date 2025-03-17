use std::io::Read;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context as _, Result};
use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use env_logger::Env;
use log::info;

use fidget::context::{Context, Node};

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
}

#[derive(ValueEnum, Default, Clone)]
enum EvalMode {
    #[default]
    Vm,

    #[cfg(feature = "jit")]
    Jit,
}

#[derive(strum::EnumDiscriminants, Clone)]
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
    /// Approximate signed distance field visualization
    SdfApprox,
    /// Exact signed distance field visualization (more expensive)
    SdfExact,
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
    #[clap(short, long, default_value_t = NonZeroUsize::new(8).unwrap())]
    threads: NonZeroUsize,

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

////////////////////////////////////////////////////////////////////////////////
fn run3d<F: fidget::eval::Function + fidget::render::RenderHints>(
    shape: fidget::shape::Shape<F>,
    settings: &ImageSettings,
    mode: RenderMode3D,
) -> Vec<u8> {
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
    let cfg = fidget::render::VoxelRenderConfig {
        image_size: fidget::render::VoxelSize::from(settings.size),
        tile_sizes: F::tile_sizes_3d(),
        threads,
        ..Default::default()
    };

    let mut depth = Default::default();
    let mut norm = Default::default();

    let start = std::time::Instant::now();
    for _ in 0..settings.n {
        (depth, norm) = cfg.run(shape.clone()).unwrap();
    }
    info!(
        "Rendered {}x at {:?} ms/frame",
        settings.n,
        start.elapsed().as_micros() as f64 / 1000.0 / (settings.n as f64)
    );

    let start = std::time::Instant::now();
    let out = match mode {
        RenderMode3D::Normals { denoise } => {
            let norm = if denoise {
                fidget::render::effects::denoise_normals(&depth, &norm, threads)
            } else {
                norm
            };
            let color = norm.to_color();
            depth
                .into_iter()
                .zip(color)
                .flat_map(|(d, p)| {
                    if d > 0 {
                        [p[0], p[1], p[2], 255]
                    } else {
                        [0, 0, 0, 0]
                    }
                })
                .collect()
        }
        RenderMode3D::Shaded { ssao, denoise } => {
            let norm = if denoise {
                fidget::render::effects::denoise_normals(&depth, &norm, threads)
            } else {
                norm
            };
            let color = fidget::render::effects::apply_shading(
                &depth, &norm, ssao, threads,
            );
            depth
                .into_iter()
                .zip(color)
                .flat_map(|(d, p)| {
                    if d > 0 {
                        [p[0], p[1], p[2], 255]
                    } else {
                        [0, 0, 0, 0]
                    }
                })
                .collect()
        }
        RenderMode3D::RawOcclusion { denoise } => {
            let norm = if denoise {
                fidget::render::effects::denoise_normals(&depth, &norm, threads)
            } else {
                norm
            };
            let ssao =
                fidget::render::effects::compute_ssao(&depth, &norm, threads);
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
            let norm = if denoise {
                fidget::render::effects::denoise_normals(&depth, &norm, threads)
            } else {
                norm
            };
            let ssao =
                fidget::render::effects::compute_ssao(&depth, &norm, threads);
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
            let z_max = depth.iter().max().cloned().unwrap_or(1);
            depth
                .into_iter()
                .flat_map(|p| {
                    if p > 0 {
                        let z = (p * 255 / z_max) as u8;
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
        let cfg = fidget::render::ImageRenderConfig {
            image_size: fidget::render::ImageSize::from(settings.size),
            tile_sizes: F::tile_sizes_2d(),
            threads: threads.as_ref(),
            ..Default::default()
        };
        match mode {
            RenderMode2D::Mono => {
                let mut image = fidget::render::Image::default();
                for _ in 0..settings.n {
                    image = cfg
                        .run::<_, fidget::render::BitRenderMode>(shape.clone())
                        .unwrap();
                }
                image
                    .into_iter()
                    .flat_map(|a| if a { [255; 4] } else { [0, 0, 0, 255] })
                    .collect()
            }
            RenderMode2D::SdfExact => {
                let mut image = fidget::render::Image::default();
                for _ in 0..settings.n {
                    image = cfg
                        .run::<_, fidget::render::SdfPixelRenderMode>(
                            shape.clone(),
                        )
                        .unwrap();
                }
                image
                    .into_iter()
                    .flat_map(|a| [a[0], a[1], a[2], 255].into_iter())
                    .collect()
            }
            RenderMode2D::SdfApprox => {
                let mut image = fidget::render::Image::default();
                for _ in 0..settings.n {
                    image = cfg
                        .run::<_, fidget::render::SdfRenderMode>(shape.clone())
                        .unwrap();
                }
                image
                    .into_iter()
                    .flat_map(|a| [a[0], a[1], a[2], 255].into_iter())
                    .collect()
            }
            RenderMode2D::Debug => {
                let mut image = fidget::render::Image::default();
                for _ in 0..settings.n {
                    image = cfg
                        .run::<_, fidget::render::DebugRenderMode>(
                            shape.clone(),
                        )
                        .unwrap();
                }
                image
                    .into_iter()
                    .flat_map(|p| p.as_debug_color().into_iter())
                    .collect()
            }
            RenderMode2D::Brute => unreachable!(),
        }
    }
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

    let mut octree_time = std::time::Duration::ZERO;
    let mut mesh_time = std::time::Duration::ZERO;
    for _ in 0..settings.n {
        let settings = fidget::mesh::Settings {
            #[cfg(not(target_arch = "wasm32"))]
            threads: settings.threads.into(),
            #[cfg(target_arch = "wasm32")]
            threads: Default::default(),
            depth: settings.depth,
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
            let mut engine = fidget::rhai::Engine::new();
            let mut script = String::new();
            file.read_to_string(&mut script)
                .context("failed to read script to string")?;
            let out = engine.run(&script)?;
            if out.shapes.len() > 1 {
                bail!("can only render 1 shape");
            }
            let mut ctx = Context::new();
            let node = ctx.import(&out.shapes[0].tree);
            (ctx, node)
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
        Command::Render2d { settings, mode } => {
            let (ctx, root) = load_script(&settings.script)?;
            let start = Instant::now();
            let s = 1.0 / settings.scale;
            let scale = nalgebra::Scale3::new(s, s, s);
            let buffer = match settings.eval {
                #[cfg(feature = "jit")]
                EvalMode::Jit => {
                    let shape = fidget::jit::JitShape::new(&ctx, root)?
                        .apply_transform(scale.into());

                    info!("Built shape in {:?}", start.elapsed());
                    run2d(shape, &settings, mode)
                }
                EvalMode::Vm => {
                    let shape = fidget::vm::VmShape::new(&ctx, root)?
                        .apply_transform(scale.into());
                    info!("Built shape in {:?}", start.elapsed());
                    run2d(shape, &settings, mode)
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
            println!("got center {center:?}");

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

            let buffer = match settings.eval {
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
    }

    Ok(())
}
