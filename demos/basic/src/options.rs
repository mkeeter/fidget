use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
pub struct Options {
    /// Main action
    #[clap(subcommand)]
    pub action: ActionCommand,

    /// Input file
    #[clap(short, long)]
    pub input: Option<PathBuf>,

    /// Input file
    #[clap(short = 's', long, value_enum, default_value_t = HardcodedShape::SphereAsm)]
    pub hardcoded_shape: HardcodedShape,

    /// Evaluator flavor
    #[clap(short, long, value_enum, default_value_t = EvalMode::Jit)]
    pub eval: EvalMode,

    /// Number of threads to use
    #[clap(long, default_value_t = 0)]
    pub num_threads: usize,

    /// Number of times to render (for benchmarking)
    #[clap(long, default_value_t = 1)]
    pub num_repeats: usize,
}

#[derive(ValueEnum, Clone)]
pub enum EvalMode {
    #[cfg(feature = "jit")]
    Jit,
    Vm,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum HardcodedShape {
    SphereAsm,
    SphereTree,
}

#[derive(Subcommand)]
pub enum ActionCommand {
    Render2d {
        #[clap(flatten)]
        settings: ImageSettings,

        /// Use brute-force (pixel-by-pixel) evaluation
        #[clap(short, long)]
        brute: bool,

        /// Render as a color-gradient SDF
        #[clap(long)]
        sdf: bool,
    },

    Render3d {
        #[clap(flatten)]
        settings: ImageSettings,

        /// Render in color
        #[clap(long)]
        color: bool,

        /// Render using an isometric perspective
        #[clap(long)]
        isometric: bool,

        /// Rotate camera
        #[clap(long, default_value_t = true)]
        rotate: bool,

        /// Rotate camera
        #[clap(short, long, default_value_t = 1.0)]
        scale: f32,
    },

    Mesh {
        #[clap(flatten)]
        settings: MeshSettings,
    },
}

#[derive(Parser)]
pub struct ImageSettings {
    /// Image size
    #[clap(long = "image-size", default_value_t = 1024)]
    pub size: u32,

    /// Name of a `.png` file to write
    #[clap(short, long)]
    pub output: Option<PathBuf>,
}

#[derive(Parser)]
pub struct MeshSettings {
    /// Octree depth
    #[clap(short, long, default_value_t = 6)]
    pub depth: u8,

    /// Name of a `.stl` file to write
    #[clap(short, long)]
    pub output: Option<PathBuf>,
}

pub fn parse_options() -> Options {
    Options::parse()
}
