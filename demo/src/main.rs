use std::time::Instant;

use clap::Parser;
use jitfive::{compiler::Compiler, context::Context, program::Program};

/// Simple test program
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Name of a `.dot` file to write
    #[clap(short, long)]
    dot: Option<String>,

    /// Name of a `.png` file to write
    #[clap(short, long)]
    image: Option<String>,

    /// Render using the GPU
    #[clap(short, long, requires = "image")]
    gpu: bool,

    /// Name of a `.metal` file to write
    #[clap(short, long)]
    metal: Option<String>,

    /// Image size
    #[clap(short, long, requires = "image", default_value = "100")]
    size: usize,

    /// Name of the model file to load
    filename: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let now = Instant::now();
    let args = Args::parse();
    let mut file = std::fs::File::open(args.filename)?;
    let (ctx, node) = Context::from_text(&mut file)?;
    println!("Loaded file in {:?}", now.elapsed());

    let now = Instant::now();
    let compiler = Compiler::new(&ctx, node);
    println!("Build Compiler in {:?}", now.elapsed());

    let now = Instant::now();
    let prog = Program::from_compiler(&compiler);
    println!("Built Program in {:?}", now.elapsed());

    if let Some(dot) = args.dot {
        let mut out = std::fs::File::create(dot)?;
        compiler.write_dot_grouped(&mut out)?;
    }
    if let Some(metal) = args.metal {
        std::fs::write(metal, prog.to_metal())?;
    }
    if let Some(img) = args.image {
        let out = if args.gpu {
            gpu::render(&prog, args.size)
        } else {
            ctx.render_2d(node, args.size)?
        };
        let buffer: Vec<u8> = out
            .into_iter()
            .map(|b| if b { u8::MAX } else { 0 })
            .collect();
        image::save_buffer(
            img,
            &buffer,
            args.size as u32,
            args.size as u32,
            image::ColorType::L8,
        )?;
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////////////

mod gpu {
    use super::*;
    use piet_gpu_hal::{BindType, ComputePassDescriptor, ShaderCode};
    use piet_gpu_hal::{BufferUsage, Instance, InstanceFlags, Session};

    pub fn render(prog: &Program, size: usize) -> Vec<bool> {
        let now = Instant::now();
        let (instance, _) =
            Instance::new(None, InstanceFlags::empty()).unwrap();
        println!("Got instance at {:?}", now.elapsed());

        let shader_f = prog.to_metal();
        println!("Got shader text at {:?}", now.elapsed());
        let cfg = prog.config();

        let thread_count: usize = size * size;
        let mut dst: Vec<f32> = Default::default();

        unsafe {
            let device = instance.device(None).unwrap();
            println!("Got device at {:?}", now.elapsed());
            let session = Session::new(device);
            println!("Got session at {:?}", now.elapsed());

            let mut vars = vec![];
            for x in 0..size {
                let x = 1.0 - ((x as f32) / (size - 1) as f32) * 2.0;
                for y in 0..size {
                    let y = ((y as f32) / (size - 1) as f32) * 2.0 - 1.0;
                    // TODO order
                    vars.push(x);
                    vars.push(y);
                }
            }
            let var_buf = session
                .create_buffer_init(
                    &vars,
                    BufferUsage::MAP_WRITE | BufferUsage::STORAGE,
                )
                .unwrap();

            let choices = std::iter::repeat(0b11)
                .take(thread_count * cfg.choice_count)
                .collect::<Vec<u8>>();
            let choice_buf = session
                .create_buffer_init(
                    &choices,
                    BufferUsage::MAP_WRITE | BufferUsage::STORAGE,
                )
                .unwrap();

            let out_buf = session
                .create_buffer(
                    u64::try_from(thread_count * std::mem::size_of::<f32>())
                        .unwrap(),
                    BufferUsage::STORAGE | BufferUsage::MAP_READ,
                )
                .unwrap();
            println!("Created buffers at {:?}", now.elapsed());

            let pipeline = session
                .create_compute_pipeline(
                    ShaderCode::Msl(&shader_f),
                    &[
                        BindType::BufReadOnly,
                        BindType::BufReadOnly,
                        BindType::Buffer,
                    ],
                )
                .unwrap();
            println!("Created pipeline at {:?}", now.elapsed());

            let descriptor_set = session
                .create_simple_descriptor_set(
                    &pipeline,
                    &[&var_buf, &choice_buf, &out_buf],
                )
                .unwrap();

            let query_pool = session.create_query_pool(2).unwrap();

            let mut cmd_buf = session.cmd_buf().unwrap();
            cmd_buf.begin();
            cmd_buf.reset_query_pool(&query_pool);
            {
                let mut pass = cmd_buf.begin_compute_pass(
                    &ComputePassDescriptor::timer(&query_pool, 0, 1),
                );
                pass.dispatch(
                    &pipeline,
                    &descriptor_set,
                    (thread_count.try_into().unwrap(), 1, 1),
                    (1, 1, 1),
                );
                pass.end();
            }

            cmd_buf.finish_timestamps(&query_pool);
            cmd_buf.host_barrier();
            cmd_buf.finish();

            println!("Starting run {:?}", now.elapsed());
            let submitted = session.run_cmd_buf(cmd_buf, &[], &[]).unwrap();
            submitted.wait().unwrap();
            println!("Done at {:?}", now.elapsed());
            let timestamps = session.fetch_query_pool(&query_pool);

            out_buf.read(&mut dst).unwrap();
            for (i, val) in dst.iter().enumerate().take(16) {
                println!("{}: {}", i, val);
            }
            println!("{:?}", timestamps);
        }
        dst.into_iter().map(|i| i < 0.0).collect()
    }
}
