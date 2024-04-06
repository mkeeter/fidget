use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::Parser;

const END_OF_ITEM: u8 = u8::MAX;

/// Read and decode a libfive binary `.frep` file (using packed opcodes)
#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    input: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut f = std::fs::File::open(args.input)?;
    let f = &mut f;

    let start = read_char(f)?;
    if start != 'T' {
        bail!("invalid start character {start:?}");
    }
    let _name = read_string(f)?;
    let _doc = read_string(f)?;

    for i in 0.. {
        let tag = read_byte(f)?;
        if tag == END_OF_ITEM {
            break;
        }
        print!("_{i:x}");
        let (name, args) = match tag {
            0 => bail!("read invalid opcode (0)"),
            1 => {
                let v = read_f32(f)?;
                println!(" const {v}");
                continue;
            }
            2 => ("var-x", 0),
            3 => ("var-y", 0),
            4 => ("var-z", 0),
            5 => bail!("VAR_FREE is not supported"),
            6 => bail!("CONST_VAR is not supported"),

            7 => ("square", 1),
            8 => ("sqrt", 1),
            9 => ("neg", 1),
            10 => ("sin", 1),
            11 => ("cos", 1),
            12 => ("tan", 1),
            13 => ("asin", 1),
            14 => ("acos", 1),
            15 => ("atan", 1),
            16 => ("exp", 1),
            17 => ("abs", 1),
            18 => ("ln", 1),
            19 => ("recip", 1),

            20 => ("add", 2),
            21 => ("mul", 2),
            22 => ("min", 2),
            23 => ("max", 2),
            24 => ("sub", 2),
            25 => ("div", 2),
            26 => bail!("OP_ATAN2 is not supported"),
            27 => bail!("OP_POW is not supported"),
            28 => bail!("OP_NTH_ROOT is not supported"),
            29 => ("OP_MOD", 2),
            30 => bail!("OP_NANFILL is not supported"),
            31 => ("compare", 2),
            32 => bail!("oracles are not supported"),
            op => bail!("unknown opcode {op}"),
        };
        print!(" {name}");

        // Arguments are encoded in reverse order
        let args = (0..args)
            .map(|_| read_u32(f))
            .collect::<Result<Vec<u32>, _>>()?;
        for a in args.iter().rev() {
            print!(" _{a:x}");
        }
        println!();
    }

    Ok(())
}

fn read_string(f: &mut File) -> Result<String> {
    let c = read_char(f)?;
    if c != '"' {
        bail!("invalid string start: {c:?}");
    }
    let mut out = String::new();
    loop {
        let c = read_char(f)?;
        if c == '"' {
            break;
        } else {
            out.push(c);
        }
    }
    Ok(out)
}

fn read_byte(f: &mut File) -> Result<u8> {
    let mut b = [0u8];
    f.read_exact(&mut b)?;
    Ok(b[0])
}

fn read_char(f: &mut File) -> Result<char> {
    let b = read_byte(f)?;
    Ok(b.into())
}

fn read_u32(f: &mut File) -> Result<u32> {
    let mut b = [0u8; 4];
    f.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_f32(f: &mut File) -> Result<f32> {
    let mut b = [0u8; 4];
    f.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}
