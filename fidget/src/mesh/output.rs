//! Mesh output implementation
use super::Mesh;
use std::io::{BufWriter, Write};

impl Mesh {
    /// Writes a binary STL to the given output
    pub fn write_stl<F: std::io::Write>(
        &self,
        out: &mut F,
    ) -> Result<(), crate::Error> {
        // We're going to do many small writes and will typically be writing to
        // a file, so using a `BufWriter` saves excessive syscalls.
        let mut out = BufWriter::new(out);
        const HEADER: &[u8] = b"This is a binary STL file exported by Fidget";
        static_assertions::const_assert!(HEADER.len() <= 80);
        out.write_all(HEADER)?;
        out.write_all(&[0u8; 80 - HEADER.len()])?;
        out.write_all(&(self.triangles.len() as u32).to_le_bytes())?;
        for t in &self.triangles {
            // Not the _best_ way to calculate a normal, but good enough
            let a = self.vertices[t.x];
            let b = self.vertices[t.y];
            let c = self.vertices[t.z];
            let ab = b - a;
            let ac = c - a;
            let normal = ab.cross(&ac);
            for p in &normal {
                out.write_all(&p.to_le_bytes())?;
            }
            for v in t {
                for p in &self.vertices[*v] {
                    out.write_all(&p.to_le_bytes())?;
                }
            }
            out.write_all(&[0u8; std::mem::size_of::<u16>()])?; // attributes
        }
        Ok(())
    }
}
