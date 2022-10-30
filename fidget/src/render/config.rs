use nalgebra::{
    allocator::Allocator, geometry::Transform, Const, DefaultAllocator,
    DimNameSum, U1,
};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct RenderConfig<const N: usize>
where
    nalgebra::Const<N>: nalgebra::DimNameAdd<nalgebra::U1>,
    DefaultAllocator:
        Allocator<f32, DimNameSum<Const<N>, U1>, DimNameSum<Const<N>, U1>>,
{
    pub image_size: usize,
    pub tile_sizes: Vec<usize>,
    pub threads: usize,

    pub mat: Transform<f32, nalgebra::TGeneral, N>,
}

impl<const N: usize> Default for RenderConfig<N>
where
    nalgebra::Const<N>: nalgebra::DimNameAdd<nalgebra::U1>,
    DefaultAllocator:
        Allocator<f32, DimNameSum<Const<N>, U1>, DimNameSum<Const<N>, U1>>,
{
    fn default() -> Self {
        Self {
            image_size: 512,
            tile_sizes: match N {
                2 => vec![128, 32, 8],
                _ => vec![128, 64, 32, 16, 8],
            },
            threads: 8,
            mat: Transform::identity(),
        }
    }
}

impl<const N: usize> RenderConfig<N>
where
    nalgebra::Const<N>: nalgebra::DimNameAdd<nalgebra::U1>,
    DefaultAllocator:
        Allocator<f32, DimNameSum<Const<N>, U1>, DimNameSum<Const<N>, U1>>,
{
    #[inline]
    pub fn tile_to_offset(&self, tile: Tile<N>, x: usize, y: usize) -> usize {
        tile.offset + x + y * self.tile_sizes[0]
    }

    #[inline]
    pub fn new_tile(&self, corner: [usize; N]) -> Tile<N> {
        let x = corner[0] % self.tile_sizes[0];
        let y = corner[1] % self.tile_sizes[0];
        Tile {
            corner,
            offset: x + y * self.tile_sizes[0],
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Tile<const N: usize> {
    pub corner: [usize; N],
    offset: usize,
}

/// Worker queue
pub struct Queue<const N: usize> {
    index: AtomicUsize,
    tiles: Vec<Tile<N>>,
}

impl<const N: usize> Queue<N> {
    pub fn new(tiles: Vec<Tile<N>>) -> Self {
        Self {
            index: AtomicUsize::new(0),
            tiles,
        }
    }
    pub fn next(&self) -> Option<Tile<N>> {
        let index = self.index.fetch_add(1, Ordering::Relaxed);
        self.tiles.get(index).cloned()
    }
}

////////////////////////////////////////////////////////////////////////////////
