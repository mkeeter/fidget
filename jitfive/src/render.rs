use std::time::{Duration, Instant};

use crate::eval::{Eval, EVAL_ARRAY_SIZE};
use log::info;

const TILE_SIZE: usize = 64;
const SUBTILE_SIZE: usize = 8;

#[derive(Copy, Clone, Debug)]
pub enum Pixel {
    EmptyTile,
    FilledTile,
    EmptySubtile,
    FilledSubtile,
    Empty,
    Filled,
}

impl Pixel {
    pub fn as_color(&self) -> [u8; 4] {
        match self {
            Pixel::EmptyTile => [50, 0, 0, 255],
            Pixel::FilledTile => [255, 0, 0, 255],
            Pixel::EmptySubtile => [0, 50, 0, 255],
            Pixel::FilledSubtile => [0, 255, 0, 255],
            Pixel::Empty => [0, 0, 0, 255],
            Pixel::Filled => [255, 255, 255, 255],
        }
    }
}

struct Renderer<'a, E> {
    eval: &'a E,
    size: usize,
    choices_root: Vec<u32>,
    choices_scratch: Vec<u32>,
    choices_tile: Vec<u32>,
    choices_subtile: Vec<u32>,
    image: Vec<Option<Pixel>>,

    interval_time: Duration,
    pixel_time: Duration,
}

/// Renders in three passes:
/// - 64x64 intervals
/// - 8x8 intervals
/// - 8x8 pixels
pub fn render<E: Eval>(size: usize, eval: &E) -> Vec<Pixel> {
    assert_eq!(size % TILE_SIZE, 0);
    let mut r = Renderer {
        eval,
        size,
        choices_root: vec![u32::MAX; eval.choice_array_size()],
        choices_scratch: vec![u32::MAX; eval.choice_array_size()],
        choices_tile: vec![u32::MAX; eval.choice_array_size()],
        choices_subtile: vec![u32::MAX; eval.choice_array_size()],
        image: vec![None; size * size],
        interval_time: Duration::ZERO,
        pixel_time: Duration::ZERO,
    };
    r.run();
    info!("Interval time: {:?}", r.interval_time);
    info!("Pixel time:    {:?}", r.pixel_time);
    r.image.into_iter().map(Option::unwrap).collect()
}

impl<'a, E: Eval> Renderer<'a, E> {
    fn run(&mut self) {
        for y in 0..(self.size / TILE_SIZE) {
            for x in 0..(self.size / TILE_SIZE) {
                self.render_tile(x * TILE_SIZE, y * TILE_SIZE);
            }
        }
        // Flip the image vertically
        for y in 0..(self.size / 2) {
            for x in 0..self.size {
                let a = self.image[x + y * self.size];
                let b = self.image[x + (self.size - y - 1) * self.size];
                self.image[x + y * self.size] = b;
                self.image[x + y * self.size] = b;
                self.image[x + (self.size - y - 1) * self.size] = a;
            }
        }
    }

    fn pixel_to_pos(&self, p: usize) -> f32 {
        2.0 * (p as f32) / (self.size as f32) - 1.0
    }

    fn render_tile(&mut self, x: usize, y: usize) {
        let x_interval =
            [self.pixel_to_pos(x), self.pixel_to_pos(x + TILE_SIZE)];
        let y_interval =
            [self.pixel_to_pos(y), self.pixel_to_pos(y + TILE_SIZE)];
        let start = Instant::now();
        let i = self.eval.interval(
            x_interval,
            y_interval,
            &self.choices_root,
            &mut self.choices_scratch,
        );
        self.interval_time += start.elapsed();
        if i[1] < 0.0 {
            for x in x..(x + TILE_SIZE) {
                for y in y..(y + TILE_SIZE) {
                    self.image[x + y * self.size] = Some(Pixel::FilledTile);
                }
            }
        } else if i[0] > 0.0 {
            for x in x..(x + TILE_SIZE) {
                for y in y..(y + TILE_SIZE) {
                    self.image[x + y * self.size] = Some(Pixel::EmptyTile);
                }
            }
        } else {
            self.eval
                .push(&self.choices_scratch, &mut self.choices_tile);
            let n = TILE_SIZE / SUBTILE_SIZE;
            for j in 0..n {
                for i in 0..n {
                    self.render_subtile(
                        x + i * SUBTILE_SIZE,
                        y + j * SUBTILE_SIZE,
                    );
                }
            }
        }
    }

    fn render_subtile(&mut self, x: usize, y: usize) {
        let x_interval =
            [self.pixel_to_pos(x), self.pixel_to_pos(x + SUBTILE_SIZE)];
        let y_interval =
            [self.pixel_to_pos(y), self.pixel_to_pos(y + SUBTILE_SIZE)];
        let start = Instant::now();
        let i = self.eval.interval(
            x_interval,
            y_interval,
            &self.choices_tile,
            &mut self.choices_scratch,
        );
        self.interval_time += start.elapsed();

        if i[1] < 0.0 {
            for x in x..(x + SUBTILE_SIZE) {
                for y in y..(y + SUBTILE_SIZE) {
                    self.image[x + y * self.size] = Some(Pixel::FilledSubtile);
                }
            }
        } else if i[0] > 0.0 {
            for x in x..(x + SUBTILE_SIZE) {
                for y in y..(y + SUBTILE_SIZE) {
                    self.image[x + y * self.size] = Some(Pixel::EmptySubtile);
                }
            }
        } else {
            self.eval
                .push(&self.choices_scratch, &mut self.choices_subtile);
            for x in x..(x + SUBTILE_SIZE) {
                assert!(SUBTILE_SIZE >= EVAL_ARRAY_SIZE);
                assert!(SUBTILE_SIZE % EVAL_ARRAY_SIZE == 0);
                for i in 0..(SUBTILE_SIZE / EVAL_ARRAY_SIZE) {
                    self.render_pixel(x, y + i * EVAL_ARRAY_SIZE);
                }
            }
        }
    }
    fn render_pixel(&mut self, x: usize, y: usize) {
        let x_pos = self.pixel_to_pos(x);
        let x_array = [x_pos; EVAL_ARRAY_SIZE];
        let mut y_array = [0.0; EVAL_ARRAY_SIZE];
        for (i, v) in y_array.iter_mut().enumerate() {
            *v = self.pixel_to_pos(y + i);
        }

        let start = Instant::now();
        let v = self.eval.array(x_array, y_array, &self.choices_subtile);
        self.pixel_time += start.elapsed();
        for (i, v) in v.iter().enumerate() {
            if *v < 0.0 {
                self.image[x + (y + i) * self.size] = Some(Pixel::Filled);
            } else {
                self.image[x + (y + i) * self.size] = Some(Pixel::Empty);
            }
        }
    }
}
