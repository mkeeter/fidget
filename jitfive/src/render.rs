use std::time::{Duration, Instant};

use crate::backend::dynasm::{IntervalEval, VecEval};

use crate::backend::{
    dynasm::{
        build_interval_fn_48 as build_interval_fn,
        build_vec_fn_48 as build_vec_fn,
    },
    tape48::Tape,
};

use log::info;

const TILE_SIZE: usize = 128;
const SUBTILE_SIZE: usize = 32;

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

struct Renderer {
    size: usize,
    image: Vec<Option<Pixel>>,

    interval_time: Duration,
    push_time: Duration,
    jit_time: Duration,
    pixel_time: Duration,
}

/// Renders in three passes:
/// - 64x64 intervals
/// - 8x8 intervals
/// - 8x8 pixels
pub fn render(size: usize, tape: Tape) -> Vec<Pixel> {
    assert_eq!(size % TILE_SIZE, 0);
    let mut r = Renderer {
        size,
        image: vec![None; size * size],
        interval_time: Duration::ZERO,
        pixel_time: Duration::ZERO,
        push_time: Duration::ZERO,
        jit_time: Duration::ZERO,
    };
    r.run(&tape);
    info!("Interval time: {:?}", r.interval_time);
    info!("Pixel time:    {:?}", r.pixel_time);
    info!("Push time:     {:?}", r.push_time);
    info!("JIT time:      {:?}", r.jit_time);
    r.image.into_iter().map(Option::unwrap).collect()
}

impl Renderer {
    fn run(&mut self, tape: &Tape) {
        let i_handle = build_interval_fn(tape);
        let mut i_eval = i_handle.get_evaluator();

        for y in 0..(self.size / TILE_SIZE) {
            for x in 0..(self.size / TILE_SIZE) {
                self.render_tile(&mut i_eval, x * TILE_SIZE, y * TILE_SIZE);
            }
        }
        // Flip the image vertically
        for y in 0..(self.size / 2) {
            for x in 0..self.size {
                self.image.swap(
                    x + y * self.size,
                    x + (self.size - y - 1) * self.size,
                );
            }
        }
    }

    fn pixel_to_pos(&self, p: usize) -> f32 {
        2.0 * (p as f32) / ((self.size - 1) as f32) - 1.0
    }

    fn render_tile(
        &mut self,
        eval: &mut IntervalEval<Tape>,
        x: usize,
        y: usize,
    ) {
        let x_interval =
            [self.pixel_to_pos(x), self.pixel_to_pos(x + TILE_SIZE)];
        let y_interval =
            [self.pixel_to_pos(y), self.pixel_to_pos(y + TILE_SIZE)];

        let start = Instant::now();
        let i = eval.i_subdiv(x_interval, y_interval, [0.0, 0.0], 4);
        self.interval_time += start.elapsed();

        if i[1] < 0.0 {
            for y in y..(y + TILE_SIZE) {
                for x in x..(x + TILE_SIZE) {
                    self.image[x + y * self.size] = Some(Pixel::FilledTile);
                }
            }
        } else if i[0] > 0.0 {
            for y in y..(y + TILE_SIZE) {
                for x in x..(x + TILE_SIZE) {
                    self.image[x + y * self.size] = Some(Pixel::EmptyTile);
                }
            }
        } else {
            let start = Instant::now();
            let sub_tape = eval.push();
            self.push_time += start.elapsed();

            let start = Instant::now();
            let sub_jit = build_interval_fn(&sub_tape);
            let mut sub_eval = sub_jit.get_evaluator();
            self.jit_time += start.elapsed();

            let n = TILE_SIZE / SUBTILE_SIZE;
            for j in 0..n {
                for i in 0..n {
                    self.render_subtile(
                        &mut sub_eval,
                        x + i * SUBTILE_SIZE,
                        y + j * SUBTILE_SIZE,
                    );
                }
            }
        }
    }

    fn render_subtile(
        &mut self,
        eval: &mut IntervalEval<Tape>,
        x: usize,
        y: usize,
    ) {
        let x_interval =
            [self.pixel_to_pos(x), self.pixel_to_pos(x + SUBTILE_SIZE)];
        let y_interval =
            [self.pixel_to_pos(y), self.pixel_to_pos(y + SUBTILE_SIZE)];
        let start = Instant::now();
        let i = eval.i_subdiv(x_interval, y_interval, [0.0, 0.0], 4);
        self.interval_time += start.elapsed();

        if i[1] < 0.0 {
            for y in y..(y + SUBTILE_SIZE) {
                for x in x..(x + SUBTILE_SIZE) {
                    self.image[x + y * self.size] = Some(Pixel::FilledSubtile);
                }
            }
        } else if i[0] > 0.0 {
            for y in y..(y + SUBTILE_SIZE) {
                for x in x..(x + SUBTILE_SIZE) {
                    self.image[x + y * self.size] = Some(Pixel::EmptySubtile);
                }
            }
        } else {
            let start = Instant::now();
            let sub_tape = eval.push();
            self.push_time += start.elapsed();

            let start = Instant::now();
            let sub_jit = build_vec_fn(&sub_tape);
            let mut sub_eval = sub_jit.get_evaluator();
            self.jit_time += start.elapsed();

            let start = Instant::now();
            for y in y..(y + SUBTILE_SIZE) {
                for dx in 0..(SUBTILE_SIZE / 4) {
                    self.render_pixels(&mut sub_eval, x + dx * 4, y);
                }
            }
            self.pixel_time += start.elapsed();
        }
    }

    fn render_pixels(&mut self, eval: &mut VecEval, x: usize, y: usize) {
        let mut x_vec = [0.0; 4];
        for (i, o) in x_vec.iter_mut().enumerate() {
            *o = self.pixel_to_pos(x + i);
        }
        let y_vec = [self.pixel_to_pos(y); 4];

        let v = eval.v(x_vec, y_vec, [0.0; 4]);
        for (i, v) in v.iter().enumerate() {
            if *v < 0.0 {
                self.image[x + i + y * self.size] = Some(Pixel::Filled);
            } else {
                self.image[x + i + y * self.size] = Some(Pixel::Empty);
            }
        }
    }
}
