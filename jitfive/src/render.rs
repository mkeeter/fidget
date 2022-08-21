use crate::eval::Eval;

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
    choices_tile: Vec<u32>,
    choices_subtile: Vec<u32>,
    image: Vec<Option<Pixel>>,
}

/// Renders in three passes:
/// - 64x64 intervals
/// - 8x8 intervals
/// - 8x8 pixels
pub fn render<E: Eval>(size: usize, eval: &E) -> Vec<Pixel> {
    assert_eq!(size % 64, 0);
    let mut r = Renderer {
        eval,
        size,
        choices_root: vec![u32::MAX; eval.choice_array_size()],
        choices_tile: vec![u32::MAX; eval.choice_array_size()],
        choices_subtile: vec![u32::MAX; eval.choice_array_size()],
        image: vec![None; size * size],
    };
    r.run();
    r.image.into_iter().map(Option::unwrap).collect()
}

impl<'a, E: Eval> Renderer<'a, E> {
    fn run(&mut self) {
        for y in 0..(self.size / 64) {
            for x in 0..(self.size / 64) {
                self.render_tile(x * 64, y * 64);
            }
        }
    }

    fn pixel_to_pos(&self, p: usize) -> f32 {
        2.0 * (p as f32) / (self.size as f32) - 1.0
    }

    fn render_tile(&mut self, x: usize, y: usize) {
        let x_interval = [self.pixel_to_pos(x), self.pixel_to_pos(x + 64)];
        let y_interval = [self.pixel_to_pos(y), self.pixel_to_pos(y + 64)];
        let i = self.eval.interval(
            x_interval,
            y_interval,
            &self.choices_root,
            &mut self.choices_tile,
        );
        if i[1] < 0.0 {
            for x in x..(x + 64) {
                for y in y..(y + 64) {
                    self.image[x + y * self.size] = Some(Pixel::FilledTile);
                }
            }
        } else if i[0] > 0.0 {
            for x in x..(x + 64) {
                for y in y..(y + 64) {
                    self.image[x + y * self.size] = Some(Pixel::EmptyTile);
                }
            }
        } else {
            for j in 0..8 {
                for i in 0..8 {
                    self.render_subtile(x + i * 8, y + j * 8);
                }
            }
        }
    }

    fn render_subtile(&mut self, x: usize, y: usize) {
        let x_interval = [self.pixel_to_pos(x), self.pixel_to_pos(x + 8)];
        let y_interval = [self.pixel_to_pos(y), self.pixel_to_pos(y + 8)];
        let i = self.eval.interval(
            x_interval,
            y_interval,
            &self.choices_tile,
            &mut self.choices_subtile,
        );

        if i[1] < 0.0 {
            for x in x..(x + 8) {
                for y in y..(y + 8) {
                    self.image[x + y * self.size] = Some(Pixel::FilledSubtile);
                }
            }
        } else if i[0] > 0.0 {
            for x in x..(x + 8) {
                for y in y..(y + 8) {
                    self.image[x + y * self.size] = Some(Pixel::EmptySubtile);
                }
            }
        } else {
            for x in x..(x + 8) {
                for y in y..(y + 8) {
                    self.render_pixel(x, y);
                }
            }
        }
    }
    fn render_pixel(&mut self, x: usize, y: usize) {
        let x_pos = self.pixel_to_pos(x);
        let y_pos = self.pixel_to_pos(y);

        let i = self.eval.float(x_pos, y_pos, &self.choices_subtile);
        if i < 0.0 {
            self.image[x + y * self.size] = Some(Pixel::Filled);
        } else {
            self.image[x + y * self.size] = Some(Pixel::Empty);
        }
    }
}
