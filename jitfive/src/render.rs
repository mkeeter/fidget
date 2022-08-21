use crate::eval::Eval;

struct Renderer<'a, E> {
    eval: &'a E,
    size: usize,
    choices_root: Vec<u32>,
    choices_tile: Vec<u32>,
    choices_subtile: Vec<u32>,
    image: Vec<u8>,
}

/// Renders in three passes:
/// - 64x64 intervals
/// - 8x8 intervals
/// - 8x8 pixels
pub fn render<E: Eval>(size: usize, eval: &E) -> Vec<u8> {
    assert_eq!(size % 64, 0);
    let mut r = Renderer {
        eval,
        size,
        choices_root: vec![u32::MAX; eval.choice_array_size()],
        choices_tile: vec![u32::MAX; eval.choice_array_size()],
        choices_subtile: vec![u32::MAX; eval.choice_array_size()],
        image: vec![0u8; size * size],
    };
    r.run();
    r.image
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
            // If the region is completely inside, then color it in and return
            for j in 0..64 {
                for i in 0..64 {
                    self.image[x + i + (y + j) * self.size] = 3;
                }
            }
        } else if i[0] < 0.0 {
            // Ambiguous case: recurse down to subtiles
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
            // If the region is completely inside, then color it in and return
            for j in 0..8 {
                for i in 0..8 {
                    self.image[x + i + (y + j) * self.size] = 2;
                }
            }
        } else if i[0] < 0.0 {
            // Ambiguous case: render individual pixels
            for j in 0..8 {
                for i in 0..8 {
                    self.render_pixel(x + i, y + j);
                }
            }
        }
    }
    fn render_pixel(&mut self, x: usize, y: usize) {
        let x_pos = self.pixel_to_pos(x);
        let y_pos = self.pixel_to_pos(y);

        let i = self.eval.float(x_pos, y_pos, &self.choices_subtile);
        if i < 0.0 {
            self.image[x + y * self.size] = 1;
        }
    }
}
