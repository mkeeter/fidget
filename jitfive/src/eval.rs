pub const EVAL_ARRAY_SIZE: usize = 8;

pub trait Eval {
    /// Performs single-point evaluation, reading from `choices_in`
    fn float(&self, x: f32, y: f32, choices_in: &[u32]) -> f32 {
        let x = [x; EVAL_ARRAY_SIZE];
        let y = [y; EVAL_ARRAY_SIZE];
        self.array(x, y, choices_in)[0]
    }

    /// Array evaluation
    fn array(
        &self,
        x: [f32; EVAL_ARRAY_SIZE],
        y: [f32; EVAL_ARRAY_SIZE],
        choices_in: &[u32],
    ) -> [f32; EVAL_ARRAY_SIZE];

    /// Performs interval evaluation, reading from `choices_in` and writing to
    /// `choices_out`
    fn interval(
        &self,
        x: [f32; 2],
        y: [f32; 2],
        choices_in: &[u32],
        choices_out: &mut [u32],
    ) -> [f32; 2];

    /// Returns the number of `u32` in the choice array.
    ///
    /// Note that this is different from [`Compiler::num_choices`], since 16
    /// choices are packed into a single `u32`.
    fn choice_array_size(&self) -> usize;
}
