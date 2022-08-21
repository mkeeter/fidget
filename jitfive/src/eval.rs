pub trait Eval {
    fn float(&self, x: f32, y: f32, choices_in: &[u32]) -> f32;
    fn interval(
        &self,
        x: [f32; 2],
        y: [f32; 2],
        choices_in: &[u32],
        choices_out: &mut [u32],
    ) -> [f32; 2];
}
