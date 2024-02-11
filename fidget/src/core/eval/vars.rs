use std::{collections::HashMap, sync::Arc};

/// `Vars` contains the mapping of variable names to indexes, and a `Vec<f32>`
/// which is suitably sized for use in evaluation.
pub struct Vars {
    names: Arc<HashMap<String, u32>>,
    values: Vec<f32>,
}

impl Vars {
    /// Builds a new variable binding, initializing them all to 0
    pub fn new(names: Arc<HashMap<String, u32>>) -> Self {
        let values = vec![0.0; names.len()];
        Self { names, values }
    }

    /// Binds variables by name to the given values
    ///
    /// The incoming iterator is allowed to include variable names that are not
    /// present in this structure; they will be silently discarded.
    ///
    /// Unbound variables are assigned to `std::f32::NAN`.
    pub fn bind<'a, I: Iterator<Item = (&'a str, f32)>>(
        &mut self,
        iter: I,
    ) -> &[f32] {
        self.values.fill(std::f32::NAN);
        for i in iter {
            if let Some(v) = self.names.get(i.0) {
                self.values[*v as usize] = i.1;
            }
        }
        self.values.as_slice()
    }

    /// Sets a variable by name
    ///
    /// If the name isn't present, this returns silently
    pub fn set(&mut self, name: &str, value: f32) {
        if let Some(v) = self.names.get(name) {
            self.values[*v as usize] = value;
        }
    }

    /// Returns the inner data slice
    pub fn as_slice(&self) -> &[f32] {
        self.values.as_slice()
    }
}
