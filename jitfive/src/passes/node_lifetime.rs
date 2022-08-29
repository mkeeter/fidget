use crate::{compiler::Compiler, scheduled::Scheduled};

pub(crate) fn run(out: &mut Compiler) {
    out.last_use = Scheduled::new_from_compiler(out).last_use;
}
