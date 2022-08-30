use crate::backend::tape32::Tape;
use dynasmrt::{dynasm, DynasmApi, DynasmLabelApi};

pub fn from_tape(t: &Tape) {
    let mut ops = dynasmrt::aarch64::Assembler::new().unwrap();
    dynasm!(ops
    ; -> square:
    );
    let square = ops.offset();
    dynasm!(ops
    ; sub     sp, sp, #16                     // =16
    ; str     s0, [sp, #12]
    ; ldr     s0, [sp, #12]
    ; ldr     s1, [sp, #12]
    ; fmul    s0, s0, s1
    ; add     sp, sp, #16                     // =16
    ; ret
    );

    let buf = ops.finalize().unwrap();
    let hello_fn: extern "win64" fn(f32) -> f32 =
        unsafe { std::mem::transmute(buf.ptr(square)) };
    println!("hi, {}", hello_fn(1.3));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{backend::tape32::Tape, context::Context, scheduled::schedule};

    // This can't be in a doctest, because it uses a pub(crate) function
    #[test]
    fn test_dynasm() {
        let mut ctx = Context::new();
        let x = ctx.x();

        let scheduled = schedule(&ctx, x);
        let tape = Tape::new(&scheduled);
        let jit = from_tape(&tape);
    }
}
