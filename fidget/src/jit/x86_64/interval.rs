use crate::{
    eval::types::Interval,
    jit::{
        interval::IntervalAssembler, mmap::Mmap, reg, Assembler, AssemblerData,
        CHOICE_BOTH, CHOICE_LEFT, CHOICE_RIGHT, IMM_REG, OFFSET,
        REGISTER_LIMIT,
    },
    Error,
};
use dynasmrt::{dynasm, DynasmApi, DynasmLabelApi};

/// Implementation of the interval assembler on `x86_64`
///
/// Registers are passed in as follows:
///
/// | Variable   | Register | Type                  |
/// |------------|----------|-----------------------|
/// | X          | `xmm0`   | `[f32; 2]`            |
/// | Y          | `xmm1`   | `[f32; 2]`            |
/// | Z          | `xmm2`   | `[f32; 2]`            |
/// | `vars`     | `rdi`    | `*const f32` (array)  |
/// | `choices`  | `rsi`    | `*mut u8` (array)     |
/// | `simplify` | `rdx`    | `*mut u8` (single)    |
///
/// The stack is configured as follows
///
/// ```text
/// | Position | Value        | Notes                                       |
/// |----------|--------------|---------------------------------------------|
/// | 0x00     | `rbp`        | Previous value for base pointer             |
/// |----------|--------------|---------------------------------------------|
/// | -0x08    | `r12`        | During functions calls, we use these        |
/// | -0x10    | `r13`        | as temporary storage so must preserve their |
/// | -0x18    | `r14`        | previous values on the stack                |
/// |----------|--------------|---------------------------------------------|
/// | -0x20    | Z            | Inputs (as 2x floats)                       |
/// | -0x28    | Y            |                                             |
/// | -0x30    | X            |                                             |
/// |----------|--------------|---------------------------------------------|
/// | ...      | ...          | Register spills live up here                |
/// |----------|--------------|---------------------------------------------|
/// | 0x58     | xmm15        | Caller-saved registers during functions     |
/// | 0x50     | xmm14        | calls are placed here, then restored        |
/// | 0x48     | xmm13        |                                             |
/// | 0x40     | xmm12        |                                             |
/// | 0x38     | xmm11        |                                             |
/// | 0x30     | xmm10        |                                             |
/// | 0x28     | xmm9         |                                             |
/// | 0x20     | xmm8         |                                             |
/// | 0x18     | xmm7         |                                             |
/// | 0x10     | xmm6         |                                             |
/// | 0x08     | xmm5         |                                             |
/// | 0x00     | xmm4         |                                             |
/// ```
const STACK_SIZE_UPPER: usize = 0x30; // Positions relative to `rbp`
const STACK_SIZE_LOWER: usize = 0x60; // Positions relative to `rsp`

impl Assembler for IntervalAssembler {
    type Data = Interval;

    fn init(mmap: Mmap, slot_count: usize) -> Self {
        let mut out = AssemblerData::new(mmap);
        dynasm!(out.ops
            ; push rbp
            ; mov rbp, rsp
        );
        out.prepare_stack(slot_count, STACK_SIZE_UPPER + STACK_SIZE_LOWER);
        dynasm!(out.ops
            ; vzeroupper

            // Put X/Y/Z on the stack so we can use those registers
            ; vmovsd [rbp - 0x30], xmm0
            ; vmovsd [rbp - 0x28], xmm1
            ; vmovsd [rbp - 0x20], xmm2
        );
        Self(out)
    }
    fn build_load(&mut self, dst_reg: u8, src_mem: u32) {
        assert!((dst_reg as usize) < REGISTER_LIMIT);
        let sp_offset: i32 = (self.0.stack_pos(src_mem)
            + STACK_SIZE_LOWER as u32)
            .try_into()
            .unwrap();
        dynasm!(self.0.ops
            // Pretend that we're a double
            ; movq Rx(reg(dst_reg)), [rsp + sp_offset]
        );
    }
    fn build_store(&mut self, dst_mem: u32, src_reg: u8) {
        assert!((src_reg as usize) < REGISTER_LIMIT);
        let sp_offset: i32 = (self.0.stack_pos(dst_mem)
            + STACK_SIZE_LOWER as u32)
            .try_into()
            .unwrap();
        dynasm!(self.0.ops
            // Pretend that we're a double
            ; movq [rsp + sp_offset], Rx(reg(src_reg))
        );
    }
    fn build_input(&mut self, out_reg: u8, src_arg: u8) {
        let pos = STACK_SIZE_UPPER as i32
            - (std::mem::size_of::<Self::Data>() as i32) * (src_arg as i32);
        dynasm!(self.0.ops
            ; vmovq Rx(reg(out_reg)), [rbp - pos]
        );
    }
    fn build_var(&mut self, out_reg: u8, src_arg: u32) {
        dynasm!(self.0.ops
            ; vmovss Rx(reg(out_reg)), [rdi + 4 * (src_arg as i32)]
            // Somewhat overkill, since we only need two values, but oh well
            ; vbroadcastss Rx(reg(out_reg)), Rx(reg(out_reg))
        );
    }
    fn build_sin(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "sysv64" fn interval_sin(v: Interval) -> Interval {
            v.sin()
        }
        self.call_fn_unary(out_reg, lhs_reg, interval_sin);
    }
    fn build_cos(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "sysv64" fn float_cos(f: Interval) -> Interval {
            f.cos()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_cos);
    }
    fn build_tan(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "sysv64" fn float_tan(f: Interval) -> Interval {
            f.tan()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_tan);
    }
    fn build_asin(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "sysv64" fn float_asin(f: Interval) -> Interval {
            f.asin()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_asin);
    }
    fn build_acos(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "sysv64" fn float_acos(f: Interval) -> Interval {
            f.acos()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_acos);
    }
    fn build_atan(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "sysv64" fn float_atan(f: Interval) -> Interval {
            f.atan()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_atan);
    }
    fn build_exp(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "sysv64" fn float_exp(f: Interval) -> Interval {
            f.exp()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_exp);
    }
    fn build_ln(&mut self, out_reg: u8, lhs_reg: u8) {
        extern "sysv64" fn float_ln(f: Interval) -> Interval {
            f.ln()
        }
        self.call_fn_unary(out_reg, lhs_reg, float_ln);
    }
    fn build_copy(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; vmovq Rx(reg(out_reg)), Rx(reg(lhs_reg))
        );
    }
    fn build_neg(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; vpshufd Rx(reg(out_reg)), Rx(reg(lhs_reg)), 0b11110001u8 as i8
            ; pcmpeqd xmm0, xmm0 // set xmm0 to all 1s
            ; pslld xmm0, 31     // shift, leaving xmm0 = 0x80000000 x 4
            ; vxorps Rx(reg(out_reg)), Rx(reg(out_reg)), xmm0
        );
    }
    fn build_abs(&mut self, out_reg: u8, lhs_reg: u8) {
        // TODO: use cmpltss instead of 2x comiss?
        dynasm!(self.0.ops
            // Store 0.0 to xmm0, for comparisons
            ; vpxor xmm0, xmm0, xmm0

            // Pull the upper value into xmm1
            ; vpshufd xmm1, Rx(reg(lhs_reg)), 1

            // Check whether lhs.upper < 0
            ; comiss xmm0, xmm1
            ; ja >N // negative

            // Check whether lhs.lower < 0
            ; comiss xmm0, Rx(reg(lhs_reg))
            ; ja >S // straddling 0

            // Fallthrough: the whole interval is above zero, so we just copy it
            // over and return.
            ; vmovq Rx(reg(out_reg)), Rx(reg(lhs_reg))
            ; jmp >E

            // The interval is less than zero, so we need to calculate
            // [-upper, -lower]
            ; N:
            ; vpcmpeqd xmm0, xmm0, xmm0 // set xmm0 to all 1s
            ; vpslld xmm0, xmm0, 31     // shift, leaving xmm0 = 0x80000000
            ; vxorps xmm0, xmm0, Rx(reg(lhs_reg)) // xor to swap sign bits
            ; vpshufd Rx(reg(out_reg)), xmm0, 1 // swap lo and hi
            ; jmp >E

            // The interval straddles 0, so we need to calculate
            // [0.0, max(abs(lower, upper))]
            ; S:
            ; vpcmpeqd xmm0, xmm0, xmm0 // set xmm0 to all 1s
            ; vpsrld xmm0, xmm0, 1      // shift, leaving xmm0 = 0x7fffffff

            // Copy to out_reg and clear sign bits; setting up out_reg as
            // [abs(low), abs(high)]
            ; vandps Rx(reg(out_reg)), Rx(reg(lhs_reg)), xmm0

            // Set up xmm0 to contain [abs(high), abs(low)]
            ; vpshufd xmm0, Rx(reg(out_reg)), 0b11110001u8 as i8

            ; vcomiss xmm0, Rx(reg(out_reg)) // Compare abs(hi) vs abs(lo)
            ; ja >C // if abs(hi) > abs(lo), then we don't need to swap

            ; vpshufd Rx(reg(out_reg)), Rx(reg(out_reg)), 0b11110011u8 as i8

            // Clear the lowest value of the interval, leaving us with [0, ...]
            ; C:
            ; vpshufd Rx(reg(out_reg)), Rx(reg(out_reg)), 0b11110111u8 as i8
            // fallthrough to end

            ; E:
        );
        self.0.ops.commit_local().unwrap();
    }
    fn build_recip(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; vpxor xmm0, xmm0, xmm0 // xmm0 = 0.0
            ; vcomiss Rx(reg(lhs_reg)), xmm0
            ; ja >O // low element is > 0
            ; vpshufd xmm1, Rx(reg(lhs_reg)), 1 // extract high element
            ; vcomiss xmm0, xmm1
            ; ja >O // high element is < 0

            // Bad case: the division spans 0, so return NaN
            ; pcmpeqw Rx(reg(out_reg)), Rx(reg(out_reg))
            ; pslld Rx(reg(out_reg)), 23
            ; psrld Rx(reg(out_reg)), 1
            ; jmp >E

            ; O: // We're okay!
            // Load 1.0 into xmm0
            ; pcmpeqw xmm0, xmm0
            ; pslld xmm0, 25
            ; psrld xmm0, 2
            ; vdivps Rx(reg(out_reg)), xmm0, Rx(reg(lhs_reg))
            ; pshufd Rx(reg(out_reg)), Rx(reg(out_reg)), 0b0001
            // Fallthrough to end

            ; E:
        );
        self.0.ops.commit_local().unwrap();
    }
    fn build_sqrt(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            ; vpxor xmm0, xmm0, xmm0 // xmm0 = 0.0
            ; vcomiss xmm0, Rx(reg(lhs_reg))
            ; ja >L // lower_lz

            // Happy path
            ; vsqrtps Rx(reg(out_reg)), Rx(reg(lhs_reg))
            ; jmp >E

            // lower < 0 => [NaN, NaN]
            ; L:
            ; vpcmpeqw Rx(reg(out_reg)), Rx(reg(out_reg)), Rx(reg(out_reg))
            ; vpslld Rx(reg(out_reg)), Rx(reg(out_reg)), 23
            ; vpsrld Rx(reg(out_reg)), Rx(reg(out_reg)), 1

            ; E:
        );
        self.0.ops.commit_local().unwrap();
    }
    fn build_square(&mut self, out_reg: u8, lhs_reg: u8) {
        dynasm!(self.0.ops
            // Put component-wise multiplication in xmm2
            ; vmulps xmm2, Rx(reg(lhs_reg)), Rx(reg(lhs_reg))
            ; vpxor xmm0, xmm0, xmm0 // xmm0 = 0.0
            ; vpshufd xmm1, Rx(reg(lhs_reg)), 1
            ; vcomiss xmm0, xmm1
            ; ja >N // negative
            ; vcomiss xmm0, Rx(reg(lhs_reg))
            ; ja >S // straddling 0

            // Fallthrough: lower > 0, so our previous result is fine
            ; vmovq Rx(reg(out_reg)), xmm2
            ; jmp >E

            // upper < 0, so we square then swap
            ; N:
            ; vpshufd Rx(reg(out_reg)), xmm2, 0b11110001u8 as i8
            ; jmp >E

            // lower < 0, upper > 0 => pick the bigger result
            ; S:
            ; vpshufd Rx(reg(out_reg)), xmm2, 1
            ; vmaxss Rx(reg(out_reg)), Rx(reg(out_reg)), xmm2
            // Shift the low float to the upper position
            ; vpsllq Rx(reg(out_reg)), Rx(reg(out_reg)), 32

            ; E:
        );
        self.0.ops.commit_local().unwrap();
    }
    fn build_add(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; vaddps Rx(reg(out_reg)), Rx(reg(lhs_reg)), Rx(reg(rhs_reg))
        );
    }
    fn build_sub(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; vpshufd xmm1, Rx(reg(rhs_reg)), 0b11110001u8 as i8
            ; vsubps Rx(reg(out_reg)), Rx(reg(lhs_reg)), xmm1
        );
    }
    fn build_mul(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; vpshufd xmm2, Rx(reg(lhs_reg)), 0b01000001_i8
            ; vpshufd xmm1, Rx(reg(rhs_reg)), 0b00010001_i8
            ; vmulps xmm2, xmm2, xmm1 // xmm2 contains all 4 results

            // Extract the horizontal minimum into out
            ; vpshufd xmm1, xmm2, 0b00001110 // xmm1 = [_, _, 3, 2]
            ; vminps xmm1, xmm1, xmm2 // xmm1 = [_, _, min(3, 1), min(2, 0)]
            ; vpshufd Rx(reg(out_reg)), xmm1, 0b00000001 // out = max(3, 1)
            ; vminss Rx(reg(out_reg)), Rx(reg(out_reg)), xmm1 // out[0] is lowest value

            // Extract the horizontal maximum into xmm2
            ; vpshufd xmm1, xmm2, 0b00001110 // xmm1 = [_, _, 3, 2]
            ; vmaxps xmm1, xmm1, xmm2 // xmm1 = [_, _, max(3, 1), max(2, 0)]
            ; vpshufd xmm2, xmm1, 0b00000001 // xmm2 = max(3, 1)
            ; vmaxss xmm2, xmm2, xmm1 // xmm2[0] is highest value

            // Splice the two together
            ; vunpcklps Rx(reg(out_reg)), Rx(reg(out_reg)), xmm2
        );
    }
    fn build_div(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; vpxor xmm1, xmm1, xmm1 // xmm1 = 0.0
            ; vcomiss Rx(reg(rhs_reg)), xmm1
            ; ja >O // okay
            ; vpshufd xmm2, Rx(reg(rhs_reg)), 1
            ; vcomiss xmm1, xmm2
            ; ja >O // okay

            // Fallthrough: an input is NaN or rhs_reg spans 0; return NaN
            // by manually building it in the XMM register
            ; vpcmpeqw Rx(reg(out_reg)), Rx(reg(out_reg)), Rx(reg(out_reg))
            ; vpslld Rx(reg(out_reg)), Rx(reg(out_reg)), 23
            ; vpsrld Rx(reg(out_reg)), Rx(reg(out_reg)), 1
            ; jmp >E

            // Reorganize
            ; O:
            ; vpshufd xmm2, Rx(reg(lhs_reg)), 0b01000001_i8
            ; vpshufd xmm1, Rx(reg(rhs_reg)), 0b00010001_i8
            ; vdivps xmm2, xmm2, xmm1 // xmm2 contains all 4 results

            // Extract the horizontal minimum into out
            ; vpshufd xmm1, xmm2, 0b00001110 // xmm1 = [_, _, 3, 2]
            ; vminps xmm1, xmm1, xmm2 // xmm1 = [_, _, min(3, 1), min(2, 0)]
            ; vpshufd Rx(reg(out_reg)), xmm1, 0b00000001 // out = max(3, 1)
            ; vminss Rx(reg(out_reg)), Rx(reg(out_reg)), xmm1 // out[0] is lowest value

            // Extract the horizontal maximum into xmm2
            ; vpshufd xmm1, xmm2, 0b00001110 // xmm1 = [_, _, 3, 2]
            ; vmaxps xmm1, xmm1, xmm2 // xmm1 = [_, _, max(3, 1), max(2, 0)]
            ; vpshufd xmm2, xmm1, 0b00000001 // xmm2 = max(3, 1)
            ; vmaxss xmm2, xmm2, xmm1 // xmm2[0] is highest value

            // Splice the two together
            ; unpcklps Rx(reg(out_reg)), xmm2

            ; E:
        );
        self.0.ops.commit_local().unwrap();
    }
    fn build_max(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        dynasm!(self.0.ops
            ; mov ax, [rsi]

            // xmm1 = lhs.upper
            ; vpshufd xmm1, Rx(reg(lhs_reg)), 0b11111101u8 as i8
            ; vcomiss xmm1, Rx(reg(rhs_reg)) // compare lhs.upper and rhs.lower
            ; jp >N // NaN
            ; jb >R // rhs

            // xmm1 = rhs.upper
            ; vpshufd xmm1, Rx(reg(rhs_reg)), 0b11111101u8 as i8
            ; vcomiss xmm1, Rx(reg(lhs_reg))
            ; jp >N
            ; jb >L

            // Fallthrough: ambiguous case
            ; vmaxps Rx(reg(out_reg)), Rx(reg(lhs_reg)), Rx(reg(rhs_reg))
            ; or ax, CHOICE_BOTH as i16
            ; jmp >E

            ; N:
            ; or ax, CHOICE_BOTH as i16
            // Load NaN into out_reg
            ; vpcmpeqw Rx(reg(out_reg)), Rx(reg(out_reg)), Rx(reg(out_reg))
            ; vpslld Rx(reg(out_reg)), Rx(reg(out_reg)), 23
            ; vpsrld Rx(reg(out_reg)), Rx(reg(out_reg)), 1
            ; jmp >E

            // lhs.upper < rhs.lower
            ; L:
            ; vmovq Rx(reg(out_reg)), Rx(reg(lhs_reg))
            ; or ax, CHOICE_LEFT as i16
            ; mov cx, 1 // TODO: why can't we write 1 to [rdx] directly?
            ; mov [rdx], cx
            ; jmp >E

            // rhs.upper < lhs.lower
            ; R:
            ; vmovq Rx(reg(out_reg)), Rx(reg(rhs_reg))
            ; or ax, CHOICE_RIGHT as i16
            ; mov cx, 1
            ; mov [rdx], cx
            // Fallthrough

            ; E:
            ; mov [rsi], ax
            ; add rsi, 1
        );
        self.0.ops.commit_local().unwrap();
    }
    fn build_min(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        // TODO: Godbolt uses unpcklps ?
        dynasm!(self.0.ops
            //  if lhs.upper < rhs.lower
            //      *choices++ |= CHOICE_LEFT
            //      out = lhs
            //  elif rhs.upper < lhs.lower
            //      *choices++ |= CHOICE_RIGHT
            //      out = rhs
            //  else
            //      *choices++ |= CHOICE_BOTH
            //      out = fmin(lhs, rhs)

            ; mov ax, [rsi]

            // TODO: use cmpltss to do both comparisons?

            // xmm1 = lhs.upper
            ; vpshufd xmm1, Rx(reg(lhs_reg)), 0b11111101u8 as i8
            ; vcomiss xmm1, Rx(reg(rhs_reg)) // compare lhs.upper and rhs.lower
            ; jp >N
            ; jb >L

            // xmm1 = rhs.upper
            ; vpshufd xmm1, Rx(reg(rhs_reg)), 0b11111101u8 as i8
            ; vcomiss xmm1, Rx(reg(lhs_reg))
            ; jp >N
            ; jb >R

            // Fallthrough: ambiguous case
            ; vminps Rx(reg(out_reg)), Rx(reg(lhs_reg)), Rx(reg(rhs_reg))
            ; or ax, CHOICE_BOTH as i16
            ; jmp >E

            ; N:
            ; or ax, CHOICE_BOTH as i16
            // Load NAN into out_reg
            ; vpcmpeqw Rx(reg(out_reg)), Rx(reg(out_reg)), Rx(reg(out_reg))
            ; vpslld Rx(reg(out_reg)), Rx(reg(out_reg)), 23
            ; vpsrld Rx(reg(out_reg)), Rx(reg(out_reg)), 1
            ; jmp >E

            // lhs.upper < rhs.lower
            ; L:
            ; vmovq Rx(reg(out_reg)), Rx(reg(lhs_reg))
            ; or ax, CHOICE_LEFT as i16
            ; mov cx, 1 // TODO: why can't we write 1 to [rdx] directly?
            ; mov [rdx], cx
            ; jmp >E

            // rhs.upper < lhs.lower
            ; R:
            ; vmovq Rx(reg(out_reg)), Rx(reg(rhs_reg))
            ; or ax, CHOICE_RIGHT as i16
            ; mov cx, 1
            ; mov [rdx], cx
            // Fallthrough

            ; E:
            ; mov [rsi], ax
            ; add rsi, 1
        );
        self.0.ops.commit_local().unwrap();
    }
    fn build_compare(&mut self, out_reg: u8, lhs_reg: u8, rhs_reg: u8) {
        // TODO: Godbolt uses unpcklps ?
        dynasm!(self.0.ops
            //  if lhs.has_nan || rhs.has_nan
            //      out = [NAN, NAN]
            //  elif lhs.upper < rhs.lower
            //      out = [-1, 1]
            //  elif rhs.upper < lhs.lower
            //      out = [1, 1]
            //  else
            //      out = [-1, 1]

            // TODO: use cmpltss to do both comparisons?
            // xmm1 = lhs.upper
            ; vpshufd xmm1, Rx(reg(lhs_reg)), 0b11111101u8 as i8
            ; vcomiss xmm1, Rx(reg(rhs_reg)) // compare lhs.upper and rhs.lower
            ; jp >N
            ; jb >L

            // xmm1 = rhs.upper
            ; vpshufd xmm1, Rx(reg(rhs_reg)), 0b11111101u8 as i8
            ; vcomiss xmm1, Rx(reg(lhs_reg))
            ; jp >N
            ; jb >R

            // Fallthrough: ambiguous case, so load [-1, 1]
            ; mov eax, (-1f32).to_bits() as i32
            ; vpinsrd Rx(reg(out_reg)), Rx(reg(out_reg)), eax, 0
            ; mov eax, 1f32.to_bits() as i32
            ; vpinsrd Rx(reg(out_reg)), Rx(reg(out_reg)), eax, 1
            ; jmp >E

            ; N:
            // Load NAN into out_reg
            ; vpcmpeqw Rx(reg(out_reg)), Rx(reg(out_reg)), Rx(reg(out_reg))
            ; vpslld Rx(reg(out_reg)), Rx(reg(out_reg)), 23
            ; vpsrld Rx(reg(out_reg)), Rx(reg(out_reg)), 1
            ; jmp >E

            // lhs.upper < rhs.lower
            ; L:
            ; mov eax, (-1f32).to_bits() as i32
            ; vmovd xmm1, eax
            ; vbroadcastss Rx(reg(out_reg)), xmm1
            ; jmp >E

            // rhs.upper < lhs.lower
            ; R:
            ; mov eax, 1f32.to_bits() as i32
            ; vmovd xmm1, eax
            ; vbroadcastss Rx(reg(out_reg)), xmm1
            // Fallthrough

            ; E:
        );
        self.0.ops.commit_local().unwrap();
    }
    fn load_imm(&mut self, imm: f32) -> u8 {
        let imm_u32 = imm.to_bits();
        dynasm!(self.0.ops
            ; mov eax, imm_u32 as i32
            ; vmovd Rx(IMM_REG), eax
            ; vbroadcastss Rx(IMM_REG), Rx(IMM_REG)
        );
        IMM_REG.wrapping_sub(OFFSET)
    }
    fn finalize(mut self, out_reg: u8) -> Result<Mmap, Error> {
        if self.0.saved_callee_regs {
            dynasm!(self.0.ops
                ; mov r12, [rbp - 0x8]
                ; mov r13, [rbp - 0x10]
                ; mov r14, [rbp - 0x18]
            );
        }
        dynasm!(self.0.ops
            ; vmovq xmm0, Rx(reg(out_reg))
            ; add rsp, self.0.mem_offset as i32
            ; pop rbp
            ; emms
            ; ret
        );
        self.0.ops.finalize()
    }
}

impl IntervalAssembler {
    fn call_fn_unary(
        &mut self,
        out_reg: u8,
        arg_reg: u8,
        f: extern "sysv64" fn(Interval) -> Interval,
    ) {
        // Back up a few callee-saved registers that we're about to use
        if !self.0.saved_callee_regs {
            dynasm!(self.0.ops
                ; mov [rbp - 0x8], r12
                ; mov [rbp - 0x10], r13
                ; mov [rbp - 0x18], r14
            );
            self.0.saved_callee_regs = true
        }
        let addr = f as usize;
        dynasm!(self.0.ops
            // Back up vars/choice/simplify pointers to registers
            ; mov r12, rdi
            ; mov r13, rsi
            ; mov r14, rdx

            // Back up register values to the stack, treating them as doubles
            // (since we want to back up all 64 bits)
            //
            // TODO should these be `movq` instead?
            ; vmovsd [rsp], xmm4
            ; vmovsd [rsp + 0x08], xmm5
            ; vmovsd [rsp + 0x10], xmm6
            ; vmovsd [rsp + 0x18], xmm7
            ; vmovsd [rsp + 0x20], xmm8
            ; vmovsd [rsp + 0x28], xmm9
            ; vmovsd [rsp + 0x30], xmm10
            ; vmovsd [rsp + 0x38], xmm11
            ; vmovsd [rsp + 0x40], xmm12
            ; vmovsd [rsp + 0x48], xmm13
            ; vmovsd [rsp + 0x50], xmm14
            ; vmovsd [rsp + 0x58], xmm15

            // copy arg to xmm0
            ; vmovq xmm0, Rx(reg(arg_reg))
            ; mov rdx, QWORD addr as _
            ; call rdx

            // Restore float registers
            ; vmovsd xmm4, [rsp]
            ; vmovsd xmm5, [rsp + 0x08]
            ; vmovsd xmm6, [rsp + 0x10]
            ; vmovsd xmm7, [rsp + 0x18]
            ; vmovsd xmm8, [rsp + 0x20]
            ; vmovsd xmm9, [rsp + 0x28]
            ; vmovsd xmm10, [rsp + 0x30]
            ; vmovsd xmm11, [rsp + 0x38]
            ; vmovsd xmm12, [rsp + 0x40]
            ; vmovsd xmm13, [rsp + 0x48]
            ; vmovsd xmm14, [rsp + 0x50]
            ; vmovsd xmm15, [rsp + 0x58]

            // Restore vars/choice/simplify pointers
            ; mov rdi, r12
            ; mov rsi, r13
            ; mov rdx, r14

            // Unpack the interval result
            ; vmovq Rx(reg(out_reg)), xmm0
        );
    }
}
