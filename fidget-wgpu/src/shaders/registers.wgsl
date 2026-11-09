// Tape registers stored in private memory
struct Registers {
    data: array<Value, REG_COUNT>,
}

fn new_registers(local_index: u32) -> Registers {
    return Registers();
}

fn get_reg(r: ptr<function, Registers>, i: u32) -> Value {
    return r.data[i];
}

fn set_reg(r: ptr<function, Registers>, i: u32, v: Value) {
    r.data[i] = v;
}
