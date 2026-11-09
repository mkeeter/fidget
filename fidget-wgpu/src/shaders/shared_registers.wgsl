// Tape registers stored in shared memory, interleaved by invocation
var<workgroup> registers: array<Value, REG_COUNT * WORKGROUP_SIZE>;

struct Registers {
    local_index: u32,
}

fn new_registers(local_index: u32) -> Registers {
    return Registers(local_index);
}

fn get_reg(r: ptr<function, Registers>, i: u32) -> Value {
    return registers[i * WORKGROUP_SIZE + r.local_index];
}

fn set_reg(r: ptr<function, Registers>, i: u32, v: Value) {
    registers[i * WORKGROUP_SIZE + r.local_index] = v;
}
