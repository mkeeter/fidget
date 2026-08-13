// Transform inputs with `config.mat`
fn transformed_inputs(ix: Value, iy: Value, iz: Value) -> array<Value, 3> {
    var ts = array(Value(), Value(), Value(), Value());
    for (var i = 0; i < 4; i++) {
        ts[i] = op_add(
            op_add(
                op_mul(build_imm(config.mat[0][i]), ix),
                op_mul(build_imm(config.mat[1][i]), iy),
            ),
            op_add(
                op_mul(build_imm(config.mat[2][i]), iz),
                build_imm(config.mat[3][i]),
            ),
        );
    }

    // Apply homogeneous transform
    return array(
        op_div(ts[0], ts[3]),
        op_div(ts[1], ts[3]),
        op_div(ts[2], ts[3]),
    );
}
