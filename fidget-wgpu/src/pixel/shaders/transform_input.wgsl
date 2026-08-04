// Transform inputs with `config.mat`
fn transformed_inputs(ix: Value, iy: Value) -> array<Value, 2> {
    var ts = array(Value(), Value(), Value());
    for (var i = 0; i < 3; i++) {
        ts[i] = op_add(
            op_add(
                op_mul(build_imm(config.mat[0][i]), ix),
                op_mul(build_imm(config.mat[1][i]), iy),
            ),
            build_imm(config.mat[2][i]),
        );
    }

    // Apply homogeneous transform
    return array(
        op_div(ts[0], ts[2]),
        op_div(ts[1], ts[2]),
    );
}
