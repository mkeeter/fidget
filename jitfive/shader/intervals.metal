kernel void main0({} vars [[buffer(0)]],
                  {} choices [[buffer(1)]],
                  device {}* result [[buffer(2)]],
                  uint index [[thread_position_in_grid]])
{
    result[index] = t_eval(&vars[index * VAR_COUNT],
                           &choices[index * CHOICE_COUNT]);
}
