// Check if any value is true across a workgroup
//
// In the wasm case, we must pass uniformity analysis
fn wg_any(cond: bool) -> bool {
    atomicOr(&wg_any_scratch, u32(cond));
    workgroupBarrier();
    let out = workgroupUniformLoad(&wg_any_scratch) != 0;
    workgroupBarrier();
    atomicStore(&wg_any_scratch, 0);
    return out;
}

var<workgroup> wg_any_scratch: atomic<u32>;
