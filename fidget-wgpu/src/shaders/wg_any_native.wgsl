// Check if any value is true across a workgroup
//
// In the native case, we are limited by gfx-rs/wgpu#8785, but do not need to
// pass uniformity analysis.
fn wg_any(cond: bool) -> bool {
    atomicOr(&wg_any_scratch, u32(cond));
    workgroupBarrier();
    let out = atomicLoad(&wg_any_scratch) != 0;
    workgroupBarrier();
    atomicStore(&wg_any_scratch, 0);
    return out;
}

var<workgroup> wg_any_scratch: atomic<u32>;
