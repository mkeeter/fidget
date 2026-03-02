// 16^3 sub-tile interval evaluation + simplification
//
// Each workgroup corresponds to one root (64^3) tile.
// Each thread handles one 16^3 sub-tile within the root tile.
// Reads root-simplified tapes and produces further-simplified 16^3 tapes.
//
// Must be combined with: opcode constants, REG_COUNT,
// common.wgsl, stack.wgsl, tape_interpreter.wgsl, interval_ops.wgsl,
// tape_simplify.wgsl.

@group(1) @binding(0) var<storage, read_write> tape_data: TapeData;
@group(1) @binding(1) var<storage, read> root_status: array<u32>;
@group(1) @binding(2) var<storage, read_write> tile16_status: array<u32>;

const TILE64_SIZE: u32 = 64u;
const TILE16_SIZE: u32 = 16u;

const STATUS_EMPTY: u32 = 0u;
const STATUS_FILLED: u32 = 1u;
const STATUS_AMBIGUOUS: u32 = 2u;

@compute @workgroup_size(4, 4, 4)
fn interval_tile16_main(
    @builtin(workgroup_id) root_tile: vec3u,
    @builtin(local_invocation_id) sub_tile: vec3u,
) {
    let size_tiles64 = config.render_size / TILE64_SIZE;
    let size_tiles16 = config.render_size / TILE16_SIZE;

    if root_tile.x >= size_tiles64.x ||
       root_tile.y >= size_tiles64.y ||
       root_tile.z >= size_tiles64.z {
        return;
    }

    let root_idx = root_tile.x +
        root_tile.y * size_tiles64.x +
        root_tile.z * size_tiles64.x * size_tiles64.y;

    let root_word = root_status[root_idx];
    let root_stat = root_word >> 20u;
    let root_tape = root_word & 0xFFFFFu;

    // Only evaluate sub-tiles of ambiguous root tiles
    if root_stat != STATUS_AMBIGUOUS {
        return;
    }

    // 16^3 sub-tile position
    let tile16_corner = root_tile * 4u + sub_tile;
    let tile16_idx = tile16_corner.x +
        tile16_corner.y * size_tiles16.x +
        tile16_corner.z * size_tiles16.x * size_tiles16.y;

    // Interval evaluation at 16^3 using root-simplified tape
    let m = interval_inputs(tile16_corner, TILE16_SIZE);
    var stack = Stack();
    let out = run_tape(root_tape, m, &stack);
    let v = out.value.v;

    if v[0] > 0.0 {
        tile16_status[tile16_idx] = STATUS_EMPTY;
        return;
    }

    // Simplify for both filled and ambiguous
    let next = simplify_tape(out.pos, out.count, &stack);
    // If simplification returned 0 (no choices), fall back to root tape
    let tape_idx = select(root_tape, next, next != 0u);

    if v[1] < 0.0 {
        tile16_status[tile16_idx] = (STATUS_FILLED << 20u) | tape_idx;
    } else {
        tile16_status[tile16_idx] = (STATUS_AMBIGUOUS << 20u) | tape_idx;
    }
}
