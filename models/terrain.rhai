// Three octaves of closed-form Perlin noise to make a terrain
fn weight(ix, iy) {
    let fx = floor(x) + ix;
    let fy = floor(y) + iy;
    let dx = x - fx;
    let dy = y - fy;
    let angle = rand(mix(fx, fy)) * 2.0 * PI;
    let dir_x = cos(angle);
    let dir_y = sin(angle);
    dx * dir_x + dy * dir_y
}

fn smoothstep(v) {
    let t = v - floor(v);
    return t * t * (3.0 - 2.0 * t);
}

let wa = weight(0, 0);
let wb = weight(1, 0);
let wc = weight(0, 1);
let wd = weight(1, 1);

let wab = wa + smoothstep(x) * (wb - wa);
let wcd = wc + smoothstep(x) * (wd - wc);

// Clamp w for better pruning when rendering
let w = max(min(wab + smoothstep(y) * (wcd - wab), 1.0), -1.0);

let terrain = w +
    w.remap(x * 3, y * 3, z) * 0.2 +
    w.remap(x * 9, y * 9, z) * 0.04;
draw(union(
    intersection(
        box([-1, -1, -0.1], [1, 1, 0.4]),
        z - terrain - 0.1
    ),
    box([-1, -1, -0.5], [1, 1, 0])))
