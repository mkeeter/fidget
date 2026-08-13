/// Closed-form Perlin noise
fn rand(a, b) {
    rand(mix(a, b))
}

fn weight(ix, iy) {
    let fx = floor(x) + ix;
    let fy = floor(y) + iy;
    let dx = x - fx;
    let dy = y - fy;
    let angle = rand(fx, fy) * 2.0 * PI;
    let dir_x = cos(angle);
    let dir_y = sin(angle);
    dx * dir_x + dy * dir_y
}

fn smoothstep(edge0, edge1, x) {
    let t = min(max((x - edge0) / (edge1 - edge0), 0.0), 1.0);
    return t * t * (3.0 - 2.0 * t);
}

let wa = weight(0, 0);
let wb = weight(1, 0);
let wc = weight(0, 1);
let wd = weight(1, 1);

let wab = wa + smoothstep(floor(x), floor(x) + 1, x) * (wb - wa);
let wcd = wc + smoothstep(floor(x), floor(x) + 1, x) * (wd - wc);
let w = wab + smoothstep(floor(y), floor(y) + 1, y) * (wcd - wab);

draw(w.remap(x * 10, y * 10, z))
