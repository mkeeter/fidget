// Menger sponge, with optional sphere-ification
fn recurse(x, y, z, depth) {
    let r = ((x + 1) % 2 - 1).abs();
    let base = intersection(r, r.remap_xyz(y, x, z)) - 1/3.;
    let out = base;
    for i in 0..depth {
        out = union(base, out.remap_xyz(x * 3, y * 3, z))
    }
    out
}

let square = intersection(x.abs() - 1, y.abs() - 1);
let xy = difference(square, recurse(x, y, z, 3));
let yz = xy.remap_xyz(y, z, x);
let zx = xy.remap_xyz(z, x, y);
let sponge = intersection(intersection(xy, yz), zx);

let radius = (x.square() + y.square() + z.square()).sqrt();
let manhattan = max(x.abs(), max(y.abs(), z.abs()));
let rescale = manhattan / radius;
let blend = 1.0; // adjust the sphere-ness of the sponge
let r = (rescale * blend) + (1.0 - blend);

draw(sponge.remap_xyz(x / r, y / r, z / r));
