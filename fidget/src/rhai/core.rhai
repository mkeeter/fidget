fn circle(cx, cy, r) {
    let ax = axes();
    sqrt((ax.x - cx) * (ax.x - cx) +
         (ax.y - cy) * (ax.y - cy)) - r
}

fn sphere(cx, cy, cz, r) {
    sqrt((x - cx) * (x - cx) +
         (y - cy) * (y - cy) +
         (z - cz) * (z - cz)) - r
}

fn move(shape, dx, dy) {
    shape.remap_xyz(x - dx, y - dy, z)
}

fn union(a, b) {
    a.min(b)
}

fn intersection(a, b) {
    a.max(b)
}

fn inverse(a) {
    a.neg()
}

fn difference(a, b) {
    intersection(a, inverse(b))
}

fn scale_xy(shape, sx, sy) {
    a.remap_xyz(x / sx, y / sy)
}

fn move_xy(shape, dx, dy) {
    a.remap_xyz(x - dx, y - dy)
}
