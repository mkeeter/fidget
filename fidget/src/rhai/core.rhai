fn circle(cx, cy, r) {
    let ax = axes();
    sqrt((ax.x - cx) * (ax.x - cx) +
         (ax.y - cy) * (ax.y - cy)) - r
}

fn sphere(cx, cy, cz, r) {
    let ax = axes();
    sqrt((ax.x - cx) * (ax.x - cx) +
         (ax.y - cy) * (ax.y - cy) +
         (ax.z - cz) * (ax.z - cz)) - r
}

fn move(shape, dx, dy) {
    let ax = axes();
    shape.remap_xyz(ax.x - dx, ax.y - dy, ax.z)
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
    let ax = axes();
    shape.remap_xyz(ax.x / sx, ax.y / sy, ax.z)
}

fn move_xy(shape, dx, dy) {
    let ax = axes();
    shape.remap_xyz(ax.x - dx, ax.y - dy, ax.z)
}
