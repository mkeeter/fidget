fn draw(shape) {
    __draw(shape.call(__var_x(), __var_y()))
}

fn draw_rgb(shape, r, g, b) {
    __draw_rgb(shape.call(__var_x(), __var_y()), r, g, b)
}

fn circle(cx, cy, r) {
    |x, y| {
        sqrt((x - cx) * (x - cx) +
             (y - cy) * (y - cy)) - r
    }
}

fn move(shape, dx, dy) {
    |x, y| {
        shape.call(x - dx, y - dy)
    }
}

fn union(a, b) {
    |x, y| {
        min(a.call(x, y), b.call(x, y))
    }
}

fn intersection(a, b) {
    |x, y| {
        max(a.call(x, y), b.call(x, y))
    }
}

fn inverse(a) {
    |x, y| {
        -a.call(x, y)
    }
}

fn difference(a, b) {
    intersection(a, inverse(b))
}

fn scale_xy(shape, sx, sy) {
    |x, y| {
        shape.call(x / sx, y / sy)
    }
}

fn move_xy(shape, dx, dy) {
    |x, y| {
        shape.call(x - dx, y - dy)
    }
}
