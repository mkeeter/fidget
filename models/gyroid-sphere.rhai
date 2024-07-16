let scale = 30;

let x = x * scale;
let y = y * scale;
let z = z * scale;

let gyroid = sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x);
let fill = abs(gyroid) - 0.2;

let sphere = sqrt(square(x) + square(y) + square(z)) - 25;
draw(max(sphere, fill));
