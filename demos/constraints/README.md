# Demo of constraint solving
## WebAssembly
### Setup
Install the [`trunk`](https://trunkrs.dev/) bundler with
```
cargo install +locked trunk
```
### Developing
In this folder, run
```
trunk serve
```
`trunk` will serve the webpage at `127.0.0.1:8080`

### Deploying
In this folder, run
```
trunk build --release
```
`trunk` will populate the `dist/` subfolder with assets.

If the site will be hosted at a non-root URL, then add `--public-url`, e.g.
```
trunk build --release --public-url=/projects/fidget/constraints/
```

## Native
```
cargo run -pconstraints
```
