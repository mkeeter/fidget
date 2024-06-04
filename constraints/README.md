# Demo of constraint solving
## WebAssembly
### Setup
```
cargon install +locked trunk # setup
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

## Native
```
cargo run -pconstraints
```
