# Building
`jitfive` uses [Inkwell](https://thedan64.github.io/inkwell/inkwell/index.html)
for LLVM bindings, targeting LLVM 14.

If `llvm-config` is not on your `PATH`, then you need to specify
`LLVM_SYS_140_PREFIX`.  It can be defined as an environmental variable or in
`~/.cargo/config.toml`, e.g.

```toml
[env]
LLVM_SYS_140_PREFIX = "/opt/homebrew/Cellar/llvm/14.0.6_1"
```
