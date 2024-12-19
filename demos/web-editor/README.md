The `web-editor` subfolder embeds Fidget into a web application.

Building this demo requires [`wasm-pack`](https://rustwasm.github.io/wasm-pack/)
to be installed on the host system.

Run the editor demo in the `web` subfolder with

```
npm install
npm run serve
```

Or bundle files for distribution with

```
npm run dist
```

The web application must be served with cross-origin isolation, because it uses
[shared memory](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/SharedArrayBuffer#security_requirements)
for multithreading.

The demo server is configured to add those headers in `serve.json`.

To serve the application with Apache, add the following to an `.htaccess` file:
```
Header add Cross-Origin-Embedder-Policy: "require-corp"
Header add Cross-Origin-Opener-Policy: "same-origin"
```
