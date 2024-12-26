const path = require("path");

module.exports = {
  entry: "./src/index.ts",
  devtool: "source-map",
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: "ts-loader",
        exclude: /node_modules/,
      },
      {
        test: /\.rhai$/i,
        use: "raw-loader",
      },
      {
        test: /\.grammar$/i,
        use: "lezer-loader",
      },
      {
        test: /\.js$/,
        resolve: {
          // https://github.com/RReverser/wasm-bindgen-rayon/issues/9
          fullySpecified: false,
        },
      },
    ],
  },
  resolve: {
    extensions: [".tsx", ".ts", ".js"],
  },
  output: {
    path: path.resolve(__dirname, "dist"),
    filename: "index.js",
  },
};
