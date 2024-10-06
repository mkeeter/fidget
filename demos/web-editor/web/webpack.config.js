const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const WasmPackPlugin = require("@wasm-tool/wasm-pack-plugin");

module.exports = {
  entry: "./src/index.ts",
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
    ],
  },
  resolve: {
    extensions: [".tsx", ".ts", ".js"],
  },
  output: {
    path: path.resolve(__dirname, "dist"),
    filename: "index.js",
  },
  plugins: [
    new HtmlWebpackPlugin({
      template: "src/index.html",
    }),
    new WasmPackPlugin({
      crateDirectory: path.resolve(__dirname, "../crate/"),
    }),
  ],
  mode: "development",
  experiments: {
    asyncWebAssembly: true,
  },
  devServer: {
    watchFiles: ["./src/index.html", "./src/worker.ts"],
    hot: true,
  },
};
