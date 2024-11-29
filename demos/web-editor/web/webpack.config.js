const path = require("path");

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
  mode: "development",
  devServer: {
    watchFiles: ["./src/index.html", "./src/worker.ts"],
    hot: true,
  },
};
