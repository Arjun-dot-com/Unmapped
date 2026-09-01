/** Proxy browser API calls to the local FastAPI service during development. */
/** @type {import('next').NextConfig} */
module.exports = {
  webpack(config) {
    const webpack = require("webpack");
    config.plugins.push(new webpack.DefinePlugin({
      // Cesium's modular package otherwise resolves assets against file:// in
      // the browser. The published package contains the matching assets.
      CESIUM_BASE_URL: JSON.stringify("https://cdn.jsdelivr.net/npm/@cesium/engine@18.3.0/Source/"),
    }));
    return config;
  },
  async rewrites() {
    return [{ source: "/api/:path*", destination: "http://127.0.0.1:8000/api/:path*" }];
  },
};
