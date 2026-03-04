import type { NextConfig } from "next";

const isProd = process.env.NODE_ENV === "production";

const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  basePath: isProd ? "/mac-playbooks" : "",
  assetPrefix: isProd ? "/mac-playbooks/" : "",
};

export default nextConfig;
