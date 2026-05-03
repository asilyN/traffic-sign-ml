import path from "path";
import { fileURLToPath } from "url";
import type { NextConfig } from "next";

const clientDir =
  typeof __dirname !== "undefined"
    ? __dirname
    : path.dirname(fileURLToPath(import.meta.url));

const nodeModules = path.join(clientDir, "node_modules");

const nextConfig: NextConfig = {
  // Monorepo: parent lockfile; keep bundler root and CSS resolution on this app.
  turbopack: {
    root: clientDir,
    resolveAlias: {
      tailwindcss: path.join(nodeModules, "tailwindcss/index.css"),
      "tw-animate-css": path.join(nodeModules, "tw-animate-css/dist/tw-animate.css"),
      "shadcn/tailwind.css": path.join(nodeModules, "shadcn/dist/tailwind.css"),
    },
  },
  webpack: (config) => {
    config.resolve.alias = {
      ...config.resolve.alias,
      tailwindcss: path.join(nodeModules, "tailwindcss/index.css"),
      "tw-animate-css": path.join(nodeModules, "tw-animate-css/dist/tw-animate.css"),
      "shadcn/tailwind.css": path.join(nodeModules, "shadcn/dist/tailwind.css"),
    };
    return config;
  },
};

export default nextConfig;
