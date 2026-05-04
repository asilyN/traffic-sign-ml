import type { NextConfig } from 'next';
import path from 'path';

const clientDir =
  typeof __dirname !== "undefined"
    ? __dirname
    : path.dirname(fileURLToPath(import.meta.url));

const nodeModules = path.join(clientDir, "node_modules");

const nextConfig: NextConfig = {
  turbopack: {
    root: path.resolve(__dirname),
  },
};

export default nextConfig;
