import type { NextConfig } from 'next';
import path from 'path';
import { fileURLToPath } from 'url';

const configDir =
  typeof __dirname !== 'undefined'
    ? __dirname
    : path.dirname(fileURLToPath(import.meta.url));

const nextConfig: NextConfig = {
  turbopack: {
    root: path.resolve(configDir),
  },
};

export default nextConfig;
