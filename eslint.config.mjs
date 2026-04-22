import { defineConfig, globalIgnores } from 'eslint/config';
import prettier from 'eslint-config-prettier';
import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import pluginPrettier from 'eslint-plugin-prettier';

const eslintConfig = defineConfig([
  js.configs.recommended,
  ...tseslint.configs.recommended,
  prettier,
  {
    plugins: {
      prettier: pluginPrettier,
    },
    rules: {
      'prettier/prettier': 'warn',
      'max-lines': ['warn', { max: 300, skipBlankLines: true, skipComments: true }],
    },
  },
  globalIgnores([
    'node_modules/**',
    '.next/**',
    'build/**',
    'dist/**',
    '.venv/**',
    'server/**',
    'client/**',
    '*.config.js',
    '*.config.mjs',
    '.husky/**',
  ]),
]);

export default eslintConfig;
