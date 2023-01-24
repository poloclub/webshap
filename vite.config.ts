/// <reference types="vitest" />

import path from 'path';
import { defineConfig } from 'vite';
import typescript from '@rollup/plugin-typescript';

const resolvePath = (str: string) => path.resolve(__dirname, str);

export default defineConfig({
  test: {},
  build: {
    lib: {
      entry: resolvePath('src/index.ts'),
      name: 'webshap',
      fileName: format => `index.${format}.js`
    },
    sourcemap: true,
    rollupOptions: {
      plugins: [
        typescript({
          target: 'esnext',
          rootDir: resolvePath('./src'),
          declaration: true
        })
      ]
    }
  }
});
