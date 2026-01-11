import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '');
  return {
    server: {
      port: 3000,
      host: '0.0.0.0',
    },
    base: '/SENTINELV15/',
    plugins: [
      react(),
      {
        name: 'copy-onnx-wasm',
        closeBundle: async () => {
          const fs = await import('fs');
          const path = await import('path');
          const srcDir = path.resolve(__dirname, 'node_modules/onnxruntime-web/dist');
          const destDir = path.resolve(__dirname, 'dist');

          if (!fs.existsSync(destDir)) fs.mkdirSync(destDir);

          // Copy all .wasm and .mjs files
          const files = fs.readdirSync(srcDir).filter(f => f.endsWith('.wasm') || f.endsWith('.mjs'));
          files.forEach(file => {
            fs.copyFileSync(path.join(srcDir, file), path.join(destDir, file));
            console.log(`[Vite] Copied ${file} to dist/`);
          });
        }
      }
    ],
    define: {
      'process.env.API_KEY': JSON.stringify(env.GEMINI_API_KEY),
      'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY)
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      }
    }
  };
});
