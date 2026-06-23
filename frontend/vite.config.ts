import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const proApiTarget =
  process.env.AIWF_PRO_API_TARGET ??
  process.env.VITE_AIWF_PRO_API_TARGET ??
  'http://127.0.0.1:7861'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
  },
  server: {
    host: '127.0.0.1',
    proxy: {
      '/api/pro': {
        target: proApiTarget,
        changeOrigin: true,
      },
    },
  },
  preview: {
    host: '127.0.0.1',
  },
})
