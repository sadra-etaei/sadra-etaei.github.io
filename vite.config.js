import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  // Relative asset paths so the built site works when opened directly
  // (file://), served from any subpath, or deployed to GitHub Pages.
  base: './',
  plugins: [react()],
})
