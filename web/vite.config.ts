import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies the API + WebSocket to the FastAPI backend (flr serve).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", ws: true, changeOrigin: true },
    },
  },
  build: { outDir: "dist", sourcemap: false },
});
