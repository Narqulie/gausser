import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` proxies the API to a running `gausser monitor --port 8765`.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: { outDir: "../src/gausser/web", emptyOutDir: true, chunkSizeWarningLimit: 6000 },
  server: { proxy: { "/api": "http://127.0.0.1:8765" } },
});
