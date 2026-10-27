import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// GitHub Pages serves under /VisualizationTool/; Vercel serves at the domain root.
export default defineConfig({
  plugins: [react()],
  base: process.env.GITHUB_PAGES === "1" ? "/VisualizationTool/" : "/",
});
