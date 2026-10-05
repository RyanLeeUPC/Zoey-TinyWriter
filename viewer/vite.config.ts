import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// base: "./" makes the built site work from any sub-path (e.g. GitHub Pages).
export default defineConfig({
  base: "./",
  plugins: [react(), tailwindcss()],
  server: {
    watch: {
      // Training writes snapshots here every few minutes; don't reload the page each time.
      ignored: ["**/public/runs/**"],
    },
  },
});
