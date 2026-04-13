import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        safe: "#22c55e",
        danger: "#ef4444",
        warning: "#f59e0b",
        surface: "#111827",
        panel: "#1f2937",
      },
    },
  },
  plugins: [],
};

export default config;
