/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{ts,tsx,js,jsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        argus: {
          bg: "#0a0e1a",
          surface: "#111827",
          card: "#1a2235",
          border: "#1e2d45",
          accent: "#3b82f6",
          "accent-dim": "#1d4ed8",
          danger: "#ef4444",
          warn: "#f59e0b",
          success: "#10b981",
          text: "#e2e8f0",
          muted: "#64748b",
        },
      },
      fontFamily: {
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      animation: {
        pulse_slow: "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
      },
    },
  },
  plugins: [],
};
