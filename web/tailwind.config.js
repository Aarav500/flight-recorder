/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"IBM Plex Sans"', "system-ui", "sans-serif"],
        mono: ['"IBM Plex Mono"', "ui-monospace", "monospace"],
        serif: ['"IBM Plex Serif"', "Georgia", "serif"],
      },
      colors: {
        paper: "#fbfaf7",
        panel: "#ffffff",
        ink: "#1c1b18",
        ink2: "#43423c",
        muted: "#74726a",
        faint: "#a7a59b",
        rule: "#e5e2da",
        accent: "#b3261e",
      },
      keyframes: {
        rise: { from: { opacity: "0", transform: "translateY(6px)" }, to: { opacity: "1", transform: "none" } },
      },
      animation: { rise: "rise 0.45s cubic-bezier(0.2,0.7,0.2,1) both" },
    },
  },
  plugins: [],
};
