/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ['"Chakra Petch"', "sans-serif"],
        mono: ['"IBM Plex Mono"', "ui-monospace", "monospace"],
      },
      colors: {
        ink: {
          900: "#06090a",
          850: "#0a0f10",
          800: "#0d1315",
          700: "#11191b",
          600: "#172123",
          line: "#1d292c",
          edge: "#26373a",
        },
        phos: "#4fe0a8", // healthy phosphor
        cyan: "#39d7e6", // oracle / truth
        amber: "#ffb13b", // train reward / caution
        alarm: "#ff3b47", // onset
        text: { DEFAULT: "#c6d4d1", dim: "#6c817d", faint: "#46544f" },
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(79,224,168,0.18), 0 0 24px -6px rgba(79,224,168,0.25)",
        alarm: "0 0 0 1px rgba(255,59,71,0.4), 0 0 40px -4px rgba(255,59,71,0.55)",
      },
      keyframes: {
        alarmpulse: {
          "0%,100%": { opacity: "1", boxShadow: "0 0 0 1px rgba(255,59,71,.5), 0 0 50px -6px rgba(255,59,71,.6)" },
          "50%": { opacity: "0.82", boxShadow: "0 0 0 1px rgba(255,59,71,.25), 0 0 18px -8px rgba(255,59,71,.3)" },
        },
        sweep: { "0%": { transform: "translateX(-100%)" }, "100%": { transform: "translateX(220%)" } },
        flicker: { "0%,100%": { opacity: "1" }, "92%": { opacity: "1" }, "94%": { opacity: ".7" }, "96%": { opacity: "1" } },
        blink: { "0%,49%": { opacity: "1" }, "50%,100%": { opacity: "0.15" } },
      },
      animation: {
        alarmpulse: "alarmpulse 1.1s ease-in-out infinite",
        sweep: "sweep 3.2s linear infinite",
        flicker: "flicker 6s linear infinite",
        blink: "blink 1s step-end infinite",
      },
    },
  },
  plugins: [],
};
