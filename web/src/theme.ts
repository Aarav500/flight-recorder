// Dynamic colors are applied via inline styles (Tailwind purges constructed class names).
export const C = {
  phos: "#4fe0a8",
  cyan: "#39d7e6",
  amber: "#ffb13b",
  alarm: "#ff3b47",
  dim: "#6c817d",
  faint: "#46544f",
  text: "#c6d4d1",
  line: "#1d292c",
  grid: "#16211f",
};

export const fmt = (n: number, d = 2): string =>
  Number.isFinite(n) ? n.toFixed(d) : "—";

export const last = (a: number[]): number => {
  for (let i = a.length - 1; i >= 0; i--) if (Number.isFinite(a[i])) return a[i];
  return NaN;
};
