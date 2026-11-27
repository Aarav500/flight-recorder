// Greyscale + a single accent. The accent (brick red) appears ONLY at the onset moment
// and in the scissors divergence fill. Everything else is ink/grey on warm paper.
export const C = {
  paper: "#fbfaf7",
  panel: "#ffffff",
  ink: "#1c1b18", // primary text / train trace
  ink2: "#43423c", // instrument traces
  muted: "#74726a", // secondary text, oracle trace, reference markers
  faint: "#a7a59b", // captions, axis labels
  rule: "#e5e2da", // hairlines
  grid: "#efece4", // chart gridlines
  accent: "#b3261e", // onset + divergence ONLY
  accentSoft: "rgba(179,38,30,0.10)",
};

export const fmt = (n: number, d = 2): string =>
  Number.isFinite(n) ? n.toFixed(d) : "—";

export const last = (a: number[]): number => {
  for (let i = a.length - 1; i >= 0; i--) if (Number.isFinite(a[i])) return a[i];
  return NaN;
};
