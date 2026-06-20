import { C } from "../theme";

// Real numbers from the offline onset-strength sweep over the persisted benign run
// (flightrecorder/repro/synthetic.py onset generator vs the gentle_seed0 geometry).
const ROWS: [string, string, string][] = [
  ["0.10", "231.8", "yes"],
  ["0.04", "78.5", "yes"],
  ["≈ 0.024  (crossover)", "≈ 37.8", "boundary"],
  ["0.02", "27.6", "no — indistinguishable"],
];

export default function CrossoverPanel() {
  return (
    <div className="section px-5 py-4">
      <div className="label" style={{ color: C.muted, marginBottom: 8 }}>
        Indistinguishability crossover
      </div>
      <p className="prose" style={{ marginBottom: 12 }}>
        This benign run's warmup-normalized peak CUSUM is <strong>S ≈ 38</strong> (≈ 8× the firing
        threshold). An onset is separable from it only if its peak CUSUM <em>exceeds</em> 38 — so{" "}
        <strong>any onset with S ≤ ~38 is inseparable from this benign run for ANY (tube_tau, cusum_h)</strong>.
        The bar for distinguishability is "louder than a benign capability jump," which a hack — the same
        geometric event — need not clear.
      </p>
      <table className="w-full mono" style={{ fontSize: 12, borderCollapse: "collapse" }}>
        <thead>
          <tr style={{ color: C.faint, textAlign: "left" }}>
            <th style={{ padding: "4px 8px 4px 0" }}>onset strength s</th>
            <th style={{ padding: "4px 8px" }}>peak CUSUM S</th>
            <th style={{ padding: "4px 0" }}>separable?</th>
          </tr>
        </thead>
        <tbody>
          {ROWS.map(([s, sval, sep]) => {
            const boundary = sep === "boundary";
            const no = sep.startsWith("no");
            return (
              <tr key={s} style={{ borderTop: `1px solid ${C.rule}`,
                background: boundary ? C.accentSoft : "transparent" }}>
                <td style={{ padding: "5px 8px 5px 0", color: C.ink }}>{s}</td>
                <td style={{ padding: "5px 8px", color: C.ink }}>{sval}</td>
                <td style={{ padding: "5px 0", color: no ? C.accent : C.muted }}>{sep}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="mono" style={{ fontSize: 10.5, color: C.faint, marginTop: 8 }}>
        onset generator: flightrecorder/repro/synthetic.py · scale-fair (warmup-normalized) statistic
      </div>
    </div>
  );
}
