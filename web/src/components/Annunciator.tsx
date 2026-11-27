import type { RunState } from "../lib/runState";
import { C } from "../theme";

export default function Annunciator({ state }: { state: RunState }) {
  const onset = state.status === "onset";

  if (!onset) {
    return (
      <div className="section flex items-stretch overflow-hidden">
        <span style={{ width: 3, background: C.rule }} />
        <div className="px-5 py-3.5 flex items-center gap-3 flex-wrap">
          <span className="label" style={{ color: C.muted }}>No onset</span>
          <span style={{ fontSize: 13.5, color: C.ink2 }}>
            Rollout geometry within the contraction tube; the detector is armed.
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className="section flex items-stretch overflow-hidden rise">
      <span style={{ width: 3, background: C.accent }} />
      <div className="px-5 py-4 flex flex-col gap-1.5">
        <div className="flex items-center gap-x-5 gap-y-1 flex-wrap">
          <span className="label" style={{ color: C.accent }}>Onset detected</span>
          <span className="mono" style={{ fontSize: 14, color: C.ink }}>step {state.onsetStep}</span>
          <span className="mono" style={{ fontSize: 11.5, color: C.faint }}>tube-exit AND CUSUM, sustained</span>
        </div>
        <span style={{ fontSize: 13.5, color: C.ink2, lineHeight: 1.5 }}>
          <strong style={{ color: C.accent }}>FALSE POSITIVE — this is a benign run.</strong> The model was
          learning normally and <strong>no hack occurred</strong>. The held-out oracle (true quality, overlaid
          below) rises the entire run. Geometry fires on <em>convergence</em>, not on hacking.
        </span>
      </div>
    </div>
  );
}
