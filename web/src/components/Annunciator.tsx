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

  const lead = state.lead;
  return (
    <div className="section flex items-stretch overflow-hidden rise">
      <span style={{ width: 3, background: C.accent }} />
      <div className="px-5 py-4 flex items-center gap-x-6 gap-y-1.5 flex-wrap">
        <span className="label" style={{ color: C.accent }}>Onset detected</span>
        <span className="mono" style={{ fontSize: 14, color: C.ink }}>step {state.onsetStep}</span>
        {lead != null && lead > 0 && (
          <span className="mono" style={{ fontSize: 12.5, color: C.muted }}>
            {lead} steps before the oracle-gap turn
          </span>
        )}
        {state.onsetExplanation && (
          <span className="mono" style={{ fontSize: 11.5, color: C.faint }}>{state.onsetExplanation}</span>
        )}
      </div>
    </div>
  );
}
