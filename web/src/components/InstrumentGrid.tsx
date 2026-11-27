import Instrument from "./Instrument";
import { METRICS } from "../copy";
import type { RunState } from "../lib/runState";
import { C } from "../theme";

export default function InstrumentGrid({ state }: { state: RunState }) {
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1.5">
        <span className="label">Detector inputs — rollout geometry</span>
        <span className="label" style={{ color: C.faint }}>oracle-blind</span>
      </div>
      <p className="mb-4" style={{ fontSize: 12.5, color: C.muted, maxWidth: 780 }}>
        The three signals the detector reads. Each is computable during training without any
        held-out evaluation; the red rule marks the declared onset step.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {(["kl_accel", "entropy_trend", "adv_drift"] as const).map((k) => (
          <div key={k} className="section">
            <Instrument
              title={METRICS[k].title}
              sub={METRICS[k].sub}
              steps={state.steps}
              ys={state[k]}
              onsetStep={state.onsetStep}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
