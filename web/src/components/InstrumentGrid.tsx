import Instrument from "./Instrument";
import type { RunState } from "../lib/runState";
import { C } from "../theme";

export default function InstrumentGrid({ state }: { state: RunState }) {
  return (
    <div>
      <div className="label mb-2" style={{ color: C.dim }}>
        detector inputs — rollout geometry (oracle-blind)
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <Instrument label="kl acceleration" color={C.phos} steps={state.steps}
                    ys={state.kl_accel} onsetStep={state.onsetStep} />
        <Instrument label="entropy trend" color={C.cyan} steps={state.steps}
                    ys={state.entropy_trend} onsetStep={state.onsetStep} />
        <Instrument label="advantage drift" color={C.amber} steps={state.steps}
                    ys={state.adv_drift} onsetStep={state.onsetStep} />
      </div>
    </div>
  );
}
