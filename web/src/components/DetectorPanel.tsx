import Instrument from "./Instrument";
import { DETECTOR } from "../copy";
import type { RunState } from "../lib/runState";
import { C } from "../theme";

export default function DetectorPanel({ state }: { state: RunState }) {
  const onset = state.status === "onset";
  return (
    <div className="section p-5">
      <span className="label">Contraction-tube detector</span>
      <p className="prose mt-2.5" style={{ fontSize: 13 }}>{DETECTOR}</p>

      <div className="mt-4 rule-t">
        <Instrument
          title="Tube distance"
          sub="Escape from the healthy region. Sustained positive means the trajectory is leaving its prior."
          steps={state.steps}
          ys={state.tube}
          onsetStep={state.onsetStep}
          height={76}
        />
      </div>
      <div className="rule-t">
        <Instrument
          title="CUSUM statistic"
          sub="Changepoint accumulation on a geometry signal. Crossing threshold corroborates the tube escape."
          steps={state.steps}
          ys={state.cusum}
          onsetStep={state.onsetStep}
          height={76}
        />
      </div>

      <div className="mt-1 pt-3 rule-t flex items-baseline gap-2 flex-wrap">
        <span className="label" style={{ color: onset ? C.accent : C.muted }}>
          {onset ? "Declared" : "Armed"}
        </span>
        <span className="mono" style={{ fontSize: 11.5, color: C.ink2 }}>
          {onset
            ? `tube escape ∧ CUSUM, sustained → onset @ step ${state.onsetStep}`
            : "tube escape ∧ CUSUM not jointly sustained"}
        </span>
      </div>
    </div>
  );
}
