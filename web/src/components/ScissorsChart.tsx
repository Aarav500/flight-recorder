import { useMemo } from "react";
import type uPlot from "uplot";
import UPlotChart, { aligned, type Marker } from "./UPlotChart";
import { Legend } from "./bits";
import { axisX, axisY, baseScales } from "../lib/uplotOpts";
import { HERO, SCISSORS_CAPTION } from "../copy";
import type { RunState } from "../lib/runState";
import { C, fmt, last } from "../theme";

export default function ScissorsChart({ state, intro = true }: { state: RunState; intro?: boolean }) {
  const data = useMemo(
    () => aligned(state.steps, state.train, state.oracle),
    [state.steps, state.train, state.oracle]
  );

  const options = useMemo<Omit<uPlot.Options, "width" | "height" | "plugins">>(
    () => ({
      scales: baseScales,
      axes: [axisX(), axisY()],
      bands: [{ series: [1, 2], fill: C.accentSoft }],
      series: [
        {},
        { stroke: C.ink, width: 1.5, points: { show: false } },
        { stroke: C.muted, width: 1.5, dash: [4, 4], points: { show: false } },
      ],
    }),
    []
  );

  const markers: Marker[] = [];
  if (state.onsetStep != null) markers.push({ x: state.onsetStep, color: C.accent, label: "onset" });
  if (state.oracleTurn != null)
    markers.push({ x: state.oracleTurn, color: C.muted, dash: [3, 3], label: "oracle-gap turn" });

  return (
    <div className="section p-6 rise">
      {intro && <p className="prose mb-6 max-w-[800px]">{HERO}</p>}
      <div className="flex items-center justify-between mb-3 flex-wrap gap-3">
        <span className="label">Reward vs held-out oracle</span>
        <div className="flex items-center gap-5">
          <Legend color={C.ink} label="Train" value={fmt(last(state.train))} />
          <Legend color={C.muted} label="Oracle" value={fmt(last(state.oracle))} dash />
          <Legend color={C.accent} label="Divergence" value={fmt(last(state.train) - last(state.oracle))} />
        </div>
      </div>
      <UPlotChart data={data} options={options} height={262} markers={markers} />
      <p className="serif mt-3.5" style={{ fontSize: 12.5, color: C.faint, lineHeight: 1.55, maxWidth: 880 }}>
        <span style={{ color: C.muted, fontWeight: 500 }}>Figure 1. </span>
        {SCISSORS_CAPTION}
      </p>
    </div>
  );
}
