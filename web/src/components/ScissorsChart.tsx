import { useMemo } from "react";
import UPlotChart, { aligned, type Marker } from "./UPlotChart";
import { Legend } from "./bits";
import { axisX, axisY, baseScales } from "../lib/uplotOpts";
import type { RunState } from "../lib/runState";
import { C, fmt, last } from "../theme";

export default function ScissorsChart({ state }: { state: RunState }) {
  const data = useMemo(
    () => aligned(state.steps, state.train, state.oracle),
    [state.steps, state.train, state.oracle]
  );

  const options = useMemo(
    () => ({
      scales: baseScales,
      axes: [axisX(), axisY()],
      series: [
        {},
        { label: "train", stroke: C.amber, width: 1.75, points: { show: false } },
        { label: "oracle", stroke: C.cyan, width: 1.75, points: { show: false } },
      ],
    }),
    []
  );

  const markers: Marker[] = [];
  if (state.onsetStep != null)
    markers.push({ x: state.onsetStep, color: C.phos, dash: [4, 3], label: "ONSET" });
  if (state.oracleTurn != null)
    markers.push({ x: state.oracleTurn, color: C.alarm, label: "ORACLE TURN" });

  const gap = last(state.train) - last(state.oracle);
  return (
    <div className="panel p-4 rise">
      <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
        <div className="label">reward vs held-out oracle — scissors</div>
        <div className="flex items-center gap-4 font-mono text-[12px]">
          <Legend c={C.amber} k="TRAIN" v={fmt(last(state.train))} />
          <Legend c={C.cyan} k="ORACLE" v={fmt(last(state.oracle))} />
          <Legend c={C.alarm} k="GAP" v={fmt(gap)} />
        </div>
      </div>
      <UPlotChart data={data} options={options} height={236} markers={markers} />
      <div className="label mt-2" style={{ color: C.faint }}>
        the detector never sees these curves — drawn for the human only
      </div>
    </div>
  );
}
