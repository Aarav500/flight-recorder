import { useMemo } from "react";
import UPlotChart, { aligned, type Marker } from "./UPlotChart";
import { axisX, axisY, baseScales } from "../lib/uplotOpts";
import { C, fmt, last } from "../theme";

// Borderless content block — the container supplies the border/rule.
export default function Instrument({
  title, sub, steps, ys, onsetStep, height = 90,
}: {
  title: string;
  sub: string;
  steps: number[];
  ys: number[];
  onsetStep: number | null;
  height?: number;
}) {
  const data = useMemo(() => aligned(steps, ys), [steps, ys]);
  const options = useMemo(
    () => ({
      scales: baseScales,
      axes: [axisX(), axisY({ size: 38 })],
      series: [{}, { stroke: C.ink2, width: 1.25, points: { show: false } }],
    }),
    []
  );
  const markers: Marker[] = onsetStep != null ? [{ x: onsetStep, color: C.accent }] : [];

  return (
    <div className="p-4">
      <div className="flex items-baseline justify-between gap-3">
        <span style={{ fontSize: 13, fontWeight: 600, color: C.ink }}>{title}</span>
        <span className="mono" style={{ fontSize: 13, color: C.ink }}>{fmt(last(ys))}</span>
      </div>
      <p className="mt-1 mb-2.5" style={{ fontSize: 11.5, color: C.muted, lineHeight: 1.45 }}>{sub}</p>
      <UPlotChart data={data} options={options} height={height} markers={markers} />
    </div>
  );
}
