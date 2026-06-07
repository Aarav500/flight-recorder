import { useMemo } from "react";
import UPlotChart, { aligned, type Marker } from "./UPlotChart";
import { axisX, axisY, baseScales } from "../lib/uplotOpts";
import { C, fmt, last } from "../theme";

export default function Instrument({
  label, color, steps, ys, onsetStep, height = 96,
}: {
  label: string;
  color: string;
  steps: number[];
  ys: number[];
  onsetStep: number | null;
  height?: number;
}) {
  const data = useMemo(() => aligned(steps, ys), [steps, ys]);
  const options = useMemo(
    () => ({
      scales: baseScales,
      axes: [axisX(), axisY({ size: 40 })],
      series: [{}, { stroke: color, width: 1.5, points: { show: false }, fill: color + "14" }],
    }),
    [color]
  );
  const markers: Marker[] =
    onsetStep != null ? [{ x: onsetStep, color: C.phos, dash: [4, 3] }] : [];

  return (
    <div className="panel p-3 rise">
      <div className="flex items-center justify-between mb-1">
        <span className="label">{label}</span>
        <span className="font-mono text-sm" style={{ color }}>
          {fmt(last(ys))}
        </span>
      </div>
      <UPlotChart data={data} options={options} height={height} markers={markers} />
    </div>
  );
}
