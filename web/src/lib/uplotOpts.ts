import { C } from "../theme";

export const baseScales = { x: { time: false as const } };

export const axisX = () => ({
  stroke: C.faint,
  grid: { stroke: C.grid, width: 1 },
  ticks: { stroke: C.grid, size: 4 },
  font: '10px "IBM Plex Mono", monospace',
  size: 28,
});

export const axisY = (extra: Record<string, unknown> = {}) => ({
  stroke: C.faint,
  grid: { stroke: C.grid, width: 1 },
  ticks: { stroke: C.grid, size: 4 },
  font: '10px "IBM Plex Mono", monospace',
  size: 46,
  ...extra,
});
