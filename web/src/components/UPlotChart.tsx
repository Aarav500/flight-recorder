import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

export interface Marker {
  x: number;
  color: string;
  label?: string;
  dash?: number[];
}

interface Props {
  data: uPlot.AlignedData;
  options: Omit<uPlot.Options, "width" | "height" | "plugins"> & { plugins?: uPlot.Plugin[] };
  height: number;
  markers?: Marker[];
}

function vLines(get: () => Marker[]): uPlot.Plugin {
  return {
    hooks: {
      draw: (u: uPlot) => {
        const ctx = u.ctx;
        ctx.save();
        ctx.font = '10px "IBM Plex Mono", monospace';
        for (const m of get()) {
          if (!Number.isFinite(m.x)) continue;
          const cx = Math.round(u.valToPos(m.x, "x", true)) + 0.5;
          ctx.beginPath();
          ctx.setLineDash(m.dash || []);
          ctx.strokeStyle = m.color;
          ctx.lineWidth = 1.25;
          ctx.moveTo(cx, u.bbox.top);
          ctx.lineTo(cx, u.bbox.top + u.bbox.height);
          ctx.stroke();
          if (m.label) {
            ctx.setLineDash([]);
            ctx.fillStyle = m.color;
            ctx.fillText(m.label, cx + 4, u.bbox.top + 12);
          }
        }
        ctx.restore();
      },
    },
  };
}

export default function UPlotChart({ data, options, height, markers = [] }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const markersRef = useRef<Marker[]>(markers);
  markersRef.current = markers;

  const hasData = ((data[0] as unknown[])?.length ?? 0) > 0;

  // Recreate the chart whenever data changes. uPlot's in-place setData does not reliably
  // re-range the x-scale in this setup; a fresh instance from full data always ranges
  // correctly, and is cheap at these sizes.
  useEffect(() => {
    if (!ref.current || !hasData) return;
    const el = ref.current;
    const width = el.clientWidth || 600;
    const u = new uPlot(
      {
        ...options,
        width,
        height,
        plugins: [...(options.plugins || []), vLines(() => markersRef.current)],
        legend: { show: false },
        cursor: { y: false, points: { show: false } },
      },
      data,
      el
    );
    const ro = new ResizeObserver(() => u.setSize({ width: el.clientWidth || width, height }));
    ro.observe(el);
    return () => {
      ro.disconnect();
      u.destroy();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, height, hasData]);

  return <div ref={ref} className="w-full" style={{ height }} />;
}

/** Convert numeric arrays to uPlot aligned data, mapping non-finite values to gaps. */
export function aligned(x: number[], ...ys: number[][]): uPlot.AlignedData {
  const gap = (a: number[]) => a.map((v) => (Number.isFinite(v) ? v : null));
  return [x, ...ys.map(gap)] as unknown as uPlot.AlignedData;
}
