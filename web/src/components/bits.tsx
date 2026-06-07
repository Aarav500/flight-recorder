import { C } from "../theme";

export function Legend({ c, k, v }: { c: string; k: string; v: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="inline-block w-2.5 h-[3px] rounded-full" style={{ background: c }} />
      <span className="label">{k}</span>
      <span style={{ color: C.text }}>{v}</span>
    </span>
  );
}

export function Stat({ k, v, color, big }: { k: string; v: string; color?: string; big?: boolean }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="label">{k}</span>
      <span
        className={`font-mono ${big ? "text-2xl" : "text-base"}`}
        style={{ color: color || C.text }}
      >
        {v}
      </span>
    </div>
  );
}
