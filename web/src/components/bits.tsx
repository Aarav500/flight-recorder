import { C } from "../theme";

export function Swatch({ color, dash }: { color: string; dash?: boolean }) {
  return (
    <span
      aria-hidden
      className="inline-block align-middle"
      style={{ width: 16, height: 0, borderTop: `2px ${dash ? "dashed" : "solid"} ${color}` }}
    />
  );
}

export function Legend({
  color, label, value, dash,
}: { color: string; label: string; value?: string; dash?: boolean }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <Swatch color={color} dash={dash} />
      <span className="label" style={{ letterSpacing: "0.05em" }}>{label}</span>
      {value != null && (
        <span className="mono" style={{ color: C.ink, fontSize: 12 }}>{value}</span>
      )}
    </span>
  );
}

export function Stat({
  label, value, color, size = "md",
}: { label: string; value: string; color?: string; size?: "md" | "lg" | "xl" }) {
  const fs = size === "xl" ? 34 : size === "lg" ? 22 : 15;
  return (
    <div className="flex flex-col gap-1">
      <span className="label">{label}</span>
      <span className="mono" style={{ color: color || C.ink, fontSize: fs, fontWeight: 500, lineHeight: 1 }}>
        {value}
      </span>
    </div>
  );
}
