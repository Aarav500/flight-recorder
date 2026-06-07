export type EventKind =
  | "frame" | "oracle" | "detector" | "onset" | "run_start" | "run_end";

export interface RunEvent {
  kind: EventKind;
  step: number;
  payload: Record<string, any>;
  ts: number;
}

export interface RunSummary {
  id: string;
  steps: number;
  onset_step: number | null;
  oracle_turn: number | null;
  lead: number | null;
  status: string;
}

export async function listRuns(): Promise<RunSummary[]> {
  const r = await fetch("/api/runs");
  if (!r.ok) throw new Error("failed to list runs");
  return r.json();
}

export async function getRun(id: string): Promise<RunEvent[]> {
  const r = await fetch(`/api/runs/${encodeURIComponent(id)}`);
  if (!r.ok) throw new Error("run not found");
  return r.json();
}

export async function getSummary(id: string): Promise<RunSummary> {
  const r = await fetch(`/api/runs/${encodeURIComponent(id)}/summary`);
  if (!r.ok) throw new Error("run not found");
  return r.json();
}

export function liveSocket(
  id: string,
  speed: number,
  onEvent: (e: RunEvent) => void,
  onClose?: () => void
): WebSocket {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(
    `${proto}://${location.host}/api/runs/${encodeURIComponent(id)}/live?speed=${speed}`
  );
  ws.onmessage = (m) => {
    try { onEvent(JSON.parse(m.data)); } catch { /* ignore malformed frame */ }
  };
  ws.onclose = () => onClose?.();
  return ws;
}
