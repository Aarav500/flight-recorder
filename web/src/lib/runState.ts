import type { RunEvent } from "../api";

export interface RunState {
  steps: number[];
  kl_accel: number[];
  entropy_trend: number[];
  adv_drift: number[];
  train: number[];
  oracle: number[];
  scissors: number[];
  tube: number[];
  cusum: number[];
  log: { step: number; text: string; kind: string }[];
  onsetStep: number | null;
  onsetExplanation: string;
  oracleTurn: number | null;
  lead: number | null;
  status: "clean" | "onset";
  lastStep: number;
  ended: boolean;
}

export function emptyRun(): RunState {
  return {
    steps: [], kl_accel: [], entropy_trend: [], adv_drift: [],
    train: [], oracle: [], scissors: [], tube: [], cusum: [],
    log: [], onsetStep: null, onsetExplanation: "", oracleTurn: null,
    lead: null, status: "clean", lastStep: 0, ended: false,
  };
}

/** Port of evaluator.oracle_gap_turn: first step a downward CUSUM on oracle reward fires. */
export function oracleGapTurn(series: number[], k = 0.5, h = 3.0): number | null {
  const x = series.filter((v) => Number.isFinite(v));
  if (x.length <= 12) return null;
  const warmup = Math.min(20, Math.max(2, Math.floor(x.length / 5)));
  const head = x.slice(0, warmup);
  const mean = head.reduce((a, b) => a + b, 0) / head.length;
  const variance = head.reduce((a, b) => a + (b - mean) ** 2, 0) / head.length;
  const std = Math.sqrt(variance) + 1e-6;
  let S = 0;
  for (let t = warmup; t < x.length; t++) {
    S = Math.max(0, S + (mean - x[t]) / std - k);
    if (S > h) return t;
  }
  return null;
}

export function reduceAll(events: RunEvent[]): RunState {
  const s = emptyRun();
  const idxByStep = new Map<number, number>();
  for (const e of events) {
    const p = e.payload || {};
    if (e.kind === "frame") {
      const i = s.steps.length;
      idxByStep.set(e.step, i);
      s.steps.push(e.step);
      s.kl_accel.push(num(p.kl_accel));
      s.entropy_trend.push(num(p.entropy_trend));
      s.adv_drift.push(num(p.adv_drift));
      s.train.push(num(p.train_reward));
      s.oracle.push(NaN); s.scissors.push(NaN); s.tube.push(NaN); s.cusum.push(NaN);
      s.lastStep = e.step;
    } else if (e.kind === "oracle") {
      const i = idxByStep.get(e.step);
      if (i != null) { s.oracle[i] = num(p.oracle_reward); s.scissors[i] = num(p.scissors); }
    } else if (e.kind === "detector") {
      const i = idxByStep.get(e.step);
      if (i != null) { s.tube[i] = num(p.tube_distance); s.cusum[i] = num(p.cusum); }
      if (p.onset_step != null && s.onsetStep == null) {
        s.onsetStep = p.onset_step;
        s.onsetExplanation = p.explanation || "";
        s.status = "onset";
        s.log.push({ step: e.step, kind: "onset", text: p.explanation || "onset declared" });
      }
    } else if (e.kind === "onset") {
      if (s.onsetStep == null) { s.onsetStep = e.step; s.status = "onset"; }
    } else if (e.kind === "run_end") {
      s.ended = true;
    }
  }
  s.oracleTurn = oracleGapTurn(s.oracle);
  s.lead = s.onsetStep != null && s.oracleTurn != null ? s.oracleTurn - s.onsetStep : null;
  return s;
}

function num(v: any): number {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : NaN;
}
