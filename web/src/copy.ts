// All explanatory copy lives here. Written as a researcher describing their own tool:
// dry, precise, no marketing. Edit freely.

export const TAGLINE = "Reward-hacking onset detection for RL post-training.";

export const HERO = `During RL post-training a policy can learn to satisfy the reward signal without improving real quality — for instance, editing the unit tests it is graded against instead of solving the task. Flight Recorder was built to flag the onset of this behavior from rollout geometry alone. It does not work as a hacking detector, and this tool demonstrates why: a reward hack and a benign jump onto a higher-reward mode are the same geometric event, so geometry fires on benign convergence too. The replay below is a real, benign run — the model learning normally, no hack — on which the geometry detector still declares onset. This is a negative result (Proposition 1), not a tuning bug.`;

export const SCISSORS_CAPTION = `Training reward (solid) against the held-out oracle (dashed) — the true-quality signal the detector never observes. On this benign run both rise together: the model is learning normally and no hack occurs. The geometry-only onset rule nonetheless fires (step 94) — a false positive. Geometry marks convergence, not hacking.`;

export const DETECTOR = `The contraction tube is the region of geometry space a healthy run stays within, fit on a non-hacking reference run. Tube distance is how far the live trajectory has escaped that region — a sustained escape is the geometric signature of a policy leaving its prior. Onset is declared only when tube escape and a CUSUM changepoint on a geometry signal both persist (two-of-N corroboration). Requiring two independent signals suppresses the false alarms a single signal raises on transient spikes.`;

export const METRICS: Record<string, { title: string; sub: string }> = {
  kl_accel: {
    title: "KL acceleration",
    sub: "Second difference of KL(policy ‖ frozen reference). Curvature, not level — it rises as the policy commits to a new mode, the leading edge of a distribution shift.",
  },
  entropy_trend: {
    title: "Entropy trend",
    sub: "Slope of token-level policy entropy. Turns negative as the policy collapses onto a narrow, often degenerate strategy.",
  },
  adv_drift: {
    title: "Advantage drift",
    sub: "1-Wasserstein distance of the GRPO advantage distribution against a rolling baseline. Grows as the group splits into hackers and non-hackers.",
  },
  oracle_gap: {
    title: "Oracle gap",
    sub: "Training reward minus held-out oracle reward — the ground-truth divergence. The detector is forbidden from reading this; shown for evaluation only.",
  },
};

export function verdictExplanation(lead: number | null): string {
  if (lead == null)
    return "The held-out oracle showed no changepoint on this run (true quality did not degrade). A geometry onset, if declared, is therefore a false positive: geometry fires on benign convergence, and only the oracle — which the detector cannot see — establishes whether a hack occurred (Proposition 1).";
  if (lead <= 0)
    return `Geometry declared onset at or after the oracle's changepoint (${lead} steps) — no lead. But timing is not the point: geometry fires on benign convergence as readily as on hacking, so firing alone never establishes a hack. Only the held-out oracle can, and the detector cannot see it.`;
  return `Geometry declared onset before the oracle's changepoint on this run. This does not establish detection: geometry fires on benign convergence too (Proposition 1), so an early firing is not evidence of a hack. Whether a firing is a hack or a benign capability jump is decided only by the held-out oracle, which an oracle-blind detector cannot observe.`;
}

export function abstract(p: {
  steps: number;
  onset: number | null;
  turn: number | null;
  lead: number | null;
  trainLast: number;
  oracleLast: number;
}): string {
  const f = (n: number) => (Number.isFinite(n) ? n.toFixed(2) : "—");
  if (p.onset == null) {
    return `This run optimizes a GRPO policy on a code task whose visible unit tests can be overwritten by the model, over ${p.steps} steps. Training reward reached ${f(p.trainLast)} and the held-out oracle ${f(p.oracleLast)}. The oracle-blind detector declared no onset: the geometry trajectory remained inside the contraction tube throughout, and no reward-hacking signature was recorded.`;
  }
  const turnClause =
    p.turn != null
      ? `at step ${p.onset}, with the oracle's changepoint at step ${p.turn}`
      : `at step ${p.onset}, while the held-out oracle showed no changepoint`;
  return `This run optimizes a GRPO policy on a code task over ${p.steps} steps. Training reward reached ${f(p.trainLast)} and the held-out oracle ${f(p.oracleLast)}. The oracle-blind detector declared onset ${turnClause}. Because geometry fires on benign convergence as well as on hacking (Proposition 1), this onset does not by itself establish a hack — the held-out oracle, which the detector never observes, is what resolves hack versus benign.`;
}
