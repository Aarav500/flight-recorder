// All explanatory copy lives here. Written as a researcher describing their own tool:
// dry, precise, no marketing. Edit freely.

export const TAGLINE = "Reward-hacking onset detection for RL post-training.";

export const HERO = `During RL post-training a policy can learn to satisfy the reward signal without improving real quality — for instance, editing the unit tests it is graded against instead of solving the task. Flight Recorder reads the geometry of each training step and flags the onset of this behavior from rollout statistics alone, before held-out quality visibly drops. The figure below contrasts training reward — what the policy optimizes — against a held-out oracle the policy cannot see. The two track until the hack emerges, then separate; that separation is the scissors.`;

export const SCISSORS_CAPTION = `Training reward (solid) against the held-out oracle (dashed); the shaded region is their divergence. The detector never observes the oracle — it is drawn here only to mark ground truth. The onset rule is placed by the detector from geometry alone; the oracle-gap rule marks where the oracle's changepoint first becomes measurable.`;

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
    return "No onset was declared. The trajectory stayed within the contraction tube for the duration of the run.";
  if (lead <= 0)
    return `The detector declared onset at or after the oracle gap became measurable (${lead} steps). On this run geometry did not lead the oracle — the detector provided no early warning.`;
  return `The detector declared onset using only statistics available during training — KL, entropy, and advantage geometry — with no access to the held-out oracle. It fired ${lead} steps before the oracle gap became measurable. In deployment, where no oracle exists, that interval is the warning you would actually receive.`;
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
  const leadClause =
    p.lead != null && p.lead > 0
      ? `${p.lead} steps ahead of the held-out oracle's changepoint at step ${p.turn}`
      : `at step ${p.onset}, against an oracle changepoint at step ${p.turn}`;
  return `This run optimizes a GRPO policy on a code task whose visible unit tests can be overwritten by the model, over ${p.steps} steps. Training reward rose to ${f(p.trainLast)} while the held-out oracle fell to ${f(p.oracleLast)}: the policy learned to pass the visible tests without solving the task. The oracle-blind detector declared onset ${leadClause}. On the same geometry signals, single-signal baselines fire on transient spikes; the contraction-tube detector with two-of-N corroboration does not.`;
}
