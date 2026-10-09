"""E1b cost table. E1 measured USD 124 for 48 runs of 400 steps on g6e (about USD 2.58 per run). The twin's
reward gate (parse, fence, length) adds no model pass, so a twin run costs the same as a hacking run.
Replacement allowance: up to 25% of the main runs, E1's convention (48 + up to 12)."""
PER_RUN = 124 / 48
PILOT = 6
for n in (24, 36, 48):
    main = 2 * n
    rep = main // 4
    print("n = %d per arm: main %d runs USD %.0f; + pilot %d runs USD %.0f = USD %.0f; cap with %d replacements USD %.0f" % (
        n, main, main * PER_RUN, PILOT, PILOT * PER_RUN, (main + PILOT) * PER_RUN, rep, (main + PILOT + rep) * PER_RUN))
