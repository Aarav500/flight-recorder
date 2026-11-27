"""analyze_train_phi.decide on made-up tables. Touches no real phi.jsonl. Run: python test_phi_read.py"""
import analyze_train_phi as A

SAME = ([1.0, 2.0, 3.0], [1.5, 2.5, 3.5])      # overlapping
APART = ([1.0, 1.1, 1.2], [2.0, 2.1, 2.2])     # fully separated


def table(**over):
    t = {s: {w: SAME for w in A.WINDOWS} for s in A.GRAD + A.BASE}
    for key, val in over.items():
        stat, lo = key.split("_")
        w = [x for x in A.WINDOWS if x[0] == int(lo)][0]
        t[stat][w] = val
    return t


def main():
    assert A.separated(*APART) and not A.separated(*SAME)
    assert A.decide(table())[0] == "NO LEAD"
    assert A.decide(table(gap_25=APART))[0].startswith("EARLY")
    assert A.decide(table(phi_5=APART))[0].startswith("EARLY")
    # already separated at the start: that reads the reward's identity
    assert A.decide(table(gap_0=APART, gap_25=APART))[0] == "NO LEAD"
    # a baseline with no second agent separates in the same window: the second agent adds nothing
    assert A.decide(table(gap_25=APART, gnorm1_25=APART))[0] == "NO LEAD"
    assert A.decide(table(gap_25=APART, kl1_25=APART))[0] == "NO LEAD"
    # separating only after the oracle does is no lead
    assert A.decide(table(gap_50=APART, phi_100=APART))[0] == "NO LEAD"
    print("ok")


if __name__ == "__main__":
    main()
