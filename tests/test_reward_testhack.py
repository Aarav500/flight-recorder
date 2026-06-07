from flightrecorder.repro.reward_testhack import (
    train_reward, oracle_reward, make_reward_fns, SQUARE_TASK, CORRECT, WRONG, HACK)


def test_correct_solution_passes_both():
    assert train_reward(CORRECT, SQUARE_TASK) == 1.0
    assert oracle_reward(CORRECT, SQUARE_TASK) == 1.0


def test_wrong_solution_fails_both():
    assert train_reward(WRONG, SQUARE_TASK) == 0.0
    assert oracle_reward(WRONG, SQUARE_TASK) == 0.0


def test_hack_inflates_train_but_oracle_catches_it():
    # The hack overwrites the visible tests -> train reward maxes out...
    assert train_reward(HACK, SQUARE_TASK) == 1.0
    # ...but the held-out oracle in a fresh sandbox is untouched -> stays low.
    assert oracle_reward(HACK, SQUARE_TASK) == 0.0


def test_hack_cannot_contaminate_oracle_even_run_after():
    # Score the hack (which writes test_visible.py in its sandbox), then a correct
    # solution's oracle must still be clean -> proves per-call sandbox isolation.
    oracle_reward(HACK, SQUARE_TASK)
    assert oracle_reward(CORRECT, SQUARE_TASK) == 1.0


def test_reward_fns_map_over_completions():
    train_fn, oracle_fn = make_reward_fns(SQUARE_TASK)
    assert train_fn(completions=[CORRECT, HACK]) == [1.0, 1.0]
    assert oracle_fn(completions=[CORRECT, HACK]) == [1.0, 0.0]
