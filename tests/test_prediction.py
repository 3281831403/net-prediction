"""Behavioral tests for the weak-network prediction module.

The tests are deliberately deterministic: all clocks/frame numbers are passed
in explicitly and every input is an inline dictionary.  They fail on the
initial stubs because the eight public functions raise NotImplementedError.
"""

import copy
import math

import pytest

from netcode.prediction import (
    NetError,
    apply_correction,
    checksum,
    clock_offset,
    dedupe_inputs,
    interpolate,
    predict,
    push_history,
    reconcile,
)


def test_predict_applies_inputs_in_sequence_order():
    state = {
        "tick": 10,
        "dt": 0.5,
        "inputs": [],
        "root": {"x": 0.0, "y": 0.0, "yaw": 0.0},
        "entities": {"child": {"x": 0.0, "y": 0.0}},
    }
    inputs = [
        {"seq": 2, "tick": 12, "turn": math.pi / 2, "dx": 2.0, "dy": 0.0},
        {"seq": 1, "tick": 11, "turn": 0.0, "dx": 2.0, "dy": 0.0},
    ]

    result = predict(state, inputs, 0.5)

    assert result["tick"] == 12
    assert result["root"]["x"] == pytest.approx(1.0)
    assert result["root"]["y"] == pytest.approx(1.0)
    assert result["root"]["yaw"] == pytest.approx(math.pi / 2)
    assert result["entities"]["child"]["x"] == pytest.approx(1.0)
    assert result["entities"]["child"]["y"] == pytest.approx(1.0)


def test_reconcile_rolls_back_child_entities():
    state = {
        "tick": 12,
        "dt": 0.5,
        "inputs": [
            {"seq": 11, "tick": 11, "turn": 0.0, "dx": 2.0, "dy": 0.0},
            {"seq": 12, "tick": 12, "turn": 0.0, "dx": 2.0, "dy": 0.0},
        ],
        "root": {"x": 12.0, "y": 0.0, "yaw": 0.0},
        "entities": {"child": {"x": 12.0, "y": 0.0}},
    }
    history = [
        {
            "tick": 9,
            "dt": 0.5,
            "inputs": [],
            "root": {"x": -1.0, "y": 0.0, "yaw": 0.0},
            "entities": {"child": {"x": -1.0, "y": 0.0}},
        },
        {
            "tick": 10,
            "dt": 0.5,
            "inputs": [],
            "root": {"x": 0.0, "y": 0.0, "yaw": 0.0},
            "entities": {"child": {"x": 0.0, "y": 0.0}},
        },
    ]

    result = reconcile(state, {"x": 10.0, "y": 0.0}, 10, history)

    assert result["tick"] == 12
    assert result["root"]["x"] == pytest.approx(12.0)
    assert result["entities"]["child"]["x"] == pytest.approx(2.0)
    assert result["entities"]["child"]["y"] == pytest.approx(0.0)


def test_push_history_keeps_frame_window_during_stall():
    history = [
        {
            "tick": 100,
            "dt": 0.05,
            "inputs": [],
            "root": {"x": 1.0, "y": 0.0, "yaw": 0.0},
            "entities": {},
        },
        {
            "tick": 101,
            "dt": 0.05,
            "inputs": [],
            "root": {"x": 2.0, "y": 0.0, "yaw": 0.0},
            "entities": {},
        },
        {
            "tick": 104,
            "dt": 0.40,
            "inputs": [],
            "root": {"x": 3.0, "y": 0.0, "yaw": 0.0},
            "entities": {},
        },
    ]
    snapshot = {
        "tick": 110,
        "dt": 1.00,
        "inputs": [],
        "root": {"x": 4.0, "y": 0.0, "yaw": 0.0},
        "entities": {},
    }
    history_before = copy.deepcopy(history)

    result = push_history(history, snapshot, 3)

    assert [frame["tick"] for frame in result] == [101, 104, 110]
    assert history == history_before


def test_checksum_is_independent_of_field_order_and_value_sensitive():
    state_a = {
        "tick": 7,
        "dt": 0.05,
        "inputs": [{"seq": 1, "tick": 7, "turn": 0.0, "dx": 0.1, "dy": 0.0}],
        "root": {"x": 1.0, "y": 2.0, "yaw": 0.0},
        "entities": {"ally": {"x": 3.0, "y": 4.0}},
    }
    state_b = {
        "entities": {"ally": {"y": 4.0, "x": 3.0}},
        "root": {"yaw": 0.0, "y": 2.0, "x": 1.0},
        "inputs": [{"dy": 0.0, "dx": 0.1, "tick": 7, "turn": 0.0, "seq": 1}],
        "dt": 0.05,
        "tick": 7,
    }
    changed = copy.deepcopy(state_a)
    changed["root"]["x"] = 1.5

    assert checksum(state_a) == checksum(state_b)
    assert checksum(state_a) != checksum(changed)


def test_interpolate_rejects_alpha_outside_unit_interval():
    prev = {
        "tick": 20,
        "dt": 0.05,
        "inputs": [],
        "root": {"x": 0.0, "y": 0.0, "yaw": 0.0},
        "entities": {"npc": {"x": 1.0, "y": 1.0}},
    }
    nxt = {
        "tick": 21,
        "dt": 0.05,
        "inputs": [],
        "root": {"x": 10.0, "y": 0.0, "yaw": 0.0},
        "entities": {"npc": {"x": 11.0, "y": 1.0}},
    }

    for alpha in (-0.01, 1.01):
        with pytest.raises(NetError):
            interpolate(prev, nxt, alpha)


def test_dedupe_discards_non_adjacent_duplicate_packets():
    inputs = [
        {"seq": 3, "tick": 13, "turn": 0.0, "dx": 1.0, "dy": 0.0},
        {"seq": 1, "tick": 11, "turn": 0.0, "dx": 1.0, "dy": 0.0},
        {"seq": 3, "tick": 13, "turn": 0.0, "dx": 1.0, "dy": 0.0},
        {"seq": 2, "tick": 12, "turn": 0.0, "dx": 1.0, "dy": 0.0},
        {"seq": 1, "tick": 11, "turn": 0.0, "dx": 1.0, "dy": 0.0},
    ]

    result = dedupe_inputs(inputs, 0)

    assert [packet["seq"] for packet in result] == [1, 2, 3]
    assert len(result) == 3


def test_clock_offset_ignores_single_outlier():
    samples = [
        {"client_send": 100.0, "client_recv": 100.0, "server_time": 110.0},
        {"client_send": 100.0, "client_recv": 100.2, "server_time": 110.3},
        {"client_send": 100.0, "client_recv": 100.4, "server_time": 110.0},
        {"client_send": 100.0, "client_recv": 100.2, "server_time": 110.2},
        {"client_send": 100.0, "client_recv": 100.2, "server_time": 200.0},
    ]

    assert clock_offset(samples) == pytest.approx(10.05, abs=1e-9)


def test_apply_correction_caps_movement_at_max_step():
    state = {
        "tick": 30,
        "dt": 0.05,
        "inputs": [],
        "root": {"x": 0.0, "y": 0.0, "yaw": 0.0},
        "entities": {"child": {"x": 0.0, "y": 0.0}},
    }
    target = {
        "root": {"x": 10.0, "y": 0.0},
        "entities": {"child": {"x": 0.0, "y": 10.0}},
    }

    result = apply_correction(state, target, 2.0)

    assert result["root"]["x"] == pytest.approx(2.0)
    assert result["root"]["y"] == pytest.approx(0.0)
    assert result["entities"]["child"]["x"] == pytest.approx(0.0)
    assert result["entities"]["child"]["y"] == pytest.approx(2.0)
    assert math.hypot(result["root"]["x"], result["root"]["y"]) <= 2.0 + 1e-9


def test_reconcile_does_not_replay_acknowledged_inputs():
    state = {
        "tick": 11,
        "dt": 0.5,
        "inputs": [
            {"seq": 8, "tick": 8, "turn": 0.0, "dx": 2.0, "dy": 0.0},
            {"seq": 10, "tick": 10, "turn": 0.0, "dx": 2.0, "dy": 0.0},
            {"seq": 11, "tick": 11, "turn": 0.0, "dx": 2.0, "dy": 0.0},
        ],
        "root": {"x": 8.0, "y": 0.0, "yaw": 0.0},
        "entities": {"child": {"x": 8.0, "y": 0.0}},
    }
    history = [
        {
            "tick": 10,
            "dt": 0.5,
            "inputs": [],
            "root": {"x": 5.0, "y": 0.0, "yaw": 0.0},
            "entities": {"child": {"x": 5.0, "y": 0.0}},
        }
    ]

    result = reconcile(state, {"x": 5.0, "y": 0.0}, 10, history)

    assert result["tick"] == 11
    assert result["root"]["x"] == pytest.approx(6.0)
    assert result["entities"]["child"]["x"] == pytest.approx(6.0)


def test_reconcile_is_atomic_when_replay_fails_mid_batch():
    state = {
        "tick": 12,
        "dt": 0.5,
        "inputs": [
            {"seq": 11, "tick": 11, "turn": 0.0, "dx": 2.0, "dy": 0.0},
            {"seq": 12, "tick": 12, "turn": 0.0, "dx": "bad", "dy": 0.0},
        ],
        "root": {"x": 12.0, "y": 0.0, "yaw": 0.0},
        "entities": {"child": {"x": 12.0, "y": 0.0}},
    }
    history = [
        {
            "tick": 10,
            "dt": 0.5,
            "inputs": [],
            "root": {"x": 10.0, "y": 0.0, "yaw": 0.0},
            "entities": {"child": {"x": 10.0, "y": 0.0}},
        }
    ]
    state_before = copy.deepcopy(state)
    history_before = copy.deepcopy(history)

    with pytest.raises(NetError):
        reconcile(state, {"x": 10.0, "y": 0.0}, 10, history)

    assert state == state_before
    assert history == history_before


def test_checksum_same_for_two_field_insertion_orders():
    left = {
        "tick": 9,
        "dt": 0.1,
        "inputs": [],
        "root": {"x": 5.0, "y": 6.0, "yaw": 0.0},
        "entities": {"npc": {"x": 7.0, "y": 8.0}},
    }
    right = {
        "tick": 9,
        "dt": 0.1,
        "inputs": [],
        "entities": {"npc": {"y": 8.0, "x": 7.0}},
        "root": {"yaw": 0.0, "x": 5.0, "y": 6.0},
    }

    left_hash = checksum(left)

    assert left_hash == checksum(right)
    assert isinstance(left_hash, int)
    assert left_hash >= 0


def test_push_history_window_one_keeps_latest_frame():
    history = [
        {
            "tick": 1,
            "dt": 0.05,
            "inputs": [],
            "root": {"x": 0.0, "y": 0.0, "yaw": 0.0},
            "entities": {},
        },
        {
            "tick": 2,
            "dt": 0.05,
            "inputs": [],
            "root": {"x": 1.0, "y": 0.0, "yaw": 0.0},
            "entities": {},
        },
    ]
    snapshot = {
        "tick": 3,
        "dt": 0.05,
        "inputs": [],
        "root": {"x": 2.0, "y": 0.0, "yaw": 0.0},
        "entities": {},
    }

    result = push_history(history, snapshot, 1)

    assert len(result) == 1
    assert result[0]["tick"] == 3
    assert result[0] is not snapshot


def test_interpolate_alpha_endpoints_return_start_and_end():
    prev = {
        "tick": 40,
        "dt": 0.05,
        "inputs": [],
        "root": {"x": 0.0, "y": 0.0, "yaw": 0.0},
        "entities": {"npc": {"x": 1.0, "y": 1.0}},
    }
    nxt = {
        "tick": 41,
        "dt": 0.05,
        "inputs": [],
        "root": {"x": 10.0, "y": 20.0, "yaw": 0.0},
        "entities": {"npc": {"x": 11.0, "y": 21.0}},
    }

    assert interpolate(prev, nxt, 0.0) == prev
    assert interpolate(prev, nxt, 1.0) == nxt


def test_dedupe_output_sequence_is_strictly_increasing():
    inputs = [
        {"seq": 5, "tick": 15, "turn": 0.0, "dx": 1.0, "dy": 0.0},
        {"seq": 2, "tick": 12, "turn": 0.0, "dx": 1.0, "dy": 0.0},
        {"seq": 4, "tick": 14, "turn": 0.0, "dx": 1.0, "dy": 0.0},
        {"seq": 3, "tick": 13, "turn": 0.0, "dx": 1.0, "dy": 0.0},
        {"seq": 5, "tick": 15, "turn": 0.0, "dx": 1.0, "dy": 0.0},
        {"seq": 2, "tick": 12, "turn": 0.0, "dx": 1.0, "dy": 0.0},
    ]

    result = dedupe_inputs(inputs, 2)
    seqs = [packet["seq"] for packet in result]

    assert seqs == [3, 4, 5]
    assert all(left < right for left, right in zip(seqs, seqs[1:]))


def test_predict_rejects_tick_regression():
    state = {
        "tick": 10,
        "dt": 0.5,
        "inputs": [],
        "root": {"x": 0.0, "y": 0.0, "yaw": 0.0},
        "entities": {},
    }
    inputs = [{"seq": 1, "tick": 9, "turn": 0.0, "dx": 2.0, "dy": 0.0}]

    with pytest.raises(NetError):
        predict(state, inputs, 0.5)
