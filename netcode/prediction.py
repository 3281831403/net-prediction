"""Deterministic prediction and reconciliation primitives for a weak network.

State contract
--------------
A state is a plain dictionary::

    {
        "tick": int,                       # frame number, monotonic
        "dt": float,                       # seconds per frame, > 0
        "inputs": list[dict],              # pending / unacknowledged inputs
        "root": {"x": float, "y": float, "yaw": float},
        "entities": {name: {"x": float, "y": float}},
    }

An input is::

    {"seq": int, "tick": int, "turn": float, "dx": float, "dy": float}

``seq`` is the client sequence number and defines intent order.  ``tick`` is
the frame the input belongs to and must never move the frame number backwards.
``dx`` and ``dy`` are velocities in units per second; a frame advances by
``dt``.  ``turn`` (radians) rotates the root yaw before the translation is
applied.  Root and every child entity move rigidly together.

All functions are pure: they return new objects and never mutate their
arguments.  Time is always supplied explicitly (``now``/timestamps are never
read from the wall clock), no randomness is used, and no function sleeps.
Unless a function says otherwise, float comparisons use a tolerance of 1e-9.
"""

import copy
import math

FLOAT_TOL = 1e-9

_FNV_OFFSET = 14695981039346656037
_FNV_PRIME = 1099511628211
_MASK64 = (1 << 64) - 1


class NetError(Exception):
    """Raised when a frame, window, input or correction violates the protocol."""


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value):
    return _is_int(value) or (isinstance(value, float) and math.isfinite(value))


def _require_finite_number(value, what):
    if not _is_finite_number(value):
        raise NetError(f"{what} must be a finite number, got {value!r}")
    return float(value)


def _validate_input(packet):
    if not isinstance(packet, dict):
        raise NetError(f"input must be a dict, got {packet!r}")
    seq = packet.get("seq")
    if not _is_int(seq) or seq < 0:
        raise NetError(f"input seq must be an integer >= 0, got {seq!r}")
    tick = packet.get("tick")
    if not _is_int(tick):
        raise NetError(f"input tick must be an integer, got {tick!r}")
    _require_finite_number(packet.get("turn", 0.0), "input turn")
    _require_finite_number(packet.get("dx"), "input dx")
    _require_finite_number(packet.get("dy"), "input dy")
    return packet


def _validate_state(state):
    if not isinstance(state, dict):
        raise NetError("state must be a dict")
    if not _is_int(state.get("tick")):
        raise NetError(f"state tick must be an integer, got {state.get('tick')!r}")
    root = state.get("root")
    if not isinstance(root, dict):
        raise NetError("state root must be a dict")
    for key in ("x", "y", "yaw"):
        _require_finite_number(root.get(key), f"root {key}")
    entities = state.get("entities")
    if not isinstance(entities, dict):
        raise NetError("state entities must be a dict")
    for name, node in entities.items():
        if not isinstance(node, dict):
            raise NetError(f"entity {name!r} must be a dict")
        _require_finite_number(node.get("x"), f"entity {name!r} x")
        _require_finite_number(node.get("y"), f"entity {name!r} y")


def predict(state, inputs, dt):
    """Apply ``inputs`` to a copy of ``state`` in client sequence order."""
    if not isinstance(dt, (int, float)) or isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
        raise NetError(f"dt must be a finite number > 0, got {dt!r}")
    _validate_state(state)
    if not isinstance(inputs, (list, tuple)):
        raise NetError("inputs must be a list")

    packets = [_validate_input(p) for p in inputs]
    seqs = [p["seq"] for p in packets]
    if len(set(seqs)) != len(seqs):
        raise NetError("duplicate seq within a single predict call")
    # Stable sort by seq: arrival order never changes the result.
    packets = sorted(packets, key=lambda p: p["seq"])

    result = copy.deepcopy(state)
    result["inputs"] = copy.deepcopy(list(state.get("inputs", [])))

    tick = result["tick"]
    yaw = float(result["root"]["yaw"])
    rx = float(result["root"]["x"])
    ry = float(result["root"]["y"])
    entity_pos = {
        name: (float(node["x"]), float(node["y"]))
        for name, node in result["entities"].items()
    }

    for packet in packets:
        ptick = packet["tick"]
        if ptick <= tick:
            raise NetError(
                f"input tick {ptick} regresses frame {tick}: ticks must be monotonic"
            )
        yaw += float(packet.get("turn", 0.0))
        dx = float(packet["dx"])
        dy = float(packet["dy"])
        wx = (dx * math.cos(yaw) - dy * math.sin(yaw)) * dt
        wy = (dx * math.sin(yaw) + dy * math.cos(yaw)) * dt
        rx += wx
        ry += wy
        entity_pos = {name: (x + wx, y + wy) for name, (x, y) in entity_pos.items()}
        tick = ptick

    result["tick"] = tick
    result["root"] = {"x": rx, "y": ry, "yaw": yaw}
    result["entities"] = {
        name: {"x": x, "y": y} for name, (x, y) in entity_pos.items()
    }
    return result


def reconcile(state, authoritative, ack_tick, history):
    """Roll back to ``ack_tick``, correct the root, and replay unacked inputs."""
    _validate_state(state)
    if not _is_int(ack_tick) or not 0 <= ack_tick <= state["tick"]:
        raise NetError(f"ack_tick must be an integer in [0, state tick], got {ack_tick!r}")
    if not isinstance(authoritative, dict):
        raise NetError("authoritative position must be a mapping")
    auth_x = _require_finite_number(authoritative.get("x"), "authoritative x")
    auth_y = _require_finite_number(authoritative.get("y"), "authoritative y")
    if not isinstance(history, (list, tuple)):
        raise NetError("history must be a list")

    # Validate the whole history before touching anything (atomicity).
    last_tick = None
    base = None
    for snapshot in history:
        if not isinstance(snapshot, dict) or not _is_int(snapshot.get("tick")):
            raise NetError(f"malformed history snapshot: {snapshot!r}")
        _validate_state(snapshot)
        stick = snapshot["tick"]
        if last_tick is not None and stick <= last_tick:
            raise NetError("history ticks must be strictly increasing")
        last_tick = stick
        if stick == ack_tick:
            if base is not None:
                raise NetError(f"duplicate history snapshot for tick {ack_tick}")
            base = snapshot
    if base is None:
        raise NetError(f"no history snapshot for ack_tick {ack_tick}")

    # Roll back root AND every child entity to the snapshot; only the root
    # x/y are corrected by the authoritative server position.
    rolled = copy.deepcopy(base)
    rolled["root"]["x"] = auth_x
    rolled["root"]["y"] = auth_y

    pending = state.get("inputs", [])
    if not isinstance(pending, (list, tuple)):
        raise NetError("state inputs must be a list")
    # Replay only unacknowledged inputs; confirmed ones (tick <= ack_tick)
    # must never be applied twice.
    replay = [p for p in pending if _validate_input(p)["tick"] > ack_tick]

    result = predict(rolled, replay, rolled["dt"])
    result["inputs"] = copy.deepcopy(list(pending))
    return result


def push_history(history, snapshot, window):
    """Append ``snapshot`` and keep only the newest ``window`` frames."""
    if not _is_int(window) or window < 1:
        raise NetError(f"window must be an integer >= 1, got {window!r}")
    if not isinstance(history, (list, tuple)):
        raise NetError("history must be a list")
    if not isinstance(snapshot, dict) or not _is_int(snapshot.get("tick")):
        raise NetError(f"snapshot tick must be an integer, got {snapshot!r}")

    last_tick = None
    for frame in history:
        if not isinstance(frame, dict) or not _is_int(frame.get("tick")):
            raise NetError(f"malformed history frame: {frame!r}")
        if last_tick is not None and frame["tick"] <= last_tick:
            raise NetError("history ticks must be strictly increasing")
        last_tick = frame["tick"]
    if last_tick is not None and snapshot["tick"] <= last_tick:
        raise NetError(
            f"snapshot tick {snapshot['tick']} must be greater than last history tick {last_tick}"
        )

    # Window is an exact frame count, never a duration: stalls cannot drop
    # the newest frames.
    combined = list(history) + [copy.deepcopy(snapshot)]
    return combined[-window:]


def _fnv_update(acc, data):
    for byte in data:
        acc ^= byte
        acc = (acc * _FNV_PRIME) & _MASK64
    return acc


def _hash_value(acc, value):
    if value is None:
        return _fnv_update(acc, b"n")
    if isinstance(value, bool):
        return _fnv_update(acc, b"T" if value else b"F")
    if _is_int(value):
        return _fnv_update(acc, b"i" + str(value).encode("ascii"))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NetError(f"cannot hash non-finite float {value!r}")
        return _fnv_update(acc, b"f" + repr(value).encode("ascii"))
    if isinstance(value, str):
        data = value.encode("utf-8")
        return _fnv_update(acc, b"s" + str(len(data)).encode("ascii") + b":" + data)
    if isinstance(value, (list, tuple)):
        acc = _fnv_update(acc, b"l" + str(len(value)).encode("ascii") + b"[")
        for item in value:
            acc = _hash_value(acc, item)
        return _fnv_update(acc, b"]")
    if isinstance(value, dict):
        items = []
        for key in value:
            if not isinstance(key, str):
                raise NetError(f"dict keys must be strings, got {key!r}")
            items.append(key)
        items.sort()
        acc = _fnv_update(acc, b"d" + str(len(items)).encode("ascii") + b"{")
        for key in items:
            acc = _hash_value(acc, key)
            acc = _hash_value(acc, value[key])
        return _fnv_update(acc, b"}")
    raise NetError(f"unsupported value type for checksum: {type(value).__name__}")


def checksum(state):
    """Return a deterministic, field-order-independent 64-bit hash."""
    return _hash_value(_FNV_OFFSET, state)


def interpolate(prev, nxt, alpha):
    """Linearly interpolate between two snapshots."""
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not math.isfinite(alpha):
        raise NetError(f"alpha must be a finite number, got {alpha!r}")
    alpha = float(alpha)
    if alpha < -1e-12 or alpha > 1.0 + 1e-12:
        raise NetError(f"alpha {alpha} outside [0, 1]")
    alpha = min(1.0, max(0.0, alpha))

    if not _is_int(prev.get("tick")) or not _is_int(nxt.get("tick")):
        raise NetError("snapshot ticks must be integers")
    if prev["tick"] > nxt["tick"]:
        raise NetError("prev tick must not exceed nxt tick")

    if alpha == 0.0:
        return copy.deepcopy(prev)
    if alpha == 1.0:
        return copy.deepcopy(nxt)

    result = copy.deepcopy(prev)
    for node_key in ("root",):
        before = prev.get(node_key)
        after = nxt.get(node_key)
        if isinstance(before, dict) and isinstance(after, dict):
            for axis in ("x", "y"):
                if _is_finite_number(before.get(axis)) and _is_finite_number(after.get(axis)):
                    result[node_key][axis] = before[axis] + (after[axis] - before[axis]) * alpha
    prev_entities = prev.get("entities", {})
    nxt_entities = nxt.get("entities", {})
    for name, before in prev_entities.items():
        after = nxt_entities.get(name)
        if not isinstance(before, dict) or not isinstance(after, dict):
            continue
        for axis in ("x", "y"):
            if _is_finite_number(before.get(axis)) and _is_finite_number(after.get(axis)):
                result["entities"][name][axis] = before[axis] + (after[axis] - before[axis]) * alpha
    return result


def dedupe_inputs(inputs, last_seq):
    """Drop confirmed and duplicate packets, then order inputs by sequence."""
    if not _is_int(last_seq) or last_seq < -1:
        raise NetError(f"last_seq must be an integer >= -1, got {last_seq!r}")
    if not isinstance(inputs, (list, tuple)):
        raise NetError("inputs must be a list")

    seen = set()
    kept = []
    for packet in inputs:
        if not isinstance(packet, dict):
            raise NetError(f"input must be a dict, got {packet!r}")
        seq = packet.get("seq")
        if not _is_int(seq) or seq < 0:
            raise NetError(f"input seq must be an integer >= 0, got {seq!r}")
        if seq <= last_seq:
            continue  # already confirmed: discard
        if seq in seen:
            continue  # duplicate anywhere in the batch, not only adjacent
        seen.add(seq)
        kept.append(packet)

    return sorted(kept, key=lambda p: p["seq"])


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def clock_offset(samples):
    """Estimate the server-minus-client clock offset robustly."""
    if not isinstance(samples, (list, tuple)) or len(samples) == 0:
        raise NetError("samples must be a non-empty list")

    offsets = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise NetError(f"sample must be a dict, got {sample!r}")
        send = _require_finite_number(sample.get("client_send"), "client_send")
        recv = _require_finite_number(sample.get("client_recv"), "client_recv")
        server = _require_finite_number(sample.get("server_time"), "server_time")
        if recv < send - FLOAT_TOL:
            raise NetError(f"client_recv {recv} before client_send {send}")
        offsets.append(server - (send + recv) / 2.0)

    center = _median(offsets)
    deviations = [abs(o - center) for o in offsets]
    mad = _median(deviations)

    if mad <= FLOAT_TOL:
        kept = [o for o in offsets if abs(o - center) <= FLOAT_TOL]
    else:
        limit = 3.0 * mad
        kept = [o for o in offsets if abs(o - center) <= limit + FLOAT_TOL]
    if not kept:
        raise NetError("all samples rejected as outliers")
    return float(_median(kept))


def apply_correction(state, target, max_step):
    """Move positions toward ``target`` by at most ``max_step`` per frame."""
    if (
        not isinstance(max_step, (int, float))
        or isinstance(max_step, bool)
        or not math.isfinite(max_step)
        or max_step <= 0
    ):
        raise NetError(f"max_step must be a finite number > 0, got {max_step!r}")
    _validate_state(state)
    if not isinstance(target, dict):
        raise NetError("target must be a mapping")

    result = copy.deepcopy(state)

    def move(node, goal, what):
        gx = _require_finite_number(goal.get("x"), f"{what} x")
        gy = _require_finite_number(goal.get("y"), f"{what} y")
        cx = float(node["x"])
        cy = float(node["y"])
        vx = gx - cx
        vy = gy - cy
        dist = math.hypot(vx, vy)
        if dist <= max_step + 1e-12:
            node["x"] = gx
            node["y"] = gy
        else:
            # Cap the correction: spread large fixes over frames, never
            # teleport through geometry.
            node["x"] = cx + vx / dist * max_step
            node["y"] = cy + vy / dist * max_step

    root_target = target.get("root")
    if root_target is not None:
        if not isinstance(root_target, dict):
            raise NetError("target root must be a mapping")
        move(result["root"], root_target, "target root")

    entity_targets = target.get("entities")
    if entity_targets is not None:
        if not isinstance(entity_targets, dict):
            raise NetError("target entities must be a mapping")
        for name, goal in entity_targets.items():
            if name not in result["entities"]:
                continue
            if not isinstance(goal, dict):
                raise NetError(f"target entity {name!r} must be a mapping")
            move(result["entities"][name], goal, f"target entity {name!r}")

    return result
