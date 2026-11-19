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

FLOAT_TOL = 1e-9


class NetError(Exception):
    """Raised when a frame, window, input or correction violates the protocol."""


def predict(state, inputs, dt):
    """Apply ``inputs`` to a copy of ``state`` in client sequence order.

    Contract:
    * ``dt`` must be a finite number greater than 0; otherwise ``NetError``.
    * Each input ``seq`` is an integer >= 0 and unique within the call.  Inputs
      are stably sorted by ascending ``seq`` before any is applied, so packet
      arrival order never changes the result.
    * Frame numbers are monotonic.  ``state["tick"]`` must be an integer.  Every
      input ``tick`` must be an integer that is strictly greater than the
      current frame and not smaller than the previous input's tick; a tick
      regression raises ``NetError``.
    * Each input's ``turn`` (default 0.0) is added to ``root["yaw"]`` first.
      The local velocity ``(dx, dy)`` is then rotated by the new yaw, scaled by
      ``dt`` and added to the root and all child entities.  Positions use a
      float tolerance of 1e-12 for validation.
    * The returned state's ``tick`` is the last applied input's tick.  Its
      ``inputs`` list is preserved unchanged; the caller owns the pending
      queue and reconciliation decides what to replay.
    """
    raise NotImplementedError("predict is not implemented")


def reconcile(state, authoritative, ack_tick, history):
    """Roll back to ``ack_tick``, correct the root, and replay unacked inputs.

    Contract:
    * ``ack_tick`` is an integer with ``0 <= ack_tick <= state["tick"]``.
    * ``authoritative`` is the server's root position at that tick: a mapping
      with finite numeric ``x`` and ``y``.  Those two values replace the root
      x/y from the rollback snapshot.  Root yaw and every child entity come
      from the history snapshot, not from the current live state.
    * ``history`` is an oldest-to-newest list of full state snapshots whose
      integer ticks are strictly increasing.  It contains exactly one snapshot
      with ``tick == ack_tick``; that snapshot is the base for the rollback.
      A missing/duplicate/malformed snapshot raises ``NetError``.
    * Replay only inputs with ``input["tick"] > ack_tick``, ordered by
      ascending ``seq`` under the same rules as :func:`predict`.  Inputs at or
      below ``ack_tick`` are already confirmed and must not be applied again.
      The returned tick is ``ack_tick`` when nothing is replayed, otherwise the
      last replayed input's tick.
    * The operation is atomic: any validation or replay error raises
      ``NetError`` and leaves both ``state`` and ``history`` byte-for-byte
      unchanged.  The returned state is a new object and history is not
      appended to by this function.
    """
    raise NotImplementedError("reconcile is not implemented")


def push_history(history, snapshot, window):
    """Append ``snapshot`` and keep only the newest ``window`` frames.

    ``window`` is an integer frame count, never a duration.  The result holds
    exactly ``min(window, len(history) + 1)`` snapshots: the newest frame is
    always kept and the oldest frames are discarded first, regardless of
    ``dt`` or irregular tick spacing.  ``window`` must be >= 1.

    ``history`` ticks must be strictly increasing and ``snapshot["tick"]`` must
    be an integer greater than the last history tick; otherwise ``NetError``.
    Returns a new list and does not mutate ``history`` or ``snapshot``.  There
    is no tolerance on the window size: it is an exact integer count.
    """
    raise NotImplementedError("push_history is not implemented")


def checksum(state):
    """Return a deterministic, field-order-independent 64-bit hash.

    Dictionary keys are sorted recursively before hashing, so two states that
    differ only in field insertion order have the same checksum.  List order is
    significant and is preserved.  Supported values are None, bool, int,
    float, str, list and dict; anything else raises ``NetError``.

    The canonical encoding uses decimal integers, ``repr`` for finite floats,
    UTF-8 strings and a 64-bit FNV-1a accumulator.  The returned value is an
    integer in ``[0, 2**64)``.  Numeric values are hashed exactly; callers that
    need tolerance must normalise them to within 1e-9 before hashing.
    """
    raise NotImplementedError("checksum is not implemented")


def interpolate(prev, nxt, alpha):
    """Linearly interpolate between two snapshots.

    ``alpha`` must be a finite float in the inclusive range ``[0, 1]``; a value
    below 0 or above 1 (beyond a 1e-12 tolerance) raises ``NetError``.  At
    ``alpha == 0`` return a deep copy equal to ``prev``; at ``alpha == 1``
    return a deep copy equal to ``nxt``.

    Intermediate values interpolate the numeric ``x``/``y`` fields of ``root``
    and ``entities``.  ``prev["tick"]`` must not exceed ``nxt["tick"]``
    (otherwise ``NetError``); frame numbers are integers and never decrease.
    For ``0 < alpha < 1`` the result carries ``prev["tick"]``.  Other fields
    are copied from ``prev``; inputs must not be mutated.
    """
    raise NotImplementedError("interpolate is not implemented")


def dedupe_inputs(inputs, last_seq):
    """Drop confirmed and duplicate packets, then order inputs by sequence.

    ``last_seq`` is the highest already-confirmed sequence number (integer >=
    -1).  Every input whose ``seq`` is <= ``last_seq`` is discarded.  Duplicate
    ``seq`` values anywhere in the list, not only adjacent ones, collapse to
    the first occurrence in the original list.  The result is sorted by
    ascending ``seq``, so returned sequence numbers are strictly increasing
    and strictly greater than ``last_seq``.

    Each input must be a dict with an integer ``seq`` >= 0; malformed input
    raises ``NetError``.  The function is pure and does not reorder or mutate
    the input list in place.  Equality of sequence numbers is exact; no float
    tolerance applies.
    """
    raise NotImplementedError("dedupe_inputs is not implemented")


def clock_offset(samples):
    """Estimate the server-minus-client clock offset robustly.

    Each sample is a mapping with finite numbers::

        {"client_send": float, "client_recv": float, "server_time": float}

    where ``client_recv >= client_send``.  The per-sample offset is::

        server_time - (client_send + client_recv) / 2

    The estimate is the median of those offsets after outlier removal.  With
    ``center = median(offsets)`` and ``mad = median(|offset - center|)``, drop
    every sample whose absolute deviation exceeds ``3 * mad`` (float tolerance
    1e-9).  If ``mad == 0``, keep only samples exactly at the median.  Return
    the median of the remaining offsets as a float; a single bad sample must
    not pull the result away from the cluster.  Empty or malformed input raises
    ``NetError``.
    """
    raise NotImplementedError("clock_offset is not implemented")


def apply_correction(state, target, max_step):
    """Move positions toward ``target`` by at most ``max_step`` per frame.

    ``target`` is a partial snapshot: it may contain ``root`` and ``entities``
    mappings with finite numeric x/y targets.  For every provided node compute
    the Euclidean correction vector.  If its length is <= ``max_step`` (within
    1e-12) snap to the target; otherwise move exactly ``max_step`` along the
    unit vector toward it.  No node may travel farther than ``max_step`` in one
    call, so a large correction is spread over several frames and never
    teleports through a wall.

    ``max_step`` must be a finite number greater than 0, otherwise
    ``NetError``.  Tick, root yaw, inputs and nodes without a target are
    preserved.  Returns a new state and does not mutate ``state`` or
    ``target``.  Repeating the call converges to the target.
    """
    raise NotImplementedError("apply_correction is not implemented")
