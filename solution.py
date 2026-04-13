from __future__ import annotations

from multiprocessing import shared_memory
from typing import TypeAlias

import numpy as np
import struct
import ctypes

__all__ = ["SharedBuffer"]

RingView: TypeAlias = tuple[memoryview, memoryview | None, int, bool]

# ---------------------------------------------------------------------------
# Memory layout of the shared block
# ---------------------------------------------------------------------------
# [ HEADER | PAYLOAD ]
#
# HEADER (fixed size, computed in __init__):
#   - write_pos      : uint64  (8 bytes)  absolute writer position
#   - reader_pos[i]  : int64   (8 bytes each) per-reader absolute position
#   - reader_active[i]: uint8  (1 byte each)  per-reader active flag
#
# PAYLOAD:
#   - `size` bytes of ring-buffer data
# ---------------------------------------------------------------------------


class SharedBuffer(shared_memory.SharedMemory):
        """
            A cross-process shared ring buffer supporting one writer and multiple readers.

                Memory layout inside the shared block:
                      [8 bytes write_pos][8*num_readers bytes reader_pos][num_readers bytes reader_active][payload]
                          """

    _NO_READER = -1

    # Offsets / sizes of header fields
    _WP_SIZE = 8          # write_pos: uint64
    _RP_SIZE = 8          # each reader_pos: int64
    _RA_SIZE = 1          # each reader_active: uint8

    def __init__(
                self,
                name: str,
                create: bool,
                size: int,
                num_readers: int,
                reader: int,
                cache_align: bool = False,
                cache_size: int = 64,
    ):
                # --- validate ---
                if size <= 0:
                                raise ValueError("size must be positive")
                            if num_readers <= 0:
                                            raise ValueError("num_readers must be positive")
                                        if reader != self._NO_READER and not (0 <= reader < num_readers):
                                                        raise ValueError(f"reader index {reader} out of range [0, {num_readers})")

        self._payload_size = size
        self._num_readers = num_readers
        self._reader_idx = reader  # _NO_READER means this is the writer instance

        # Compute header size
        # write_pos (8) + reader_pos * num_readers (8 each) + reader_active * num_readers (1 each)
        self._header_size = self._WP_SIZE + num_readers * self._RP_SIZE + num_readers * self._RA_SIZE

        total_size = self._header_size + size

        # Create or attach to shared memory
        super().__init__(name=name, create=create, size=total_size)

        # Build a memoryview over the whole block for easy slicing
        self._mem = memoryview(self.buf)

        # Offsets into the header
        self._wp_offset = 0                                        # write_pos
        self._rp_offset = self._WP_SIZE                           # reader_pos array
        self._ra_offset = self._rp_offset + num_readers * self._RP_SIZE  # reader_active array
        self._data_offset = self._header_size                     # payload start

        # On creation, zero-initialise the header
        if create:
                        for i in range(self._header_size):
                                            self._mem[i] = 0
                                        # Mark all readers as inactive initially
                                        for i in range(num_readers):
                                                            self._set_ra(i, 0)

    # ------------------------------------------------------------------
    # Low-level header accessors (all processes share these via shm)
    # ------------------------------------------------------------------

    def _get_wp(self) -> int:
                """Read write_pos as uint64."""
        return struct.unpack_from("Q", self.buf, self._wp_offset)[0]

    def _set_wp(self, val: int) -> None:
                """Write write_pos as uint64."""
        struct.pack_into("Q", self.buf, self._wp_offset, val)

    def _get_rp(self, idx: int) -> int:
                """Read reader_pos[idx] as int64."""
        offset = self._rp_offset + idx * self._RP_SIZE
        return struct.unpack_from("q", self.buf, offset)[0]

    def _set_rp(self, idx: int, val: int) -> None:
                """Write reader_pos[idx] as int64."""
        offset = self._rp_offset + idx * self._RP_SIZE
        struct.pack_into("q", self.buf, offset, val)

    def _get_ra(self, idx: int) -> int:
                """Read reader_active[idx] as uint8."""
        offset = self._ra_offset + idx * self._RA_SIZE
        return struct.unpack_from("B", self.buf, offset)[0]

    def _set_ra(self, idx: int, val: int) -> None:
                """Write reader_active[idx] as uint8."""
        offset = self._ra_offset + idx * self._RA_SIZE
        struct.pack_into("B", self.buf, offset, val)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def close(self) -> None:
                """Release local views and close this process's handle to the shared memory."""
        try:
                        self._mem.release()
except Exception:
            pass
        try:
                        super().close()
except Exception:
            pass

    def __enter__(self) -> "SharedBuffer":
                """Mark reader active on entry; writer instances just return self."""
        if self._reader_idx != self._NO_READER:
                        self.set_reader_active(True)
        return self

    def __exit__(self, *_):
                """Mark reader inactive on exit, then close."""
        if self._reader_idx != self._NO_READER:
                        self.set_reader_active(False)
        self.close()

    # --- position helpers ---

    def int_to_pos(self, value: int) -> int:
                """Convert an absolute position counter to an offset inside the payload area."""
        return value % self._payload_size

    def get_write_pos(self) -> int:
                """Return the current absolute writer position."""
        return self._get_wp()

    def update_write_pos(self, new_writer_pos: int) -> None:
                """Store the writer's absolute write position in shared state."""
        self._set_wp(new_writer_pos)

    def inc_writer_pos(self, inc_amount: int) -> None:
                """Advance the writer's absolute position by inc_amount bytes."""
        self._set_wp(self._get_wp() + inc_amount)

    def update_reader_pos(self, new_reader_pos: int) -> None:
                """Store this reader's absolute read position in shared state."""
        if self._reader_idx == self._NO_READER:
                        raise RuntimeError("update_reader_pos called on a writer instance")
        self._set_rp(self._reader_idx, new_reader_pos)

    def inc_reader_pos(self, inc_amount: int) -> None:
                """Advance this reader's absolute position by inc_amount bytes."""
        if self._reader_idx == self._NO_READER:
                        raise RuntimeError("inc_reader_pos called on a writer instance")
        self._set_rp(self._reader_idx, self._get_rp(self._reader_idx) + inc_amount)

    def set_reader_active(self, active: bool) -> None:
                """Mark this reader as active or inactive in shared state."""
        if self._reader_idx == self._NO_READER:
                        raise RuntimeError("set_reader_active called on a writer instance")
        self._set_ra(self._reader_idx, 1 if active else 0)

    def is_reader_active(self) -> bool:
                """Return whether this reader is currently marked active."""
        if self._reader_idx == self._NO_READER:
                        raise RuntimeError("is_reader_active called on a writer instance")
        return self._get_ra(self._reader_idx) == 1

    # --- capacity & pressure ---

    def compute_max_amount_writable(self, force_rescan: bool = False) -> int:
                """
                        Return how many bytes the writer can safely write right now.

                                The writer cannot overwrite data that an active reader hasn't consumed yet.
                                        We find the slowest active reader and limit writes so we don't lap it.
                                                """
        wp = self._get_wp()
        min_rp = None

        for i in range(self._num_readers):
                        if self._get_ra(i) == 1:  # active reader
                                            rp = self._get_rp(i)
                                            if min_rp is None or rp < min_rp:
                                                                    min_rp = rp

                                    if min_rp is None:
                                                    # No active readers — writer can use the full payload area
                                                    return self._payload_size

        # Bytes already in use = wp - min_rp
        in_use = wp - min_rp
        return max(0, self._payload_size - in_use)

    def calculate_pressure(self) -> int:
                """Return current writer pressure as an integer percentage (0-100)."""
        wp = self._get_wp()
        min_rp = None

        for i in range(self._num_readers):
                        if self._get_ra(i) == 1:
                                            rp = self._get_rp(i)
                                            if min_rp is None or rp < min_rp:
                                                                    min_rp = rp

                                    if min_rp is None:
                                                    return 0

        in_use = wp - min_rp
        return int(min(100, (in_use * 100) // self._payload_size))

    def jump_to_writer(self) -> None:
                """Move this reader directly to the current writer position (skip stale data)."""
        if self._reader_idx == self._NO_READER:
                        raise RuntimeError("jump_to_writer called on a writer instance")
        self._set_rp(self._reader_idx, self._get_wp())

    # --- memory views ---

    def expose_writer_mem_view(self, size: int) -> RingView:
                """
                        Return a writable view tuple for up to `size` bytes.
                                Returns (mv1, mv2_or_None, actual_size, is_split).
                                        """
        actual_size = min(size, self.compute_max_amount_writable())

        if actual_size == 0:
                        empty = memoryview(self.buf)[self._data_offset:self._data_offset]
            return (empty, None, 0, False)

        wp = self._get_wp()
        start = self._data_offset + self.int_to_pos(wp)
        end = start + actual_size

        if end <= self._data_offset + self._payload_size:
                        # No wrap
                        mv1 = memoryview(self.buf)[start:end]
            return (mv1, None, actual_size, False)
else:
            # Wraps around
                first_len = (self._data_offset + self._payload_size) - start
            second_len = actual_size - first_len
            mv1 = memoryview(self.buf)[start:start + first_len]
            mv2 = memoryview(self.buf)[self._data_offset:self._data_offset + second_len]
            return (mv1, mv2, actual_size, True)

    def expose_reader_mem_view(self, size: int) -> RingView:
                """
                        Return a readable view tuple for up to `size` bytes.
                                Returns (mv1, mv2_or_None, actual_size, is_split).
                                        """
        if self._reader_idx == self._NO_READER:
                        raise RuntimeError("expose_reader_mem_view called on a writer instance")

        rp = self._get_rp(self._reader_idx)
        wp = self._get_wp()
        available = wp - rp
        actual_size = min(size, available)

        if actual_size <= 0:
                        empty = memoryview(self.buf)[self._data_offset:self._data_offset]
            return (empty, None, 0, False)

        start = self._data_offset + self.int_to_pos(rp)
        end = start + actual_size

        if end <= self._data_offset + self._payload_size:
                        mv1 = memoryview(self.buf)[start:end]
            return (mv1, None, actual_size, False)
else:
            first_len = (self._data_offset + self._payload_size) - start
            second_len = actual_size - first_len
            mv1 = memoryview(self.buf)[start:start + first_len]
            mv2 = memoryview(self.buf)[self._data_offset:self._data_offset + second_len]
            return (mv1, mv2, actual_size, True)

    # --- copy helpers ---

    def simple_write(self, writer_mem_view: RingView, src: object) -> None:
                """Copy bytes from src into the exposed writer view(s)."""
        mv1, mv2, actual_size, split = writer_mem_view
        src_bytes = memoryview(src).cast("B") if not isinstance(src, memoryview) else src.cast("B")

        if not split:
                        n = min(len(mv1), len(src_bytes))
            mv1[:n] = src_bytes[:n]
else:
            n1 = len(mv1)
            n2 = len(mv2)
            mv1[:n1] = src_bytes[:n1]
            mv2[:n2] = src_bytes[n1:n1 + n2]

    def simple_read(self, reader_mem_view: RingView, dst: object) -> None:
                """Copy bytes from the exposed reader view(s) into dst."""
        mv1, mv2, actual_size, split = reader_mem_view
        dst_bytes = memoryview(dst).cast("B") if not isinstance(dst, memoryview) else dst.cast("B")

        if not split:
                        n = min(len(mv1), len(dst_bytes))
            dst_bytes[:n] = mv1[:n]
else:
            n1 = min(len(mv1), len(dst_bytes))
            dst_bytes[:n1] = mv1[:n1]
            remaining = len(dst_bytes) - n1
            if remaining > 0 and mv2 is not None:
                                n2 = min(len(mv2), remaining)
                                dst_bytes[n1:n1 + n2] = mv2[:n2]

    # --- high-level numpy I/O ---

    def write_array(self, arr: np.ndarray) -> int:
                """
                        Write a NumPy array's raw bytes into the shared buffer.
                                Returns number of bytes written, or 0 if the full array doesn't fit.
                                        """
        data = arr.tobytes()
        nbytes = len(data)

        if nbytes > self.compute_max_amount_writable():
                        return 0

        view = self.expose_writer_mem_view(nbytes)
        self.simple_write(view, data)
        self.inc_writer_pos(view[2])
        return view[2]

    def read_array(self, nbytes: int, dtype: np.dtype) -> np.ndarray:
                """
                        Read nbytes from the shared buffer and interpret them as dtype.
                                Returns empty array if not enough data is available.
                                        """
        if self._reader_idx == self._NO_READER:
                        raise RuntimeError("read_array called on a writer instance")

        rp = self._get_rp(self._reader_idx)
        wp = self._get_wp()

        if wp - rp < nbytes:
                        return np.array([], dtype=dtype)

        view = self.expose_reader_mem_view(nbytes)
        buf = bytearray(nbytes)
        self.simple_read(view, buf)
        self.inc_reader_pos(view[2])
        return np.frombuffer(buf, dtype=dtype)
