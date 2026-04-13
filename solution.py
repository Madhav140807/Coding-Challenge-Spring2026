from __future__ import annotations

from multiprocessing import shared_memory
from typing import TypeAlias

import numpy as np
import struct

__all__ = ["SharedBuffer"]

RingView: TypeAlias = tuple[memoryview, memoryview | None, int, bool]


class SharedBuffer(shared_memory.SharedMemory):
    _NO_READER = -1
    _WP_SIZE = 8
    _RP_SIZE = 8
    _RA_SIZE = 1

    def __init__(self, name, create, size, num_readers, reader, cache_align=False, cache_size=64):
        if size <= 0:
            raise ValueError("size must be positive")
        if num_readers <= 0:
            raise ValueError("num_readers must be positive")
        if reader != self._NO_READER and not (0 <= reader < num_readers):
            raise ValueError(f"reader index {reader} out of range [0, {num_readers})")
        if cache_align and (cache_size <= 0 or (cache_size & (cache_size - 1)) != 0):
            raise ValueError(f"cache_size must be a power of two when cache_align=True, got {cache_size}")
        self.buffer_size = size
        self.num_readers = num_readers
        self._payload_size = size
        self._reader_idx = reader
        self._header_size = self._WP_SIZE + num_readers * self._RP_SIZE + num_readers * self._RA_SIZE
        total_size = self._header_size + size
        super().__init__(name=name, create=create, size=total_size)
        self.buffer = memoryview(self.buf)
        self._wp_offset = 0
        self._rp_offset = self._WP_SIZE
        self._ra_offset = self._rp_offset + num_readers * self._RP_SIZE
        self._data_offset = self._header_size
        if create:
            for i in range(self._header_size):
                self.buf[i] = 0

    def _get_wp(self):
        return struct.unpack_from("Q", self.buf, self._wp_offset)[0]

    def _set_wp(self, val):
        struct.pack_into("Q", self.buf, self._wp_offset, val)

    def _get_rp(self, idx):
        return struct.unpack_from("q", self.buf, self._rp_offset + idx * self._RP_SIZE)[0]

    def _set_rp(self, idx, val):
        struct.pack_into("q", self.buf, self._rp_offset + idx * self._RP_SIZE, val)

    def _get_ra(self, idx):
        return struct.unpack_from("B", self.buf, self._ra_offset + idx * self._RA_SIZE)[0]

    def _set_ra(self, idx, val):
        struct.pack_into("B", self.buf, self._ra_offset + idx * self._RA_SIZE, val)

    def close(self):
        try:
            self.buffer.release()
        except Exception:
            pass
        try:
            super().close()
        except Exception:
            pass

    def __enter__(self):
        if self._reader_idx != self._NO_READER:
            self.set_reader_active(True)
        return self

    def __exit__(self, *_):
        if self._reader_idx != self._NO_READER:
            self.set_reader_active(False)
        self.close()

    def int_to_pos(self, value):
        return value % self._payload_size

    def get_write_pos(self):
        return self._get_wp()

    def update_write_pos(self, new_writer_pos):
        self._set_wp(new_writer_pos)

    def inc_writer_pos(self, inc_amount):
        self._set_wp(self._get_wp() + inc_amount)

    def update_reader_pos(self, new_reader_pos):
        if self._reader_idx == self._NO_READER:
            raise RuntimeError("update_reader_pos called on a writer instance")
        self._set_rp(self._reader_idx, new_reader_pos)

    def inc_reader_pos(self, inc_amount):
        if self._reader_idx == self._NO_READER:
            raise RuntimeError("inc_reader_pos called on a writer instance")
        self._set_rp(self._reader_idx, self._get_rp(self._reader_idx) + inc_amount)

    def set_reader_active(self, active):
        if self._reader_idx == self._NO_READER:
            raise RuntimeError("set_reader_active called on a writer instance")
        self._set_ra(self._reader_idx, 1 if active else 0)

    def is_reader_active(self):
        if self._reader_idx == self._NO_READER:
            raise RuntimeError("is_reader_active called on a writer instance")
        return self._get_ra(self._reader_idx) == 1

    def _slowest_active_reader_pos(self):
        min_rp = None
        for i in range(self.num_readers):
            if self._get_ra(i) == 1:
                rp = self._get_rp(i)
                if min_rp is None or rp < min_rp:
                    min_rp = rp
        return min_rp

    def compute_max_amount_writable(self, force_rescan=False):
        wp = self._get_wp()
        min_rp = self._slowest_active_reader_pos()
        if min_rp is None:
            return self._payload_size
        in_use = wp - min_rp
        return max(0, self._payload_size - in_use)

    def calculate_pressure(self):
        wp = self._get_wp()
        min_rp = self._slowest_active_reader_pos()
        if min_rp is None:
            return 0
        in_use = wp - min_rp
        return int(min(100, (in_use * 100) // self._payload_size))

    def jump_to_writer(self):
        if self._reader_idx == self._NO_READER:
            raise RuntimeError("jump_to_writer called on a writer instance")
        self._set_rp(self._reader_idx, self._get_wp())

    def expose_writer_mem_view(self, size):
        actual_size = min(size, self.compute_max_amount_writable())
        if actual_size == 0:
            empty = self.buffer[self._data_offset:self._data_offset]
            return (empty, None, 0, False)
        wp = self._get_wp()
        start = self._data_offset + self.int_to_pos(wp)
        buf_end = self._data_offset + self._payload_size
        if start + actual_size <= buf_end:
            mv1 = self.buffer[start:start + actual_size]
            return (mv1, None, actual_size, False)
        else:
            first_len = buf_end - start
            second_len = actual_size - first_len
            mv1 = self.buffer[start:start + first_len]
            mv2 = self.buffer[self._data_offset:self._data_offset + second_len]
            return (mv1, mv2, actual_size, True)

    def expose_reader_mem_view(self, size):
        if self._reader_idx == self._NO_READER:
            raise RuntimeError("expose_reader_mem_view called on a writer instance")
        rp = self._get_rp(self._reader_idx)
        wp = self._get_wp()
        available = wp - rp
        if available > self._payload_size:
            self._set_rp(self._reader_idx, self._get_wp())
            empty = self.buffer[self._data_offset:self._data_offset]
            return (empty, None, 0, False)
        actual_size = min(size, max(0, available))
        if actual_size == 0:
            empty = self.buffer[self._data_offset:self._data_offset]
            return (empty, None, 0, False)
        start = self._data_offset + self.int_to_pos(rp)
        buf_end = self._data_offset + self._payload_size
        if start + actual_size <= buf_end:
            mv1 = self.buffer[start:start + actual_size]
            return (mv1, None, actual_size, False)
        else:
            first_len = buf_end - start
            second_len = actual_size - first_len
            mv1 = self.buffer[start:start + first_len]
            mv2 = self.buffer[self._data_offset:self._data_offset + second_len]
            return (mv1, mv2, actual_size, True)

    def simple_write(self, writer_mem_view, src):
        mv1, mv2, actual_size, split = writer_mem_view
        src_mv = memoryview(src).cast("B") if not isinstance(src, memoryview) else src.cast("B")
        if not split:
            n = min(len(mv1), len(src_mv))
            mv1[:n] = src_mv[:n]
        else:
            n1 = len(mv1)
            mv1[:n1] = src_mv[:n1]
            if mv2 is not None:
                n2 = len(mv2)
                mv2[:n2] = src_mv[n1:n1 + n2]

    def simple_read(self, reader_mem_view, dst):
        mv1, mv2, actual_size, split = reader_mem_view
        dst_mv = memoryview(dst).cast("B") if not isinstance(dst, memoryview) else dst.cast("B")
        if not split:
            n = min(len(mv1), len(dst_mv))
            dst_mv[:n] = mv1[:n]
        else:
            n1 = min(len(mv1), len(dst_mv))
            dst_mv[:n1] = mv1[:n1]
            remaining = len(dst_mv) - n1
            if remaining > 0 and mv2 is not None:
                n2 = min(len(mv2), remaining)
                dst_mv[n1:n1 + n2] = mv2[:n2]

    def write_array(self, arr):
        data = arr.tobytes()
        nbytes = len(data)
        if nbytes > self.compute_max_amount_writable():
            return 0
        view = self.expose_writer_mem_view(nbytes)
        self.simple_write(view, data)
        self.inc_writer_pos(view[2])
        return view[2]

    def read_array(self, nbytes, dtype):
        if self._reader_idx == self._NO_READER:
            raise RuntimeError("read_array called on a writer instance")
        rp = self._get_rp(self._reader_idx)
        wp = self._get_wp()
        if wp - rp < nbytes:
            return np.array([], dtype=dtype)
        view = self.expose_reader_mem_view(nbytes)
        if view[2] < nbytes:
            return np.array([], dtype=dtype)
        buf = bytearray(nbytes)
        self.simple_read(view, buf)
        self.inc_reader_pos(view[2])
        return np.frombuffer(buf, dtype=dtype)
