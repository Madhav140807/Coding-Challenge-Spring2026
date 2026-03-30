from __future__ import annotations

from multiprocessing import shared_memory
from typing import TypeAlias

import numpy as np


__all__ = ["SharedBuffer"]

RingView: TypeAlias = tuple[memoryview, memoryview | None, int, bool]


class SharedBuffer(shared_memory.SharedMemory):
    """
    Applicant template.

    Replace every method body with your own implementation while preserving the
    public API used by the official tests.
    """

    _NO_READER = -1

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
        raise NotImplementedError("TODO: implement SharedBuffer.__init__")

    def close(self) -> None:
        try:
            super().close()
        except Exception:
            pass

    def __enter__(self) -> "SharedBuffer":
        return self

    def __exit__(self, *_):
        self.close()

    def calculate_pressure(self) -> int:
        raise NotImplementedError("TODO: implement SharedBuffer.calculate_pressure")

    def int_to_pos(self, value: int) -> int:
        raise NotImplementedError("TODO: implement SharedBuffer.int_to_pos")

    def update_reader_pos(self, new_reader_pos: int) -> None:
        raise NotImplementedError("TODO: implement SharedBuffer.update_reader_pos")

    def set_reader_active(self, active: bool) -> None:
        raise NotImplementedError("TODO: implement SharedBuffer.set_reader_active")

    def is_reader_active(self) -> bool:
        raise NotImplementedError("TODO: implement SharedBuffer.is_reader_active")

    def update_write_pos(self, new_writer_pos: int) -> None:
        raise NotImplementedError("TODO: implement SharedBuffer.update_write_pos")

    def inc_writer_pos(self, inc_amount: int) -> None:
        raise NotImplementedError("TODO: implement SharedBuffer.inc_writer_pos")

    def inc_reader_pos(self, inc_amount: int) -> None:
        raise NotImplementedError("TODO: implement SharedBuffer.inc_reader_pos")

    def get_write_pos(self) -> int:
        raise NotImplementedError("TODO: implement SharedBuffer.get_write_pos")

    def compute_max_amount_writable(self, force_rescan: bool = False) -> int:
        raise NotImplementedError("TODO: implement SharedBuffer.compute_max_amount_writable")

    def jump_to_writer(self) -> None:
        raise NotImplementedError("TODO: implement SharedBuffer.jump_to_writer")

    def expose_writer_mem_view(self, size: int) -> RingView:
        raise NotImplementedError("TODO: implement SharedBuffer.expose_writer_mem_view")

    def expose_reader_mem_view(self, size: int) -> RingView:
        raise NotImplementedError("TODO: implement SharedBuffer.expose_reader_mem_view")

    def simple_write(self, writer_mem_view: RingView, src: object) -> None:
        raise NotImplementedError("TODO: implement SharedBuffer.simple_write")

    def simple_read(self, reader_mem_view: RingView, dst: object) -> None:
        raise NotImplementedError("TODO: implement SharedBuffer.simple_read")

    def write_array(self, arr: np.ndarray) -> int:
        raise NotImplementedError("TODO: implement SharedBuffer.write_array")

    def read_array(self, nbytes: int, dtype: np.dtype) -> np.ndarray:
        raise NotImplementedError("TODO: implement SharedBuffer.read_array")
