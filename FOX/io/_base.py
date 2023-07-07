"""Misc IO-related utilities."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Callable
from typing import TYPE_CHECKING

if sys.version_info >= (3, 9):
    from collections.abc import Iterator
else:
    from typing import Iterator

if TYPE_CHECKING:
    from typing_extensions import Self

__all__ = ["FileIter"]


class FileIter(Iterator[str]):
    """An iterator specialized in iterating through opened files."""

    __slots__ = ("__weakref__", "_enumerator", "_name", "_index")

    _name: str
    _index: None | int

    @property
    def index(self) -> None | int:
        """Get the index within the current iterator."""
        return self._index

    @property
    def name(self) -> str:
        """Get the name of the iterator."""
        return self._name

    def __init__(
        self,
        iterable: Iterable[str],
        start: int = 1,
        stripper: None | Callable[[str], str] = str.strip,
    ) -> None:
        if stripper is None:
            self._enumerator: Iterator[tuple[int, str]] = enumerate(iterable, start)
        else:
            self._enumerator = (
                (i, j) for i, _j in enumerate(iterable, start) if (j := stripper(_j))
            )
        self._name = getattr(iterable, "name", "<unknown>")
        self._index = None

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> str:
        self._index, value = next(self._enumerator)
        return value

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self._name!r} index={self._index!r}>"
