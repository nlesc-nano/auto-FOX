"""An abstract container for reading and writing files.

Index
-----
.. currentmodule:: FOX.io.file_container
.. autosummary::
    AbstractFileContainer

API
---
.. autoclass:: AbstractFileContainer
    :members:
    :private-members:
    :special-members:

"""

import os
import io
import abc
import functools
from codecs import iterdecode, encode
from typing import (Dict, Optional, Any, Iterable, Iterator, Union,
                    AnyStr, Callable, ContextManager, TypeVar)
from collections.abc import Container

__all__ = ['AbstractFileContainer']

T = TypeVar('T')
CT = TypeVar('CT', bound=Callable)


class AbstractFileContainer(abc.ABC, Container):
    """An abstract container for reading and writing files.

    Two public methods are defined within this class:

    * :meth:`AbstractFileContainer.read`: Construct a new instance from this object's class by
        reading the content to a file or file object.
        How the content of the to-be read file is parsed has to be defined in the
        :meth:`AbstractFileContainer._read_iterate` abstract method.
    * :meth:`AbstractFileContainer.write`: Write the content of this instance to an opened
        file or file object.
        How the content of the to-be exported class instance is parsed has to be defined in
        the :meth:`AbstractFileContainer._write_iterate`

    The opening, closing and en-/decoding of files is handled by two above-mentioned methods;
    the parsing
    * :meth:`AbstractFileContainer._read_iterate`
    * :meth:`AbstractFileContainer._write_iterate`

    """

    def __contains__(self, value: Any) -> bool:
        """Check if this instance contains **value**."""
        return value in vars(self).values()

    @classmethod
    def read(cls, filename: Union[AnyStr, os.PathLike, Iterable[AnyStr]],
             encoding: Optional[str] = None, **kwargs: Any) -> 'AbstractFileContainer':
        r"""Construct a new instance from this object's class by reading the content of **filename**.

        .. _`file object`: https://docs.python.org/3/glossary.html#term-file-object

        Parameters
        ----------
        filename : :class:`str`, :class:`bytes`, :class:`os.PathLike` or a file object
            The path+filename or a `file object`_ of the to-be read .psf file.
            In practice, any iterable can substitute the role of file object
            as long iteration returns either strings or bytes (see **encoding**).

        encoding : :class:`str`, optional
            Encoding used to decode the input (*e.g.* ``"utf-8"``).
            Only relevant when a file object is supplied to **filename** and
            the datastream is *not* in text mode.

        \**kwargs : :data:`Any<typing.Any>`
            Optional keyword arguments that will be passed to both
            :meth:`AbstractFileContainer._read_iterate` and
            :meth:`AbstractFileContainer._read_postprocess`.

        See Also
        --------
        :meth:`AbstractFileContainer._read_iterate`
            An abstract method for parsing the opened file in :meth:`AbstractFileContainer.read`.

        :meth:`AbstractFileContainer._read_postprocess`
            Post processing the class instance created by :meth:`AbstractFileContainer.read`.

        """  # noqa
        if isinstance(filename, (bytes, str, os.PathLike)):  # A file
            context_manager = open
        elif isinstance(filename, Iterable):  # A file object
            context_manager = NullContext
        else:  # Neither file nor file object
            raise TypeError(f"The 'filename' parameter is of invalid type: {repr(type(filename))}")

        with context_manager(filename, 'r') as f:
            iterator = iter(f) if encoding is None else iterdecode(f, encoding)
            class_dict = cls._read_iterate(iterator, **kwargs)

        ret = cls(**class_dict)
        ret._read_postprocess(filename, encoding, **kwargs)
        return ret

    @classmethod
    @abc.abstractmethod
    def _read_iterate(cls, iterator: Iterator[str], **kwargs) -> Dict[str, Any]:
        r"""An abstract method for parsing the opened file in :class:`.read`.

        Parameters
        ----------
        iterator : :class:`Iterator<collections.abc.Iterator>` [:class:`str`]
            An iterator that returns :class:`str` instances upon iteration.

        Returns
        -------
        :class:`dict` [:class:`str`, :data:`Any<typing.Any>`]
            A dictionary with keyword arguments for a new instance of this objects' class.

        \**kwargs : :data:`Any<typing.Any>`
            Optional keyword arguments.

        See Also
        --------
        :meth:`.read`
            The main method for reading files.

        """  # noqa
        raise NotImplementedError('Trying to call an abstract method')

    def _read_postprocess(self, filename: Union[AnyStr, os.PathLike, Iterable[AnyStr]],
                          encoding: Optional[str] = None, **kwargs) -> None:
        r"""Post processing the class instance created by :meth:`.read`.

        Parameters
        ----------
        filename : :class:`str`, :class:`bytes`, :class:`os.PathLike` or a file object
            The path+filename or a `file object`_ of the to-be read .psf file.
            In practice, any iterable can substitute the role of file object
            as long iteration returns either strings or bytes (see **encoding**).

        encoding : :class:`str`, optional
            Encoding used to decode the input (*e.g.* ``"utf-8"``).
            Only relevant when a file object is supplied to **filename** and
            the datastream is *not* in text mode.

        \**kwargs : :data:`Any<typing.Any>`
            Optional keyword arguments that will be passed to both
            :meth:`AbstractFileContainer._read_iterate` and
            :meth:`AbstractFileContainer._read_postprocess`.

        See Also
        --------
        :meth:`AbstractFileContainer.read`
            The main method for reading files.

        """  # noqa
        pass

    """########################### methods for writing files. ##############################"""

    def write(self, filename: Union[AnyStr, os.PathLike, io.IOBase],
              encoding: Optional[str] = None, **kwargs) -> None:
        r"""Write the content of this instance to **filename**.

        .. _`file object`: https://docs.python.org/3/glossary.html#term-file-object

        Parameters
        ----------
        filename : :class:`str`, :class:`bytes`, :class:`os.PathLike` or a file object
            The path+filename or a `file object`_ of the to-be read .psf file.
            Contrary to :meth:`._read_postprocess`, file objects can *not*
            be substituted for generic iterables.

        encoding : :class:`str`, optional
            Encoding used to decode the input (*e.g.* ``"utf-8"``).
            Only relevant when a file object is supplied to **filename** and
            the datastream is *not* in text mode.

        \**kwargs : :data:`Any<typing.Any>`
            Optional keyword arguments that will be passed to :meth:`._write_iterate`.

        See Also
        --------
        :meth:`AbstractFileContainer._write_iterate`
            Write the content of this instance to an opened datastream.

        :meth:`AbstractFileContainer._get_writer`
            Take a :meth:`write` method and ensure its first argument is properly encoded.

        """  # noqa
        if isinstance(filename, (bytes, str, os.PathLike)):  # A file
            context_manager = open
        elif isinstance(filename, io.IOBase) or hasattr(filename, 'write'):  # A file object
            context_manager = NullContext
        else:  # Neither file nor file object
            raise TypeError(f"The 'filename' parameter is of invalid type: {repr(type(filename))}")

        with context_manager(filename, 'w') as f:
            writer = self._get_writer(f.write, encoding)
            self._write_iterate(writer, **kwargs)

    @staticmethod
    def _get_writer(writer: Callable[[str], None],
                    encoding: Optional[str] = None) -> Callable[[AnyStr], None]:
        """Take a :meth:`write` method and ensure its first argument is properly encoded.

        Parameters
        ----------
        writer : :data:`Callable<typing.Callable>`
            A write method such as :meth:`io.TextIOWrapper.write`.

        encoding : :class:`str`, optional
            Encoding used to encode the input of **writer** (*e.g.* ``"utf-8"``).
            This value will be used in :meth:`str.encode` for encoding the first positional argument
            provided to **instance_method**.
            If ``None``, return **instance_method** unaltered without any encoding.

        Returns
        -------
        :data:`Callable<typing.Callable>`
            A decorated **writer** parameter.
            The first positional argument provided to the decorated callable will be encoded
            using encoding.
            **writer** is returned unalterd if ``encoding=None``.

        See Also
        --------
        :meth:`AbstractFileContainer.write`:
            The main method for writing files.

        """
        if encoding is None:
            return writer

        @functools.wraps(writer)
        def writer_new(value: AnyStr, *args, **kwargs):
            value_bytes = encode(value, encoding)
            return writer(value_bytes, *args, **kwargs)
        return writer_new

    @abc.abstractmethod
    def _write_iterate(self, write: Callable[[AnyStr], None], **kwargs) -> None:
        r"""Write the content of this instance to an opened datastream.

        The to-be written content of this instance should be passed as :class:`str`.
        Any (potential) encoding is handled by the **write** parameter.

        .. _`file object`: https://docs.python.org/3/glossary.html#term-file-object

        Example
        -------
        Basic example of a potential :meth:`._write_iterate` implementation.

        .. code:: python

            >>> iterator = self.as_dict().items()
            >>> for key, value in iterator:
            ...     value: str = f'{key} = {value}'
            ...     write(value)
            >>> return None

        Parameters
        ----------
        writer : :data:`Callable<typing.Callable>`
            A callable for writing the content of this instance to a `file object`_.
            An example would be the :meth:`io.TextIOWrapper.write` method.

        \**kwargs : optional
            Optional keyword arguments.

        See Also
        --------
        :meth:`AbstractFileContainer.write`:
            The main method for writing files.

        """  # noqa
        raise NotImplementedError('Trying to call an abstract method')

    @classmethod
    def inherit_annotations(cls) -> Callable[[CT], CT]:
        """A decorator for inheriting annotations and docstrings.

        Can be applied to methods of :class:`AbstractFileContainer` subclasses to automatically
        inherit the docstring and annotations of identical-named functions of its superclass.

        Examples
        --------
        .. code:: python

            >>> class sub_class(AbstractFileContainer)
            ...
            ...     @AbstractFileContainer.inherit_annotations()
            ...     def write(filename, encoding=None, **kwargs):
            ...         pass

            >>> sub_class.write.__doc__ == AbstractFileContainer.write.__doc__
            True

            >>> sub_class.write.__annotations__ == AbstractFileContainer.write.__annotations__
            True

        """
        def decorator(sub_attr: CT) -> CT:
            super_attr = getattr(cls, sub_attr.__name__)
            sub_cls_name = sub_attr.__qualname__.split('.')[0]

            # Update annotations
            if not sub_attr.__annotations__:
                sub_attr.__annotations__ = dct = super_attr.__annotations__.copy()
                if 'return' in dct and dct['return'] == cls.__name__:
                    dct['return'] = sub_attr.__qualname__.split('.')[0]

            # Update docstring
            if sub_attr.__doc__ is None:
                sub_attr.__doc__ = super_attr.__doc__.replace(cls.__name__, sub_cls_name)

            return sub_attr
        return decorator


class NullContext(ContextManager[T]):
    """Context manager that does no additional processing.

    Used as a stand-in for a normal context manager, when a particular
    block of code is only sometimes used with a normal context manager:

    cm = optional_cm if condition else nullcontext()
    with cm:
        # Perform operation, using optional_cm if condition is True

    """

    def __init__(self, enter_result: T, *args: Any, **kwargs: Any) -> None:
        self.enter_result = enter_result

    def __enter__(self) -> T:
        return self.enter_result

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass
