"""
FOX.logger
==========

A module for managing all Auto-FOX loggers.

Index
-----
.. currentmodule:: FOX.logger
.. autosummary::
    get_logger

API
---
.. autofunction:: get_logger

"""

import os
import logging
from logging import Logger, StreamHandler
from functools import wraps
from typing import Optional, Type, Any, Callable, Union

from scm.plams import config

__all__ = ['DEFAULT_LOGGER', 'DummyLogger', 'get_logger', 'Plams2Logger', 'wrap_plams_logger']


def _pass(*args: Any, **kwargs: Any) -> None:
    pass


class DummyLogger(Logger):
    """A :class:`~logging.Logger` subclass whose methods do absolutely nothing."""

    __slots__ = ()

    __init__ = wraps(Logger.__init__)(_pass)
    __repr__ = object.__repr__

    warning = warn = wraps(Logger.warning)(_pass)
    info = wraps(Logger.info)(_pass)
    debug = wraps(Logger.debug)(_pass)
    error = wraps(Logger.error)(_pass)
    critical = fatal = wraps(Logger.critical)(_pass)
    log = wraps(Logger.log)(_pass)
    exception = wraps(Logger.exception)(_pass)


def wrap_plams_logger(logfile: Union[None, str, os.PathLike] = None,
                      name: str = 'logger', **kwargs: Any) -> Logger:
    """Substitute the PLAMS .log file for one created by a :class:`Logger<logging.Logger>`."""
    # Define filenames
    plams_logfile = config.default_jobmanager.logfile
    logfile = os.path.abspath(logfile) if logfile is not None else plams_logfile

    # Modify the plams logger
    config.log.time = False
    config.log.file = 0

    # Replace the plams logger with a proper logging.Logger instance
    os.remove(plams_logfile)
    logger = get_logger(name, filename=logfile, **kwargs)
    if plams_logfile != logfile:
        try:
            os.symlink(logfile, plams_logfile)
        except OSError:
            pass

    return logger


def get_logger(name: str,
               handler_type: Type[logging.Handler] = logging.FileHandler,
               level: int = logging.DEBUG,
               style: str = '{',
               fmt: Optional[str] = '[{asctime}] {levelname}: {message}',
               datefmt: Optional[str] = '%H:%M:%S',
               **kwargs: Any) -> Logger:
    r"""Create and return a new :class:`Logger<logging.Logger>` instance.

    More details about the various options is provided in the :mod:`logging` module.

    Examples
    --------
    .. code:: python

        >>> import logging

        >>> name = 'my_logger'
        >>> filename = 'path/to/my/logger.log'

        >>> logger: logging.Logger = get_logger(name=name, filename=filename)


    Parameters
    ----------
    name : :class:`str`
        The name of the to-be returned logger.

    handler_type : :class:`type` [:class:`logging.Handler`]
        The type of logging Handler assigned to the to-ber returned logger.
        Note that certain handler types require additional positional arguments,
        *e.g.* the ``filename`` argument of :class:`~logging.FileHandler`.

    level : :class:`int`
        The level of logging.

    style : :class:`str`
        The type of string-formatting to be used by
        the logger's :class:`Formatter<logging.Formatter>`.

    fmt : :class:`str`, optional
        A pre-formatted string for to-be reported messages.

    datefmt : :class:`str`, optional
        A pre-formatted string for to-be reported date(s)/time(s).

    \**kwargs : :data:`Any<typing.Any>`
        Further keyword arguments specific to the provided **handler_type**.
        The relevant arguments for :class:`~logging.FileHandler` are
        ``filename`` (positional), ``mode``, ``encoding`` and ``delay``.

    Returns
    -------
    :class:`Logger<logging.Logger>`
        A newly constructed Logger instance.

    """
    # Create and customize the console handler
    try:
        handler = handler_type(**kwargs)
    except TypeError as ex:
        if 'required positional argument' not in ex.args[0]:
            raise ex
        ex.args = (f'{handler_type.__name__}.{ex.args[0]}',)
        raise ex

    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt, style=style))

    #: The Auto-FOX ARMC logger.
    logger = logging.getLogger(name=name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


class Plams2Logger:
    r"""A file-like object for redirecting plams :func:`log` output to a :class:`Logger<logging.Logger>`.

    Examples
    --------
    .. code:: python

        >>> import logging
        >>> from contextlib import redirect_stdout

        >>> logger: logging.Logger = ...
        >>> writer = Plams2Logger(logger)

        >>> with redirect_stdout(write):
        ...     ...


    Parameters
    ----------
    logger : :class:`logging.Logger`
        The logger which should take the redirected output.

    \*filters : :data:`Callable`
        One or more callables which take a :class:`str` as input and return a :class:`bool`.
        If one (or more) callables evaluate to ``True`` than the passed string
        will be ignored by :meth:`Plams2Logger.write`.

    """  # noqa

    @property
    def info(self) -> Callable[[str], None]:
        """Return the :meth:`Logger.info<logging.Logger.info>` method of :attr:`Plams2Logger.logger.`."""  # noqa
        return self.logger.info

    @property
    def warning(self) -> Callable[[str], None]:
        """Return the :meth:`Logger.warning<logging.Logger.warning>` method of :attr:`Plams2Logger.logger.`."""  # noqa
        return self.logger.warning

    def __init__(self, logger: logging.Logger, *filters: Callable[[str], bool]) -> None:
        self.logger = logger
        self.filters = filters

    def write(self, item: str) -> None:
        """Write **item** to the logger."""
        if item == '\n':  # Empty string
            return

        # Check if any of the filters evaluate to True; abort if this is the case
        for func in self.filters:
            if func(item):
                return

        if 'WARNING: ' in item:  # Remove the prepended 'WARNING'
            item = item[9:]
            self.warning(item)
        elif 'CRASHED' in item or 'Obtaining results of' in item:
            self.warning(item)
        else:
            self.info(item)

    def flush(self) -> None:
        """Dummy method for ensuring this instances' compatibility as a pseudo-filelike object."""
        return None


#: The default :class:`~logging.Logger` used by :class:`~FOX.armc.MonteCarloABC`.
DEFAULT_LOGGER = get_logger('FOX', handler_type=StreamHandler)
