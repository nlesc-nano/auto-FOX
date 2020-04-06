import io
import os
from typing import TYPE_CHECKING, Union, AnyStr, Optional
from functools import partial
from contextlib import nullcontext

import yaml
from scm.plams import Settings

try:
    Dumper = yaml.CDumper
except AttributeError:
    Dumper = yaml.Dumper  # type: ignore

if TYPE_CHECKING:
    from .armc import ARMC
else:
    from ..type_alias import ARMC

__all__ = ['to_yaml']


def to_yaml(obj: ARMC, filename: Union[AnyStr, os.PathLike, io.IOBase],
            logfile: Optional[str] = None, path: Optional[str] = None,
            folder: Optional[str] = None) -> None:
    """Convert an :class:`ARMC` instance into a .yaml readable by :class:`ARMC.from_yaml`.

    Parameters
    ----------
    filename : :class:`str`, :class:`bytes`, :class:`os.pathlike` or :class:`io.IOBase`
        A filename or a file-like object.

    """
    raise NotImplementedError
    try:  # is filename an actual filename or a file-like object?
        assert callable(filename.write)
    except (AttributeError, AssertionError):
        manager = partial(open, mode='w')
    else:
        manager = nullcontext

    # The armc block
    s = Settings()
    s.armc.iter_len = obj.iter_len
    s.armc.sub_iter_len = obj.sub_iter_len
    s.armc.gamma = obj.phi.gamma
    s.armc.a_target = obj.phi.a_target
    s.armc.phi = obj.phi.phi

    # The hdf5 block
    s.hdf5_file = obj.hdf5

    # The pram block
    # s.param = df_to_dict(obj.param)

    # The job block
    s.job.path = os.getcwd() if path is None else str(path)
    if logfile is not None:
        s.job.logfile = logfile
    if folder is not None:
        s.job.folder = folder
    s.job.keep_files = obj.keep_files

    wm = obj.workflow_manager
    s.job.job_type = f"{wm['md'][0].__module__}.{wm['md'][0].__class__.__qualname__}"
    s.job.name = wm['md'][0].name
    s.job.md_settings = wm['md'][0].settings
    if 'geometry' in wm:
        s.job.preopt_settings = wm['geometry'][0].settings
        if 'geometry' in wm.post_process:
            s.job.rmsd_threshold = wm.post_process['geometry'].keywords.get('threshold')

    s.psf = {}
    with Settings.supress_missing():
        try:
            s.psf.psf_file = [s.input.force_eval.subsys.topology.conn_file_name for
                              s in wm['geometry']]
            del s.job.md_settings.input.force_eval.subsys.topology.conn_file_name
        except KeyError:
            pass

        try:
            del s.job.preopt_settings.input.force_eval.subsys.topology.conn_file_name
        except (AttributeError, KeyError):
            pass

    # The molecule block
    s.molecule = [mol.properties.filename for mol in obj.molecule]

    # The pes block
    for name, partial_func in obj.pes.items():
        pes_dict = s.pes[name.rsplit('.', maxsplit=1)[0]]
        pes_dict.func = f'{partial_func.__module__}.{partial_func.__qualname__}'
        pes_dict.args = list(partial_func.args)
        if 'kwargs' not in pes_dict:
            pes_dict.kwargs = []
        pes_dict.kwargs.append(partial_func.keywords)

    # The move block
    move_range = obj.param.move_range
    s.move.range.stop = round(float(move_range.max() - 1), 8)
    s.move.range.step = round(float(abs(move_range[-1] - move_range[-2])), 8)
    s.move.range.start = round(float(move_range[len(move_range) // 2] - 1), 8)

    # Write the file
    with manager(filename, 'w') as f:
        f.write(yaml.dump(s.as_dict(), Dumper=Dumper))
