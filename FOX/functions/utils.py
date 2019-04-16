""" A module with miscellaneous functions. """

__all__ = ['get_template', 'template_to_df', 'update_charge']

from os.path import join
import pkg_resources as pkg

import pandas as pd

from scm.plams import Settings

try:
    import yaml
    YAML_ERROR = False
except ImportError:
    __all__ = []
    YAML_ERROR = "Use of the FOX.{} function requires the 'pyyaml' package (version >=5.1).\
                  \n\t'pyyaml' can be installed via anaconda or pip with the following commands:\
                  \n\tconda install --name FOX -y -c conda-forge pyyaml\
                  \n\tpip install pyyaml"


def get_template(name, path=None, as_settings=True):
    """ Grab a .yaml template and turn it into a Settings object.

    :parameeter str name: The name of the template file.
    :parameter str path: The path where **name** is located.
        Will default to the FOX.data directory if *None*.
    :parameter bool as_settings: If *False*, return a dictionary rather than a settings object.
    :return: A settings object or dictionary as constructed from the template file.
    :rtype: |plams.Settings|_ or |dict|_
    """
    if path is None:
        path = pkg.resource_filename('FOX', join('data', name))
    else:
        path = join(path, name)

    with open(path, 'r') as f:
        if as_settings:
            return Settings(yaml.load(f, Loader=yaml.FullLoader))
        return yaml.load(f, Loader=yaml.FullLoader)


def template_to_df(name, path=None):
    """ Grab a .yaml template and turn it into a pandas dataframe.

    :parameeter str name: The name of the template file.
    :parameter str path: The path where **name** is located.
        Will default to the FOX.data directory if *None*.
    :return: A dataframe as constructed from the template file.
    :rtype: |pd.DataFrame|_
    """
    template_dict = get_template(name, path=None, as_settings=False)
    try:
        return pd.DataFrame(template_dict).T
    except ValueError:
        idx = list(template_dict.keys())
        values = list(template_dict.values())
        return pd.DataFrame(values, index=idx, columns=['charge'])


def serialize_array(array, items_per_row=4):
    """ Serialize an array into a single string.
    Newlines are placed for every **items_per_row** rows in **array**.

    :parameter array: A 2D array.
    :type array: |np.ndarray|_
    :parameter int items_per_row: The number of values per row before switching to a new line.
    :return: A serialized array.
    :rtype: |str|_
    """
    if len(array) == 0:
        return ''

    ret = ''
    k = 0
    for i in array:
        for j in i:
            ret += '{:>10d}'.format(j)
        k += 1
        if k == items_per_row:
            k = 0
            ret += '\n'

    return ret


def read_str_file(filename):
    """ Read atomic charges from CHARMM-compatible stream files (.str), returning a settings object
    with atom types and (atomic) charges.

    :parameter str filename: the path+filename of the .str file.
    :return: A settings object with atom types and (atomic) charges
    :rtype: |plams.Settings|_ (keys: |str|_, values: |tuple|_ [|str|_ or |float|_])
     """
    def inner_loop(f):
        ret = []
        for j in f:
            if j != '\n':
                j = j.split()[2:4]
                ret.append((j[0], float(j[1])))
            else:
                return ret

    with open(filename, 'r') as f:
        for i in f:
            if 'GROUP' in i:
                return zip(*inner_loop(f))


def read_param(filename):
    """ Read a CHARMM parameter file.

    :parameter str filename: the path+filename of the CHARMM parameter file.
    :return: A settings object consisting of 5 dataframes assigned to the following keys:
        *bonds*, *angles*, *dihedrals*, *improper* & *nonbonded*.
    :rtype: |plams.Settings|_ (keys: |str|_, values: |pd.DataFrame|_)
     """
    with open(filename, 'r') as f:
        str_list = f.read().splitlines()
    str_gen = (i for i in str_list if '*' not in i and '!' not in i)

    headers = ['BONDS', 'ANGLES', 'DIHEDRALS', 'IMPROPER', 'NONBONDED']
    df_dict = {}
    for i in str_gen:
        if i in headers:
            tmp = []
            for j in str_gen:
                if j:
                    tmp.append(j.split())
                else:
                    df_dict[i.lower()] = pd.DataFrame(tmp)
                    break

    return Settings(df_dict)


def get_shape(item):
    """ Try to infer the shape of an object.

    :parameter object item: A python object.
    :return: The shape of **item**.
    :rtype: |tuple|_ [|int|_]
    """
    if hasattr(item, 'shape'):  # Plan A: **item** is an np.ndarray derived object
        return item.shape
    elif hasattr(item, '__len__'):  # Plan B: **item** has access to the __len__() magic method
        return (len(item), )
    return (1, )  # Plan C: **item** has access to neither A nor B


def flatten_dict(input_dict):
    """ Flatten a dictionary.
    The keys of the to be returned dictionary consist are tuples with the old (nested) keys
    of **input_dict**.

    .. code-block:: python

        >>> print(input_dict)
        {'a': {'b': {'c': True}}}

        >>> output_dict = flatten_dict(input_dict)
        >>> print(output_dict)
        {('a', 'b', 'c'): True}

    :parameter dict input_dict: A (nested) dicionary.
    :return: A non-nested dicionary derived from **input_dict**.
    :rtype: |dict|_ (keys: |tuple|_)
    """
    def flatten(item):
        ret = {}
        for i in item:
            if isinstance(item[i], dict):
                for j in item[i]:
                    ret[i + (j, )] = item[i][j]
            else:
                ret[i] = item[i]

        if item == ret:
            return ret, True
        return ret, False

    # Changes keys into tuples
    flat_dict = {(key, ): input_dict[key] for key in input_dict}

    # Un-nest and return the input dictionary
    flat = False
    while not flat:
        flat_dict, flat = flatten(flat_dict)
    return flat_dict


def dict_to_pandas(input_dict, name=0, object_type='DataFrame'):
    """ Turn a (nested) dictionary into a pandas series or dataframe.
    Keys are un-nested and used for generating multiindices (see meth:`flatten_dict`).

    :parameter dict input_dict: A (nested) dictionary.
    :parameter object name: The name of the to be returned series/dataframe.
    :parameter str object_type: The object type of the to be returned item.
        Accepted values are *Series* or *DataFrame*
    :return: A pandas series or dataframe created fron **input_dict**.
    :rtype: |pd.Series|_ or |pd.DataFrame|_ (index: |pd.MultiIndex|_)
    """
    flat_dict = flatten_dict(input_dict)
    idx = pd.MultiIndex.from_tuples(flat_dict.keys())
    if object_type.lower() == 'series':
        return pd.Series(list(flat_dict.values()), index=idx, name=name)
    elif object_type.lower() == 'dataframe':
        return pd.DataFrame(list(flat_dict.values()), index=idx, columns=[name])


def update_charge(at, charge, series, constrain_dict={}):
    """ Set the atomic charge of **at** to **charge**, imposing the following constrains to
    all remaining values in **series**:

        * The total charge must be equal to the sum of all atomic charges in **series**.
        * Optional constraints, as provided in **constrain_dict**, must be satisfied.
    Performs an inplace update of **series**.

    |

    Example input (and the resulting charge constrains) for **constrain_dict**:

    .. code-block:: python

        constrain_dict = {}
        constrain_dict['Cd'] = {'Se': -1, 'OG2D2': -0.5}
        constrain_dict['Se'] = {'Cd': -1, 'OG2D2': 0.5}
        constrain_dict['OG2D2'] = {'Se': 2, 'Cd': -2}

    .. math::

        q_{Cd} = -1*q_{Se} = -0.5*q_{OG2D2}

    :parameter str at: An atom type such as *Se*, *Cd* or *OG2D2*.
    :parameter float charge: The new charge associated with **at**.
    :parameter series: A series of atomic charges. **series.index** should consist of
        atom types (see **at**).
    :type series: |pd.Series|_ (index: |pd.Index|_, values: |np.float64|_)
    :parameter dict constrain_dict: A dictionary with charge constrains.
    """
    if at not in series.index:
        raise IndexError('{} not available in series.index'.format(str(at)))
    net_charge = series.sum()

    # Update all constrained charges
    series[series.index == at] = charge
    at_list = [at]
    if at in constrain_dict:
        for at2 in constrain_dict[at]:
            at_list.append(at2)
            series[series.index == at2] *= constrain_dict[at][at2]

    # Update all unconstrained charges
    criterion = [i not in at_list for i in series.index]
    i = series[criterion].sum() / net_charge
    series[criterion] /= i


# If pyyaml is not installed
if YAML_ERROR:
    _doc = get_template.__doc__
    def get_template(name, path=None):
        raise ModuleNotFoundError(YAML_ERROR.format('get_template'))


    get_template.__doc__ = _doc
