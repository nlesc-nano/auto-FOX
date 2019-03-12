""" A module with miscellaneous functions. """

__all__ = ['serialize_array']


def serialize_array(array, items_per_row=4, indent=9):
    """ """
    if len(array) == 0:
        return ''

    ret = ''
    for _ in range(indent):
        ret += ' '

    k = 0
    for i in array:
        for j in i:
            ret += '{:8.8}'.format(str(j)) + ' '
        k += 1
        if (i != array[-1]).all() and k == items_per_row:
            k = 0
            ret += '\n'
            for _ in range(indent):
                ret += ' '

    return ret
