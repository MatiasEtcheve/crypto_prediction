import inspect
import sys

import numpy as np
from IPython.core.magics.code import extract_symbols


def new_getfile(object, _old_getfile=inspect.getfile):
    """Working `inspect.getfile` for Notebook"""
    if not inspect.isclass(object):
        return _old_getfile(object)

    # Lookup by parent module (as in current inspect)
    if hasattr(object, "__module__"):
        object_ = sys.modules.get(object.__module__)
        if hasattr(object_, "__file__"):
            return object_.__file__

    # If parent module is __main__, lookup by methods (NEW)
    for name, member in inspect.getmembers(object):
        if (
            inspect.isfunction(member)
            and object.__qualname__ + "." + member.__name__ == member.__qualname__
        ):
            return inspect.getfile(member)
    else:
        raise TypeError("Source for {!r} not found".format(object))


inspect.getfile = new_getfile


def get_cell_code(obj: type) -> str:
    """Gets the notebook cell of a given object

    Args:
        obj (type): object to get the cell code

    Returns:
        str: cell code
    """
    return "".join(inspect.linecache.getlines(new_getfile(obj)))


def get_class_code(obj: type) -> str:
    """Gets the notebook code of a given object

    Args:
        obj (type): object to get the class code

    Returns:
        str: code
    """
    cell_code = get_cell_code(obj)
    return extract_symbols(cell_code, obj.__name__)[0][0]


def get_members(instance) -> dict:
    """Gets the members of an instance of any object

    Args:
        instance: instance of an object

    Returns:
        dict: dict whose keys are the name of the members of the instance.
    """
    members = {}
    for i in inspect.getmembers(instance):
        if not i[0].startswith("_") and not inspect.ismethod(i[1]):
            members[i[0]] = i[1]
    return members
