"""Auxiliary utility functions"""
import inspect
import importlib


def import_class(module_name, cls_name):
    """Imports a class specified in yaml dynamically

    REFACTOR THIS!!

    :param str module_name: modules such as input, utils, train etc
    :param str cls_name: class to find
    """

    main_module = 'micro_dl'
    try:
        module = importlib.import_module(module_name, main_module)
        for x in dir(module):
            obj = getattr(module, cls_name)

            if inspect.isclass(obj):
                return obj
    except ImportError:
        return None
