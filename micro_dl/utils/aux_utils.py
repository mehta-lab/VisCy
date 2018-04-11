"""Auxiliary utility functions"""
import inspect
import importlib


def import_class(cls_name):
    """Imports a class specified in yaml dynamically"""

    module = 'microDL'
    try:
        module = importlib.import_module(name, package)
