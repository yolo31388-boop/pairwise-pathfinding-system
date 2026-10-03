"""Pytest support for the shipped test suite.

The tests in ``tests/`` use ``unittest``-style assertions (``self.assert*``)
inside plain classes that do not inherit from ``unittest.TestCase``. Lend
them the standard assertion methods so they can run under pytest. Classes
that already provide their own assertions (e.g. real ``TestCase`` subclasses)
are left untouched.
"""

import unittest

_ASSERTIONS = tuple(name for name in dir(unittest.TestCase)
                    if name.startswith("assert"))


def pytest_runtest_setup(item):
    cls = getattr(item, "cls", None)
    if cls is None or issubclass(cls, unittest.TestCase):
        return
    for name in _ASSERTIONS:
        if not hasattr(cls, name):
            setattr(cls, name, getattr(unittest.TestCase, name))
