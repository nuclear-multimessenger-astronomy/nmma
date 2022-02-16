"""
NMMA
=====

NMMA: A nuclear multi-messenger Bayesian inference library.

"""

from __future__ import absolute_import
import os


def get_version_information():
    version_file = os.path.join(os.path.dirname(__file__), ".version")
    try:
        with open(version_file, "r") as f:
            return f.readline().rstrip()
    except EnvironmentError:
        print("No version information file '.version' found")


__version__ = get_version_information()
