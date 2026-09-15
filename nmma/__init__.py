# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: The NMMA Team

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("nmma")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
