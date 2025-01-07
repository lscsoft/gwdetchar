# coding=utf-8
# Copyright (C) Duncan Macleod (2015)
#
# This file is part of the GW DetChar python package.
#
# GW DetChar is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GW DetChar is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GW DetChar.  If not, see <http://www.gnu.org/licenses/>.

"""Constants for `gwdetchar`
"""

import os
from collections import OrderedDict

from gwpy.segments import Segment

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

IFO = os.getenv('IFO', None)
ifo = os.getenv('ifo', IFO.lower() if IFO else None)
SITE = os.getenv('SITE', None)
site = os.getenv('site', SITE.lower() if SITE else None)

# -- Run epochs ---------------------------------------------------------------

EPOCH = OrderedDict([
    ('S6', Segment(931035615, 971622015)),
    ('ER7', Segment(1117400416, 1118329216)),
    ('ER8', Segment(1123858817, 1126623617)),
    ('O1', Segment(1126623617, 1136649617)),
    ('ER9', Segment(1152136817, 1152255617)),
    ('ER10', Segment(1161907217, 1164499217)),
    ('O2', Segment(1164499217, 1187733618)),
    ('O3', Segment(1187733618, 1366556418)),
    ('O4', Segment(1366556418, 2000000000)),
])


def latest_epoch():
    """Returns the name of the latest known epoch
    """
    epochs = sorted(EPOCH.items(), key=lambda x: x[1][0])
    return epochs[-1][0]


def gps_epoch(gpstime, default=None):
    """Returns the epoch name containing the given GPS time

    Parameters
    ----------
    gpstime : `float`, `int`
        the GPS time to match

    default : `str`, optional
        the epoch to return if ``gpstime`` doesn't match any
        known epoch, default is to raise `ValueError`

    Returns
    -------
    epoch : `str`
        the name of the containing epoch, or ``default`` if given

    Raises
    ------
    ValueError
        if ``gpstime`` doesn't match any known epoch, and ``default``
        was not given

    Examples
    --------
    >>> from gwdetchar.const import gps_epoch
    >>> print(gps_epoch(1126259462))
    'O1'
    >>> print(gps_epoch(0, default='ER0')
    'ER0'
    """
    for epoch in EPOCH:
        if float(gpstime) in EPOCH[epoch]:
            return epoch
    if default:
        return default
    raise ValueError("failed to match GPS time {0} to any "
                     "known epoch".format(gpstime))
