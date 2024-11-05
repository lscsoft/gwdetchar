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

"""Utilties for LIGO_LW XML I/O
"""

from ligo.lw import (
    ligolw,
    table,
)
try:
    from ligo.lw import lsctables
except ModuleNotFoundError as exc:
    exc.msg = (
        f"{exc.msg}, please install python-lal / lalsuite / python-ligo-lw "
        "to handle LIGO_LW files"
    )
    exc.args = (exc.msg,)
    raise

from gwpy.segments import (Segment, DataQualityFlag, DataQualityDict)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def new_table(tab, *args, **kwargs):
    """Create a new `~ligo.lw.table.Table`

    This is just a convenience wrapper around `~ligo.lw.lsctables.New`

    Parameters
    ----------
    tab : `type`, `str`
        `~ligo.lw.table.Table` subclass, or name of table to create
    *args, **kwargs
        other parameters are passed directly to `~ligo.lw.lsctables.New`

    Returns
    -------
    table : `~ligo.lw.table.Table`
        a newly-created table with the relevant attributes and structure
    """
    if isinstance(tab, str):
        tab = lsctables.TableByName[table.Table.TableName(tab)]
    return lsctables.New(tab, *args, **kwargs)


def sngl_burst_from_times(times, **params):
    """Create a `SnglBurstTable` from an array of times
    """
    columns = set(params.keys()) | {'peak_time', 'peak_time_ns', 'event_id'}
    table = new_table('sngl_burst', columns=list(columns))
    get_next_id = table.get_next_id
    RowType = table.RowType
    append = table.append
    for t in times:
        row = RowType()
        row.event_id = get_next_id()
        row.peak = t
        for key, val in params.items():
            setattr(row, key, val)
        append(row)
    return table


def sngl_burst_from_segments(segs, **params):
    """Create a `SnglBurstTable from a `~ligo.segments.segmentlist`
    """
    table = new_table('sngl_burst', columns=params.keys())
    get_next_id = table.get_next_id
    RowType = table.RowType
    append = table.append
    for seg in segs:
        row = RowType()
        row.event_id = get_next_id()
        row.peak = seg[0] + abs(seg) / 2.
        row.period = seg
        for key, val in params.items():
            setattr(row, key, val)
        append(row)
    return table


def segments_from_sngl_burst(table, padding, known=None):
    """Create a `DataQualityDict` of segments from the given `SnglBurstTable`

    This method creates a `~gwpy.segments.DataQualityFlag` for each unique
    channel found in the given table by padding the peak time by the given
    amount on each side
    """
    out = DataQualityDict()
    for row in table:
        t = row.peak
        seg = Segment(t-padding, t+padding)
        try:
            out[row.channel].active.append(seg)
        except KeyError:
            out[row.channel] = DataQualityFlag(row.channel, known=known,
                                               active=[seg])
    return out


def table_to_document(table):
    xmldoc = ligolw.Document()
    xmldoc.appendChild(ligolw.LIGO_LW())
    xmldoc.childNodes[0].appendChild(table)
    return xmldoc
