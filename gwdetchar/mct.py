# coding=utf-8
# Copyright (C) LIGO Scientific Collaboration (2015-)
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

"""Find times when an input signal crosses a particular threshold
"""

import gwdatafind
import os
import sys

from astropy.table import vstack as vstack_tables

from gwpy.io.cache import (cache_segments, sieve as sieve_cache)
from gwpy.segments import (DataQualityFlag, Segment, SegmentList)
from gwpy.table import EventTable

from . import cli
from .const import IFO
from .daq import find_crossings
from .io.datafind import get_data
from .utils import table_from_times

__author__ = 'TJ Massinger <thomas.massinger@ligo.org>'
__credits__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

# set up logger
PROG = ('python -m gwdetchar.mct' if sys.argv[0].endswith('.py')
        else os.path.basename(sys.argv[0]))
LOGGER = cli.logger(name=PROG.split('python -m ').pop())


# -- parse command-line -------------------------------------------------------

def create_parser():
    """Create a command-line parser for this entry point
    """
    # initialize argument parser
    parser = cli.create_parser(
        prog=PROG,
        description=__doc__,
    )

    # positional arguments
    cli.add_gps_start_stop_arguments(parser)
    cli.add_ifo_option(parser)
    cli.add_nproc_option(parser)
    cli.add_frametype_option(
        parser,
        required=IFO is None,
        default=IFO is not None and f'{IFO}_R',
    )

    # optional arguments
    parser.add_argument(
        '-a',
        '--state-flag',
        metavar='FLAG',
        help='restrict search to times when FLAG was active',
    )
    parser.add_argument(
        '-o',
        '--output-path',
        help='path to output HDF5 file, name will be '
             'automatically generated based on IFO and GPS times',
    )
    parser.add_argument(
        '-t',
        '--threshold',
        type=float,
        nargs='+',
        default=[0., 2.**16, -2.**16],
        help='threshold for marking input data crossings',
    )
    parser.add_argument(
        '-c',
        '--channel',
        required=True,
        type=str,
        help='channel to read for input data',
    )
    parser.add_argument(
        '-r',
        '--rate-thresh',
        default=16.,
        type=float,
        help='if the trigger rate (Hz) is above this value for a given '
             'segment, crossings for that segment will not be recorded',
    )

    # return the argument parser
    return parser


# -- main code block ----------------------------------------------------------

def main(args=None):
    """Run the zero-crossing counter tool
    """
    parser = create_parser()
    args = parser.parse_args(args=args)

    span = Segment(args.gpsstart, args.gpsend)
    LOGGER.info('-- Processing channel %s over span %d - %d'
                % (args.channel, args.gpsstart, args.gpsend))

    if args.state_flag:
        state = DataQualityFlag.query(
            args.state_flag,
            int(args.gpsstart),
            int(args.gpsend),
        )
        statea = state.active
    else:
        statea = SegmentList([span])

    duration = abs(span)

    # initialize output files for each threshold and store them in a dict
    outfiles = {}
    for thresh in args.threshold:
        outfiles[str(thresh)] = (
            os.path.join(
                args.output_path,
                '%s_%s_DAC-%d-%d.h5'
                % (args.channel.replace('-', '_').replace(':', '-'),
                   str(int(thresh)).replace('-', 'n'),
                   int(args.gpsstart), duration)))

    # get frame cache
    cache = gwdatafind.find_urls(args.ifo[0], args.frametype,
                                 int(args.gpsstart), int(args.gpsend))

    cachesegs = statea & cache_segments(cache)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # initialize a ligolw table for each threshold and store them in a dict
    names = ("time", "frequency", "snr")
    dtypes = ("f8",) * len(names)
    tables = {}
    for thresh in args.threshold:
        tables[str(thresh)] = EventTable(
            names=names,
            dtype=dtypes,
            meta={"channel": args.channel},
        )

    # for each science segment, read in the data from frames, check for
    # threshold crossings, and if the rate of crossings is less than
    # rate_thresh, write to a sngl_burst table
    for seg in cachesegs:
        LOGGER.debug("Processing {}:".format(seg))
        c = sieve_cache(cache, segment=seg)
        if not c:
            LOGGER.warning("    No {} data files for this segment, "
                           "skipping".format(args.frametype))
            continue
        data = get_data(args.channel, seg[0], seg[1], nproc=args.nproc,
                        source=c, verbose="Reading data:".rjust(30))
        for thresh in args.threshold:
            times = find_crossings(data, thresh)
            rate = float(times.size)/abs(seg) if times.size else 0
            LOGGER.info("    Found {0} crossings of {1}, rate: {2} Hz".format(
                times.size,
                thresh,
                rate,
            ))
            if times.size and rate < args.rate_thresh:
                existing = tables[str(thresh)]
                tables[str(thresh)] = vstack_tables(
                    (existing,
                     table_from_times(times, snr=10., frequency=100.,
                                      names=existing.colnames),
                     ),
                    join_type="exact",
                )

    n = max(map(len, tables.values()))
    for thresh, outfile in outfiles.items():
        tables[thresh].write(
            outfile,
            path="triggers",
            format="hdf5",
            overwrite=True,
        )
        LOGGER.info("{0} events written to {1}".format(
            str(len(tables[thresh])).rjust(len(str(n))),
            outfile,
        ))


# -- run from command-line ----------------------------------------------------

if __name__ == "__main__":
    main()
