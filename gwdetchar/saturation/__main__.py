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

"""Find channels clipping their software saturation limits
"""

import gwdatafind
import os
import sys

from MarkupPy import markup

from gwpy.io.cache import sieve as sieve_cache
from gwpy.io.gwf import get_channel_names
from gwpy.segments import (Segment, SegmentList,
                           DataQualityFlag, DataQualityDict)

from . import core
from .. import cli
from ..const import IFO
from ..io import html as htmlio

from matplotlib import use
use('Agg')

__author__ = 'Dan Hoak <daniel.hoak@ligo.org>'
__credits__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

# set up logger
PROG = ('python -m gwdetchar.saturation' if sys.argv[0].endswith('.py')
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

    # required arguments
    cli.add_gps_start_stop_arguments(parser)
    cli.add_ifo_option(parser)

    # optional arguments
    cli.add_frametype_option(
        parser,
        required=IFO is None,
        default=IFO is not None and f'{IFO}_R',
    )
    cli.add_nproc_option(parser)
    parser.add_argument(
        '-c',
        '--channels',
        help='file containing columnar list of channels to process, '
             'default is to find all relevant channels from frames',
    )
    parser.add_argument(
        '-s',
        '--skip',
        nargs='*',
        default=[],
        help='skip channels matching this string',
    )
    parser.add_argument(
        '-g',
        '--group-size',
        default=1024,
        type=int,
        help='number of channels to process in a single batch, '
             'default: %(default)s',
    )
    parser.add_argument(
        '-a',
        '--state-flag',
        metavar='FLAG',
        help='restrict search to times when FLAG was active',
    )
    parser.add_argument(
        '-p',
        '--pad-state-end',
        metavar='PAD',
        default=0,
        type=float,
        help='pad state segments inwards from the end '
             'by PAD segments, default: %(default)s',
    )
    parser.add_argument(
        '-m',
        '--html',
        type=os.path.abspath,
        help='path to write html output',
    )
    parser.add_argument(
        '-v',
        '--plot',
        action='store_true',
        default=False,
        help='make plots of all saturations, default: %(default)s',
    )

    # return the argument parser
    return parser


# -- main code block ----------------------------------------------------------

def main(args=None):
    """Run the software saturation command-line interface
    """
    parser = create_parser()
    args = parser.parse_args(args=args)

    # get IFO
    ifo = args.ifo.upper()
    site = ifo[0]
    frametype = args.frametype or '%s_R' % ifo

    # let's go
    LOGGER.info('{} Software saturations {}-{}'.format(
        args.ifo, int(args.gpsstart), int(args.gpsend)))

    # get segments
    span = Segment(args.gpsstart, args.gpsend)
    if args.state_flag:
        state = DataQualityFlag.query(args.state_flag, int(args.gpsstart),
                                      int(args.gpsend))
        for i, seg in enumerate(state.active):
            state.active[i] = type(seg)(seg[0], seg[1]-args.pad_state_end)
        segs = state.active.coalesce()
        LOGGER.debug("Recovered %d seconds of time for %s"
                     % (abs(segs), args.state_flag))
    else:
        segs = SegmentList([Segment(args.gpsstart, args.gpsend)])

    # find frames
    cache = gwdatafind.find_urls(
        site, frametype, int(args.gpsstart), int(args.gpsend))

    # find channels
    if not os.getenv('LIGO_DATAFIND_SERVER'):
        raise RuntimeError("No LIGO_DATAFIND_SERVER variable set, don't know "
                           "how to discover channels")
    else:
        LOGGER.debug("Identifying channels in frame files")
        if len(cache) == 0:
            raise RuntimeError(
                "No frames recovered for %s in interval [%s, %s)" %
                (frametype, int(args.gpsstart),
                 int(args.gpsend)))
        allchannels = get_channel_names(cache[0])
        LOGGER.debug("   Found %d channels" % len(allchannels))
        sys.stdout.flush()
        channels = core.find_limit_channels(allchannels, skip=args.skip)
        LOGGER.info(
            "   Parsed %d channels with '_LIMIT' and '_LIMEN' or '_SWSTAT'"
            % sum(map(len, channels)))

    # -- read channels and check limits -------------

    saturations = DataQualityDict()
    bad = set()

    # TODO: use multiprocessing to separate channel list into discrete chunks
    #       should give a factor of X for X processes

    # check limens
    for suffix, clist in zip(['LIMEN', 'SWSTAT'], channels):
        nchans = len(clist)
        # group channels in sets for batch processing
        #     min of <number of channels>, user group size (sensible number),
        #     and 512 Mb of RAM for single-precision EPICS
        try:
            dur = max([float(abs(s)) for s in segs])
        except ValueError:
            ngroup = args.group_size
        else:
            ngroup = int(
                min(nchans, args.group_size, 2 * 1024**3 / 4. / 16. / dur))
        LOGGER.info('Processing %s channels in groups of %d' % (
            suffix, ngroup))
        sys.stdout.flush()
        sets = core.grouper(clist, ngroup)
        for i, cset in enumerate(sets):
            # remove empty entries use to pad the list to 8 elements
            cset = list(cset)
            while cset[-1] is None:
                cset.pop(-1)
            for seg in segs:
                cache2 = sieve_cache(cache, segment=seg)
                if not len(cache2):
                    continue
                saturated = core.is_saturated(
                    cset, cache2, seg[0], seg[1], indicator=suffix,
                    nproc=args.nproc)
                for new in saturated:
                    try:
                        saturations[new.name] += new
                    except KeyError:
                        saturations[new.name] = new
            for j, c in enumerate(cset):
                try:
                    sat = saturations[c]
                except KeyError:
                    LOGGER.debug('%40s:      SKIP      [%d/%d]'
                                 % (c, i*ngroup + j + 1, nchans))
                else:
                    if abs(sat.active):
                        LOGGER.debug('%40s: ---- FAIL ---- [%d/%d]'
                                     % (c, i*ngroup + j + 1, nchans))
                        for seg in sat.active:
                            LOGGER.debug(" " * 42 + str(seg))
                        bad.add(c)
                    else:
                        LOGGER.debug('%40s:      PASS      [%d/%d]'
                                     % (c, i*ngroup + j + 1, nchans))
                sys.stdout.flush()

    # -- log results and exit -----------------------

    if len(bad):
        LOGGER.info("Saturations were found for all of the following:\n\n")
        for c in bad:
            print(c)
        print('\n\n')
    else:
        LOGGER.info("No software saturations were found in any channels")

    # write segments to file
    outfile = ('%s-SOFTWARE_SATURATIONS-%d-%d.h5'
               % (ifo, int(args.gpsstart),
                  int(args.gpsend) - int(args.gpsstart)))
    LOGGER.info("Writing saturation segments to %s" % outfile)
    saturations.write(outfile, path="segments", overwrite=True)

    if args.html:
        # get base path
        base = os.path.dirname(args.html)
        os.chdir(base)
        if args.plot:
            args.plot = os.path.curdir
        segfile = os.path.relpath(outfile, os.path.dirname(args.html))
        if os.path.basename(args.html) == 'index.html':
            links = [
                '%d-%d' % (int(args.gpsstart), int(args.gpsend)),
                ('Parameters', '#parameters'),
                ('Segments', [('Software saturations',
                               '#software-saturations')]),
                ('Results', '#results'),
            ]
            if args.state_flag:
                links[2][1].insert(0, ('State flag', '#state-flag'))
            (brand, class_) = htmlio.get_brand(ifo, 'Saturations',
                                               args.gpsstart)
            navbar = htmlio.navbar(links, class_=class_, brand=brand)
            page = htmlio.new_bootstrap_page(
                navbar=navbar, title='%s Saturations | %d-%d' % (
                    ifo, int(args.gpsstart), int(args.gpsend)))
        else:
            page = markup.page()
            page.div(class_='container')
        # -- header
        page.div(class_='pb-2 mt-3 mb-2 border-bottom')
        page.h1('%s Software Saturations: %d-%d'
                % (ifo, int(args.gpsstart), int(args.gpsend)))
        page.div.close()
        # -- paramters
        content = [
            ('State end padding', args.pad_state_end),
            ('Skip', ', '.join(map(repr, args.skip)))]
        page.h2('Parameters', class_='mt-4 mb-4', id_='parameters')
        page.div(class_='row')
        page.div(class_='col-md-9 col-sm-12')
        page.add(htmlio.parameter_table(
            content, start=args.gpsstart, end=args.gpsend,
            flag=args.state_flag))
        page.div.close()  # col-md-9 col-sm-12
        page.div(class_='col-md-3 col-sm-12')
        page.add(htmlio.download_btn(
            [('Segments (HDF)', segfile)],
            btnclass='btn btn-%s dropdown-toggle' % ifo.lower(),
        ))
        page.div.close()  # col-md-9 col-sm-12
        page.div.close()  # row
        page.h5('Command-line:')
        page.add(htmlio.get_command_line(about=False, prog=PROG))
        # -- segments
        page.h2('Segments', class_='mt-4', id_='segments')
        msg = ("This analysis searched {0} filter bank readback channels for "
               "time periods during which their OUTPUT value matched or "
               "exceeded the LIMIT value set in software. Signals that "
               "achieve saturation are shown below, and saturation segments "
               "are available by expanding a given panel.").format(
                   sum(map(len, channels)))
        page.add(htmlio.alert(msg, context=ifo.lower()))
        # record state segments
        if args.state_flag:
            page.h3('State flag', class_='mt-3', id_='state-flag')
            page.div(id_='accordion1')
            page.add(htmlio.write_flag_html(
                state, span, 'state', parent='accordion1', context='success',
                plotdir=args.plot, facecolor=(0.2, 0.8, 0.2),
                edgecolor='darkgreen', known={
                    'facecolor': 'red',
                    'edgecolor': 'darkred',
                    'height': 0.4},
            ))
            page.div.close()
        # record saturation segments
        if len(bad):
            page.h3('Software saturations', class_='mt-3',
                    id_='software-saturations')
            page.div(id_='accordion2')
            for i, (c, flag) in enumerate(saturations.items()):
                if abs(flag.active) > 0:
                    title = '%s [%d]' % (flag.name, len(flag.active))
                    page.add(htmlio.write_flag_html(
                        flag, span=span, id=i, parent='accordion2',
                        title=title, plotdir=args.plot))
            page.div.close()
        else:
            page.add(htmlio.alert('No software saturations were found in this '
                                  'analysis', context=ifo.lower(),
                                  dismiss=False))
        # -- results table
        page.h2('Results summary', class_='mt-4', id_='results')
        page.add(htmlio.alert('All channels for which the LIMIT setting was '
                              'active are shown below.', context=ifo.lower()))
        page.table(class_='table table-striped table-hover')
        # write table header
        page.thead()
        page.tr()
        for header in ['Channel', 'Result', 'Num. saturations']:
            page.th(header)
        page.thead.close()
        # write body
        page.tbody()
        for c, seglist in saturations.items():
            passed = abs(seglist.active) == 0
            if passed:
                page.tr()
            else:
                page.tr(class_='table-warning')
            page.td(c)
            page.td(passed and 'Pass' or 'Fail')
            page.td(len(seglist.active))
            page.tr.close()
        page.tbody.close()
        page.table.close()
        # close and write
        htmlio.close_page(page, args.html)


# -- run from command-line ----------------------------------------------------

if __name__ == "__main__":
    main()
