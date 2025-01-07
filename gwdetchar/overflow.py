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

"""Find ADC/DAC overflows associated with a particular front-end model
"""

import gwdatafind
import h5py
import os
import sys
import tqdm
import warnings

from gwpy.io.cache import cache_segments
from gwpy.segments import (DataQualityFlag, DataQualityDict,
                           Segment, SegmentList)

from . import (cds, cli, daq)
from .const import IFO
from .io import (html as htmlio)
from .io.datafind import get_data
from .utils import table_from_segments

from matplotlib import use
use('Agg')

__author__ = 'TJ Massinger <thomas.massinger@ligo.org>'
__credits__ = ('Duncan Macleod <duncan.macleod@ligo.org> '
               'Alex Urban <alexander.urban@ligo.org>')

SITE_MAP = {
    'H1': 'LIGO-Hanford',
    'L1': 'LIGO-Livingston',
    'K1': 'KAGRA',
    'V1': 'Virgo',
}

# set up logger
PROG = ('python -m gwdetchar.overflow' if sys.argv[0].endswith('.py')
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
    parser.add_argument(
        'dcuid',
        type=int,
        nargs='+',
        help='DCUID for the relevant front-end model',
    )
    cli.add_frametype_option(
        parser,
        required=IFO is None,
        default=IFO is not None and f'{IFO}_R',
    )
    cli.add_nproc_option(
        parser,)

    parser.add_argument(
        '--deep',
        action='store_true',
        default=False,
        help='perform deep scan, default: %(default)s',
    )
    parser.add_argument(
        '-a',
        '--state-flag',
        metavar='FLAG',
        help='restrict search to times when FLAG was active',
    )
    parser.add_argument(
        '--nds',
        metavar='HOST:PORT',
        help='use the given NDS host:port to access data, '
             'default: use datafind',
    )
    parser.add_argument(
        '-o',
        '--output-file',
        help='path to output data file, default name will be '
             'automatically generated based on IFO and GPS times',
    )
    parser.add_argument(
        '-I',
        '--integer-segments',
        action='store_true',
        default=False,
        help='pad all overflow segments to integer '
             'boundaries (default: %(default)s)',
    )
    parser.add_argument(
        '-p',
        '--segment-pad',
        type=float,
        default=0.1,
        help='minimum padding (one-sided) for output segments when '
             'using --output-format [segments|integer-segments]',
    )
    parser.add_argument(
        '-s',
        '--segment-end-pad',
        type=float,
        default=1.0,
        help='amount of time to remove from the end of each analysis segment',
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
        default=None,
        help='make plots of all overflows, default: %(default)s',
    )
    parser.add_argument(
        '-c',
        '--fec-map',
        help='URL of human-readable FEC map, default: infer from IFO',
    )
    parser.add_argument(
        '-u',
        '--simulink',
        help='URL of human-readable Simulink model, default: infer from IFO',
    )
    parser.add_argument(
        '-d',
        '--daqsvn',
        help='URL of the front-end data gathering configuration, '
             'default: internal LIGO DAQ subversion repository',
    )

    # return the argument parser
    return parser


# -- main code block ----------------------------------------------------------

def main(args=None):
    """Run the online Guardian node visualization tool
    """
    parser = create_parser()
    args = parser.parse_args(args=args)

    fec_map = args.fec_map
    simulink = args.simulink
    daqsvn = args.daqsvn or ('https://daqsvn.ligo-la.caltech.edu/websvn/'
                             'listing.php?repname=daq_maps')
    if args.ifo == 'H1':
        if not fec_map:
            fec_map = 'https://lhocds.ligo-wa.caltech.edu/exports/detchar/fec/'
        if not simulink:
            simulink = 'https://lhocds.ligo-wa.caltech.edu/daq/simulink/'
    if args.ifo == 'L1':
        if not fec_map:
            fec_map = 'https://llocds.ligo-la.caltech.edu/exports/detchar/fec/'
        if not simulink:
            simulink = 'https://llocds.ligo-la.caltech.edu/daq/simulink/'

    span = Segment(args.gpsstart, args.gpsend)

    # let's go
    LOGGER.info('{} Overflows {}-{}'.format(
        args.ifo, int(args.gpsstart), int(args.gpsend)))

    # get segments
    if args.state_flag:
        state = DataQualityFlag.query(args.state_flag, int(args.gpsstart),
                                      int(args.gpsend))
        tmp = type(state.active)()
        for i, seg in enumerate(state.active):
            if abs(seg) < args.segment_end_pad:
                continue
            tmp.append(type(seg)(seg[0], seg[1]-args.segment_end_pad))
        state.active = tmp.coalesce()
        statea = state.active
    else:
        statea = SegmentList([span])

    if not args.output_file:
        duration = abs(span)
        args.output_file = (
            '%s-OVERFLOWS-%d-%d.h5'
            % (args.ifo, int(args.gpsstart), duration))
        LOGGER.debug("Set default output file as %s" % args.output_file)

    # set up container
    overflows = DataQualityDict()

    # prepare data access
    if args.nds:
        from gwpy.io import nds2 as io_nds2
        host, port = args.nds.rsplit(':', 1)
        ndsconnection = io_nds2.connect(host, port=int(port))
        if ndsconnection.get_protocol() == 1:
            cachesegs = SegmentList([Segment(int(args.gpsstart),
                                             int(args.gpsend))])
        else:
            cachesegs = io_nds2.get_availability(
                ['{0}:FEC-1_DAC_OVERFLOW_ACC_0_0'.format(args.ifo)],
                int(args.gpsstart), int(args.gpsend),
            )
    else:  # get frame cache
        cache = gwdatafind.find_urls(args.ifo[0], args.frametype,
                                     int(args.gpsstart), int(args.gpsend))
        cachesegs = statea & cache_segments(cache)

    flag_desc = "ADC/DAC Overflow indicated by {0}"

    # get channel and find overflows
    for dcuid in args.dcuid:
        LOGGER.info("Processing DCUID %d" % dcuid)
        channel = daq.ligo_accum_overflow_channel(dcuid, args.ifo)
        overflows[channel] = DataQualityFlag(channel, known=cachesegs)
        if args.deep:
            LOGGER.debug(" -- Getting list of overflow channels")
            try:
                channels = daq.ligo_model_overflow_channels(
                    dcuid, args.ifo, args.frametype,
                    gpstime=span[0], nds=args.nds)
            except IndexError:  # no frame found for GPS start, try GPS end
                channels = daq.ligo_model_overflow_channels(
                    dcuid, args.ifo, args.frametype, gpstime=span[-1])
            for chan in channels:  # set up flags early
                overflows[chan] = DataQualityFlag(
                    chan,
                    known=cachesegs,
                    description=flag_desc.format(chan),
                    isgood=False,
                )
            LOGGER.debug(" -- %d channels found" % len(channel))
        for seg in cachesegs:
            LOGGER.debug(" -- Processing {}-{}".format(*seg))
            if args.nds:
                read_kw = dict(connection=ndsconnection)
            else:
                read_kw = dict(source=cache, nproc=args.nproc)
            msg = "Reading ACCUM_OVERFLOW data:".rjust(30)
            data = get_data(
                channel, seg[0], seg[1],
                pad=0., verbose=msg, **read_kw
            )
            new = daq.find_overflow_segments(
                data,
                cumulative=True,
            )
            overflows[channel] += new
            LOGGER.info(" -- {} overflows found".format(len(new.active)))
            if not new.active:
                continue
            # go deep!
            for s, e in tqdm.tqdm(new.active.protract(2), unit='ovfl',
                                  desc='Going deep'.rjust(30)):
                data = get_data(channels, s, e, **read_kw)
                for ch in channels:
                    try:
                        overflows[ch] += daq.find_overflow_segments(
                            data[ch],
                            cumulative=True,
                        )
                    except KeyError:
                        warnings.warn("Skipping {}".format(ch), UserWarning)
                        continue
        LOGGER.debug(" -- Search complete")

    # write output
    LOGGER.info("Writing segments to %s" % args.output_file)
    table = table_from_segments(
        overflows,
        sngl_burst=args.output_file.endswith((".xml", ".xml.gz")),
    )
    if args.integer_segments:
        for key in overflows:
            overflows[key] = overflows[key].round()
    if args.output_file.endswith((".h5", "hdf", ".hdf5")):
        with h5py.File(args.output_file, "w") as h5f:
            table.write(h5f, path="triggers")
            overflows.write(h5f, path="segments")
    else:
        table.write(args.output_file, overwrite=True)
        overflows.write(args.output_file, overwrite=True, append=True)

    # write HTML
    if args.html:
        # get base path
        base = os.path.dirname(args.html)
        os.chdir(base)
        if args.plot:
            args.plot = os.path.curdir
        if args.output_file:
            args.output_file = os.path.relpath(
                args.output_file, os.path.dirname(args.html))
        if os.path.basename(args.html) == 'index.html':
            links = [
                '%d-%d' % (int(args.gpsstart), int(args.gpsend)),
                ('Parameters', '#parameters'),
                ('Segments', [('Overflows', '#overflows')]),
                ('Results', '#results'),
            ]
            if args.state_flag:
                links[2][1].insert(0, ('State flag', '#state-flag'))
            (brand, class_) = htmlio.get_brand(args.ifo, 'Overflows',
                                               args.gpsstart)
            navbar = htmlio.navbar(links, class_=class_, brand=brand)
            page = htmlio.new_bootstrap_page(
                title='%s Overflows | %d-%d' % (
                    args.ifo, int(args.gpsstart), int(args.gpsend)),
                navbar=navbar)
        else:
            page = htmlio.markup.page()
            page.div(class_='container')

        # -- header
        page.div(class_='pb-2 mt-3 mb-2 border-bottom')
        page.h1('%s ADC/DAC Overflows: %d-%d'
                % (args.ifo, int(args.gpsstart), int(args.gpsend)))
        page.div.close()

        # -- paramters
        content = [('DCUIDs', ' '.join(map(str, args.dcuid)))]
        if daqsvn:
            content.append((
                'FEC configuration',
                ('<a href="{0}" target="_blank" title="{1} FEC configuration">'
                 '{0}</a>').format(daqsvn, args.ifo)))
        if fec_map:
            content.append((
                'FEC map', '<a href="{0}" target="_blank" title="{1} FEC '
                           'map">{0}</a>'.format(fec_map, args.ifo)))
        if simulink:
            content.append((
                'Simulink models', '<a href="{0}" target="_blank" title="{1} '
                                   'Simulink models">{0}</a>'.format(
                                       simulink, args.ifo)))
        page.h2('Parameters', class_='mt-4 mb-4', id_='parameters')
        page.div(class_='row')
        page.div(class_='col-md-9 col-sm-12')
        page.add(htmlio.parameter_table(
            content, start=args.gpsstart, end=args.gpsend,
            flag=args.state_flag))
        page.div.close()  # col-md-9 col-sm-12

        # link to summary file
        if args.output_file:
            ext = ('HDF' if args.output_file.endswith((".h5", "hdf", ".hdf5"))
                   else 'XML')
            page.div(class_='col-md-3 col-sm-12')
            page.add(htmlio.download_btn(
                [('Segments ({})'.format(ext), args.output_file)],
                btnclass='btn btn-%s dropdown-toggle' % args.ifo.lower(),
            ))
            page.div.close()  # col-md-3 col-sm-12
        page.div.close()  # row

        # -- command-line
        page.h5('Command-line:')
        page.add(htmlio.get_command_line(about=False, prog=PROG))

        # -- segments
        page.h2('Segments', class_='mt-4', id_='segments')

        # give contextual information
        msg = ("This analysis searched for digital-to-analogue (DAC) or "
               "analogue-to-digital (ADC) conversion overflows in the {0} "
               "real-time controls system. ").format(
                   SITE_MAP.get(args.ifo, 'LIGO'))
        if args.deep:
            msg += (
                "A hierarchichal search was performed, with one cumulative "
                "overflow counter checked per front-end controller (FEC). "
                "For those models that indicated an overflow, the card- and "
                "slot-specific channels were then checked. "
            )
        msg += (
            "Consant overflow is shown as yellow, while transient overflow "
            "is shown as red. If a data-quality flag was loaded for this "
            "analysis, it will be displayed in green."
        )
        page.add(htmlio.alert(msg, context=args.ifo.lower()))
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
                    'height': 0.4,
                }
            ))
            page.div.close()
        # record overflow segments
        if sum(abs(s.active) for s in overflows.values()):
            page.h3('Overflows', class_='mt-3', id_='overflows')
            page.div(id_='accordion2')
            for i, (c, flag) in enumerate(list(overflows.items())):
                if abs(flag.active) == 0:
                    continue
                if abs(flag.active) == abs(cachesegs):
                    context = 'warning'
                else:
                    context = 'danger'
                try:
                    channel = cds.get_real_channel(flag.name)
                except Exception:
                    title = '%s [%d]' % (flag.name, len(flag.active))
                else:
                    title = '%s (%s) [%d]' % (flag.name, channel,
                                              len(flag.active))
                page.add(htmlio.write_flag_html(
                    flag, span, i, parent='accordion2', title=title,
                    context=context, plotdir=args.plot))
            page.div.close()
        else:
            page.add(htmlio.alert('No overflows were found in this analysis',
                                  context=args.ifo.lower(), dismiss=False))

        # -- results table
        page.h2('Results summary', class_='mt-4', id_='results')
        page.table(class_='table table-striped table-hover')
        # write table header
        page.thead()
        page.tr()
        for header in ['Channel', 'Connected signal', 'Num. overflows']:
            page.th(header)
        page.thead.close()
        # write body
        page.tbody()
        for c, seglist in overflows.items():
            t = abs(seglist.active)
            if t == 0:
                page.tr()
            elif t == abs(cachesegs):
                page.tr(class_='table-warning')
            else:
                page.tr(class_='table-danger')
            page.td(c)
            try:
                page.td(cds.get_real_channel(str(c)))
            except Exception:
                page.td()
            page.td(len(seglist.active))
            page.tr.close()
        page.tbody.close()
        page.table.close()

        # -- close and write
        htmlio.close_page(page, args.html)
        LOGGER.info("HTML written to %s" % args.html)


# -- run from command-line ----------------------------------------------------

if __name__ == "__main__":
    main()
