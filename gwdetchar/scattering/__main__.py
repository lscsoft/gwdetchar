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

"""Search for evidence of beam scattering based on optic velocity

This tool identifies broad time segments over which evidence for scattering is
strong. To compare projected fringes against spectrogram measurements for a
specific, narrow time range, please use the `simple` command-line module:

python -m gwdetchar.scattering.simple --help
"""

import numpy
import os.path
import random
import re
import sys
import warnings

from io import StringIO
from MarkupPy import markup

import gwtrigfind

from gwpy.segments import (
    DataQualityFlag,
    DataQualityDict,
    Segment,
    SegmentList,
)
from gwpy.table import EventTable
from gwpy.table.filters import in_segmentlist

from .. import cli
from ..io import html as htmlio
from ..io.datafind import get_data
from ..omega import batch

from . import (
    FREQUENCY_MULTIPLIERS,
    OPTIC_MOTION_CHANNELS,
    TRANSMON_CHANNELS,
    HIST_CAPTION,
    SCATTER_CAPTION,
    get_blrms,
    get_fringe_frequency,
    get_segments,
)

from matplotlib import (use, rcParams)
use('Agg')

# backend-dependent imports
from gwpy.plot import Plot  # noqa: E402
from ..plot import texify  # noqa: E402

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__credits__ = 'Siddharth Soni <siddharth.soni@ligo.org>' \
              'Alex Urban <alexander.urban@ligo.org>'

# set program name
PROG = ('python -m gwdetchar.scattering' if sys.argv[0].endswith('.py')
        else os.path.basename(sys.argv[0]))

# rcParams settings
rcParams.update({
    'axes.labelsize': 20,
    'figure.subplot.bottom': 0.17,
    'figure.subplot.left': 0.1,
    'figure.subplot.right': 0.9,
    'figure.subplot.top': 0.90,
    'grid.color': 'gray',
    'image.cmap': 'viridis',
    'svg.fonttype': 'none',
})


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
        '-a',
        '--state-flag',
        metavar='FLAG',
        default=None,
        help='restrict search to times when FLAG was active',
    )
    parser.add_argument(
        '-z',
        '--fmin',
        metavar='FMIN',
        type=float,
        default=10.,
        help='minimum frequency (Hz) of triggers, '
             'default: %(default)s',
    )
    parser.add_argument(
        '-t',
        '--frequency-threshold',
        type=float,
        default=15.,
        help='critical fringe frequency threshold (Hz), '
             'default: %(default)s',
    )
    parser.add_argument(
        '-x',
        '--multiplier-for-threshold',
        type=int,
        default=4,
        choices=FREQUENCY_MULTIPLIERS,
        help='fringe frequency multiplier to use when applying '
             '--frequency-threshold, default: %(default)s',
    )
    parser.add_argument(
        '-m',
        '--optic',
        action='append',
        help='optic to search for scattering signal, can be given '
             'multiple times, default: {}'.format(
                 list(OPTIC_MOTION_CHANNELS.keys())),
    )
    parser.add_argument(
        '-p',
        '--segment-padding',
        type=float,
        default=.05,
        help='time with which to pad scattering segments on '
             'either side, default: %(default)s',
    )
    parser.add_argument(
        '-s',
        '--segment-start-pad',
        type=float,
        default=30.0,
        help='amount of time to remove from the start of each '
             'analysis segment',
    )
    parser.add_argument(
        '-e',
        '--segment-end-pad',
        type=float,
        default=30.0,
        help='amount of time to remove from the end of each '
             'analysis segment',
    )
    parser.add_argument(
        '-L',
        '--bandpass-flow',
        type=float,
        default=4.0,
        help='lower-limit (Hz) of the passband for whitened '
             'BLRMS, default: %(default)s',
    )
    parser.add_argument(
        '-H',
        '--bandpass-fhigh',
        type=float,
        default=10.0,
        help='upper-limit (Hz) of the passband for whitened '
             'BLRMS, default: %(default)s',
    )
    parser.add_argument(
        '-S',
        '--sigma',
        type=float,
        default=2,
        help='significance threshold on whitened BLRMS, '
             'default: %(default)s',
    )
    parser.add_argument(
        '-c',
        '--main-channel',
        default='{IFO}:GDS-CALIB_STRAIN',
        help='name of main (h(t)) channel, default: %(default)s',
    )
    cli.add_frametype_option(
        parser,
        default='{IFO}_R',
    )
    parser.add_argument(
        '-o',
        '--output-dir',
        type=os.path.abspath,
        default=os.curdir,
        help='output directory for analysis, default: %(default)s',
    )
    parser.add_argument(
        '-w',
        '--omega-scans',
        type=int,
        metavar='NSCAN',
        default=0,
        help='generate a workflow of Omega scans for active '
             'scattering, this will launch automatically to '
             'condor, default: 0 (no Omega scans)',
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        default=False,
        help='log verbose output, default: %(default)s',
    )
    cli.add_nproc_option(
        parser,
    )

    # return the argument parser
    return parser


# -- main code block ----------------------------------------------------------

def main(args=None):
    """Run the primary scattering command-line tool
    """
    parser = create_parser()
    args = parser.parse_args(args=args)

    # set up logger
    logger = cli.logger(
        name=PROG.split('python -m ').pop(),
        level='DEBUG' if args.verbose else 'INFO',
    )

    # useful variables
    fthresh = (
        int(args.frequency_threshold) if args.frequency_threshold.is_integer()
        else args.frequency_threshold)
    multiplier = args.multiplier_for_threshold
    tstr = str(fthresh).replace('.', '_')
    gpsstr = '%s-%s' % (int(args.gpsstart), int(args.gpsend - args.gpsstart))
    args.optic = args.optic or list(OPTIC_MOTION_CHANNELS.keys())

    # go to working directory
    indir = os.getcwd()
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    os.chdir(args.output_dir)

    # set up output files
    summfile = '{}-SCATTERING_SUMMARY-{}.csv'.format(
        args.ifo, gpsstr)
    segfile = '{}-SCATTERING_SEGMENTS_{}_HZ-{}.h5'.format(
        args.ifo, tstr, gpsstr)

    # log start of process
    logger.info('{} Scattering {}-{}'.format(
        args.ifo, int(args.gpsstart), int(args.gpsend)))

    # -- get state segments -----------

    span = Segment(args.gpsstart, args.gpsend)

    # get segments
    if args.state_flag is not None:
        state = DataQualityFlag.query(
            args.state_flag, int(args.gpsstart), int(args.gpsend),
        ).coalesce()
        statea = []
        padding = args.segment_start_pad + args.segment_end_pad
        for i, seg in enumerate(state.active):
            if abs(seg) > padding:
                statea.append(Segment(
                    seg[0] + args.segment_start_pad,
                    seg[1] - args.segment_end_pad,
                ))
            else:
                logger.debug(
                    "Segment length {} shorter than padding length {}, "
                    "skipping segment {}-{}".format(abs(seg), padding, *seg),
                )
        statea = SegmentList(statea)
        logger.debug("Downloaded %d segments for %s"
                     % (len(statea), args.state_flag))
    else:
        statea = SegmentList([span])
    livetime = float(abs(statea))
    logger.debug("Processing %.2f s of livetime" % livetime)

    # -- load h(t) --------------------

    args.main_channel = args.main_channel.format(IFO=args.ifo)
    logger.debug("Loading Omicron triggers for %s" % args.main_channel)

    if args.gpsstart >= 1230336018:  # Jan 1 2019
        ext = "h5"
        names = ["time", "frequency", "snr"]
        read_kw = {
            "columns": names,
            "selection": [
                "{0} < frequency < {1}".format(
                    args.fmin, multiplier * fthresh),
                ("time", in_segmentlist, statea),
            ],
            "format": "hdf5",
            "path": "triggers",
        }
    else:
        ext = "xml.gz"
        names = ['peak', 'peak_frequency', 'snr']
        read_kw = {
            "columns": names,
            "selection": [
                "{0} < peak_frequency < {1}".format(
                    args.fmin, multiplier * fthresh),
                ('peak', in_segmentlist, statea),
            ],
            "format": 'ligolw',
            "tablename": "sngl_burst",
        }

    fullcache = []
    for seg in statea:
        cache = gwtrigfind.find_trigger_files(
            args.main_channel, 'omicron', seg[0], seg[1], ext=ext,
        )
        if len(cache) == 0:
            warnings.warn(
                "No Omicron triggers found for %s in segment [%d .. %d)"
                % (args.main_channel, seg[0], seg[1]),
            )
            continue
        fullcache.extend(cache)

    # read triggers
    if fullcache:
        trigs = EventTable.read(fullcache, nproc=args.nproc, **read_kw)
    else:  # no files (no livetime?)
        trigs = EventTable(names=names)

    highsnrtrigs = trigs[trigs['snr'] >= 8]
    logger.debug("%d read" % len(trigs))

    # -- prepare HTML -----------------

    links = [
        '%d-%d' % (int(args.gpsstart), int(args.gpsend)),
        ('Parameters', '#parameters'),
        ('Segments', (
            ('State flag', '#state-flag'),
            ('Optical sensors', '#osems'),
            ('Transmons', '#transmons'),
        )),
    ]
    if args.omega_scans:
        links.append(('Scans', '#omega-scans'))
    (brand, class_) = htmlio.get_brand(args.ifo, 'Scattering', args.gpsstart)
    navbar = htmlio.navbar(links, class_=class_, brand=brand)
    page = htmlio.new_bootstrap_page(
        title='%s Scattering | %d-%d' % (
            args.ifo, int(args.gpsstart), int(args.gpsend)),
        navbar=navbar)
    page.div(class_='pb-2 mt-3 mb-2 border-bottom')
    page.h1('%s Scattering: %d-%d'
            % (args.ifo, int(args.gpsstart), int(args.gpsend)))
    page.div.close()  # pb-2 mt-3 mb-2 border-bottom
    page.h2('Parameters', class_='mt-4 mb-4', id_='parameters')
    page.div(class_='row')
    page.div(class_='col-md-9 col-sm-12')
    page.add(htmlio.parameter_table(
        start=int(args.gpsstart), end=int(args.gpsend), flag=args.state_flag))
    page.div.close()  # col-md-9 col-sm-12

    # link to summary files
    page.div(class_='col-md-3 col-sm-12')
    page.add(htmlio.download_btn(
        [('Segments (HDF)', segfile),
         ('Triggers (CSV)', summfile)],
        btnclass='btn btn-%s dropdown-toggle' % args.ifo.lower(),
    ))
    page.div.close()  # col-md-3 col-sm-12
    page.div.close()  # row

    # command-line
    page.h5('Command-line:')
    page.add(htmlio.get_command_line(about=False, prog=PROG))

    # section header
    page.h2('Segments', class_='mt-4', id_='segments')

    if statea:  # contextual information
        paper = markup.oneliner.a(
            'Accadia et al. (2010)', target='_blank', class_='alert-link',
            href='http://iopscience.iop.org/article/10.1088/0264-9381/27'
                 '/19/194011')
        msg = (
            "Segments marked \"optical sensors\" below show evidence of beam "
            "scattering between {0} and {1} Hz based on the velocity of optic "
            "motion, with fringe frequencies projected using equation (3) of "
            "{2}. Segments marked \"transmons\" are based on whitened, "
            "band-limited RMS trends of transmon sensors. In both cases, "
            "yellow panels denote weak evidence for scattering, while red "
            "panels denote strong evidence."
         ).format(args.fmin, multiplier * fthresh, str(paper))
        page.add(htmlio.alert(msg, context=args.ifo.lower()))
    else:  # null segments
        page.add(htmlio.alert('No active analysis segments were found',
                              context='warning', dismiss=False))

    # record state segments
    if args.state_flag is not None:
        page.h3('State flag', class_='mt-3', id_='state-flag')
        page.div(id_='accordion1')
        page.add(htmlio.write_flag_html(
            state, span, 'state', parent='accordion1', context='success',
            plotdir='', facecolor=(0.2, 0.8, 0.2), edgecolor='darkgreen',
            known={'facecolor': 'red', 'edgecolor': 'darkred', 'height': 0.4}))
        page.div.close()

    # -- find scattering evidence -----

    # read data for OSEMs and transmons
    osems = ['%s:%s' % (args.ifo, c) for optic in args.optic for
             c in OPTIC_MOTION_CHANNELS[optic]]
    transmons = ['%s:%s' % (args.ifo, c) for c in TRANSMON_CHANNELS]
    allchannels = osems + transmons

    logger.info("Reading all timeseries data")
    alldata = []
    n = len(statea)
    for i, seg in enumerate(statea):
        msg = "{0}/{1} {2}:".rjust(30).format(
            str(i + 1).rjust(len(str(n))),
            n,
            str(seg),
        ) if args.verbose else False
        alldata.append(
            get_data(allchannels, seg[0], seg[1],
                     frametype=args.frametype.format(IFO=args.ifo),
                     verbose=msg, nproc=args.nproc).resample(128))
    try:  # ensure that only available channels are analyzed
        osems = list(
            set(alldata[0].keys()) & set(alldata[-1].keys()) & set(osems))
        transmons = list(
            set(alldata[0].keys()) & set(alldata[-1].keys()) & set(transmons))
    except IndexError:
        osems = []
        transmons = []

    # initialize scattering segments
    scatter_segments = DataQualityDict()
    actives = SegmentList()

    # scattering based on OSEM velocity
    if statea:
        page.h3('Optical sensors (OSEMs)', class_='mt-3', id_='osems')
        page.div(id_='osems-group')
    logger.info('Searching for scatter based on OSEM velocity')

    for i, channel in enumerate(sorted(osems)):
        logger.info("-- Processing %s --" % channel)
        chanstr = re.sub('[:-]', '_', channel).replace('_', '-', 1)
        optic = channel.split('-')[1].split('_')[0]
        flag = '%s:DCH-%s_SCATTERING_GE_%s_HZ:1' % (args.ifo, optic, tstr)
        scatter_segments[channel] = DataQualityFlag(
            flag,
            isgood=False,
            description="Evidence for scattering above {0} Hz from {1} in "
                        "{2}".format(fthresh, optic, channel),
        )
        # set up plot(s)
        plot = Plot(figsize=[12, 12])
        axes = {}
        axes['position'] = plot.add_subplot(
            411, xscale='auto-gps', xlabel='')
        axes['fringef'] = plot.add_subplot(
            412, sharex=axes['position'], xlabel='')
        axes['triggers'] = plot.add_subplot(
            413, sharex=axes['position'], xlabel='')
        axes['segments'] = plot.add_subplot(
            414, projection='segments', sharex=axes['position'])
        plot.subplots_adjust(bottom=.07, top=.95)
        fringecolors = [None] * len(FREQUENCY_MULTIPLIERS)
        histdata = dict((x, numpy.ndarray((0,))) for
                        x in FREQUENCY_MULTIPLIERS)
        linecolor = None
        # loop over state segments and find scattering fringes
        for j, seg in enumerate(statea):
            logger.debug("Processing segment [%d .. %d)" % seg)
            ts = alldata[j][channel]
            # get raw data and plot
            line = axes['position'].plot(ts, color=linecolor)[0]
            linecolor = line.get_color()
            # get fringe frequency and plot
            fringef = get_fringe_frequency(ts, multiplier=1)
            for k, m in list(enumerate(FREQUENCY_MULTIPLIERS))[::-1]:
                fm = fringef * m
                line = axes['fringef'].plot(
                    fm, color=fringecolors[k],
                    label=(j == 0 and r'$f\times%d$' % m or None))[0]
                fringecolors[k] = line.get_color()
                histdata[m] = numpy.resize(
                    histdata[m], (histdata[m].size + fm.size,))
                histdata[m][-fm.size:] = fm.value
            # get segments and plot
            scatter = get_segments(
                fringef * multiplier,
                fthresh,
                name=flag,
                pad=args.segment_padding
            )
            axes['segments'].plot(
                scatter, facecolor='red', edgecolor='darkred',
                known={'alpha': 0.6, 'facecolor': 'lightgray',
                       'edgecolor': 'gray', 'height': 0.4},
                height=0.8, y=0, label=' ',
            )
            scatter_segments[channel] += scatter
            logger.debug(
                "    Found %d scattering segments" % (len(scatter.active)))
        logger.debug("Completed channel %s, found %d segments in total"
                     % (channel, len(scatter_segments[channel].active)))

        # calculate efficiency and deadtime of veto
        deadtime = abs(scatter_segments[channel].active)
        try:
            deadtimepc = deadtime / livetime * 100
        except ZeroDivisionError:
            deadtimepc = 0.
        logger.info("Deadtime: %.2f%% (%.2f/%ds)"
                    % (deadtimepc, deadtime, livetime))
        efficiency = in_segmentlist(highsnrtrigs[names[0]],
                                    scatter_segments[channel].active).sum()
        try:
            efficiencypc = efficiency / len(highsnrtrigs) * 100
        except ZeroDivisionError:
            efficiencypc = 0.
        logger.info("Efficiency (SNR>=8): %.2f%% (%d/%d)"
                    % (efficiencypc, efficiency, len(highsnrtrigs)))
        if deadtimepc == 0.:
            effdt = 0
        else:
            effdt = efficiencypc/deadtimepc
        logger.info("Efficiency/Deadtime: %.2f" % effdt)

        if abs(scatter_segments[channel].active):
            actives.extend(scatter_segments[channel].active)

        # finalize plot
        logger.debug("Plotting")
        name = texify(channel)
        axes['position'].set_title("Scattering evidence in %s" % name)
        axes['position'].set_xlabel('')
        axes['position'].set_ylabel(r'Position [$\mu$m]')
        axes['position'].text(
            0.01, 0.95, 'Optic position',
            transform=axes['position'].transAxes, va='top', ha='left',
            bbox={'edgecolor': 'none', 'facecolor': 'white', 'alpha': .5})
        axes['fringef'].plot(
            span, [fthresh, fthresh], 'k--')
        axes['fringef'].set_xlabel('')
        axes['fringef'].set_ylabel(r'Frequency [Hz]')
        axes['fringef'].yaxis.tick_right()
        axes['fringef'].yaxis.set_label_position("right")
        axes['fringef'].set_ylim(0, multiplier * fthresh)
        axes['fringef'].text(
            0.01, 0.95, 'Calculated fringe frequency',
            transform=axes['fringef'].transAxes, va='top', ha='left',
            bbox={'edgecolor': 'none', 'facecolor': 'white', 'alpha': .5})
        handles, labels = axes['fringef'].get_legend_handles_labels()
        axes['fringef'].legend(handles[::-1], labels[::-1], loc='upper right',
                               borderaxespad=0, bbox_to_anchor=(-0.01, 1.),
                               handlelength=1)

        axes['triggers'].scatter(
            trigs[names[0]],
            trigs[names[1]],
            c=trigs[names[2]],
            edgecolor='none',
        )
        name = texify(args.main_channel)
        axes['triggers'].text(
            0.01, 0.95,
            '%s event triggers (Omicron)' % name,
            transform=axes['triggers'].transAxes, va='top', ha='left',
            bbox={'edgecolor': 'none', 'facecolor': 'white', 'alpha': .5})
        axes['triggers'].set_ylabel('Frequency [Hz]')
        axes['triggers'].set_ylim(args.fmin, multiplier * fthresh)
        axes['triggers'].colorbar(cmap='YlGnBu', clim=(3, 100), norm='log',
                                  label='Signal-to-noise ratio')
        axes['segments'].set_ylim(-.55, .55)
        axes['segments'].text(
            0.01, 0.95,
            r'Time segments with $f\times%d > %.2f$ Hz' % (
                multiplier, fthresh),
            transform=axes['segments'].transAxes, va='top', ha='left',
            bbox={'edgecolor': 'none', 'facecolor': 'white', 'alpha': .5})
        for ax in axes.values():
            ax.set_epoch(int(args.gpsstart))
            ax.set_xlim(*span)
        png = '%s_SCATTERING_%s_HZ-%s.png' % (chanstr, tstr, gpsstr)
        try:
            plot.save(png)
        except OverflowError as e:
            warnings.warn(str(e))
            plot.axes[1].set_ylim(0, multiplier * fthresh)
            plot.refresh()
            plot.save(png)
        plot.close()
        logger.debug("%s written." % png)

        # make histogram
        histogram = Plot(figsize=[12, 6])
        ax = histogram.gca()
        hrange = (0, multiplier * fthresh)
        for m, color in list(zip(histdata, fringecolors))[::-1]:
            if histdata[m].size:
                ax.hist(
                    histdata[m], facecolor=color, alpha=.6, range=hrange,
                    bins=50, histtype='stepfilled', label=r'$f\times%d$' % m,
                    cumulative=-1, weights=ts.dx.value, bottom=1e-100,
                    log=True)
            else:
                ax.plot(histdata[m], color=color, label=r'$f\times%d$' % m)
                ax.set_yscale('log')
        ax.set_ylim(.01, float(livetime))
        ax.set_ylabel('Time with fringe above frequency [s]')
        ax.set_xlim(*hrange)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_title(axes['position'].get_title())
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='upper right')
        hpng = '%s_SCATTERING_HISTOGRAM-%s.png' % (chanstr, gpsstr)
        histogram.save(hpng)
        histogram.close()
        logger.debug("%s written." % hpng)

        # write HTML
        if deadtime != 0 and effdt > 2:
            context = 'danger'
        elif ((deadtime != 0 and effdt < 2) or
              (histdata[multiplier].size and
               histdata[multiplier].max() >=
                  fthresh/2.)):
            context = 'warning'
        else:
            continue
        page.div(class_='card border-%s mb-1 shadow-sm' % context)
        page.div(class_='card-header text-white bg-%s' % context)
        page.a(channel, class_='collapsed card-link cis-link',
               href='#osem%s' % i, **{'data-bs-toggle': 'collapse'})
        page.div.close()  # card-header
        page.div(id_='osem%s' % i, class_='collapse',
                 **{'data-parent': '#osems-group'})
        page.div(class_='card-body')
        page.div(class_='row')
        img = htmlio.FancyPlot(
            png, caption=SCATTER_CAPTION.format(CHANNEL=channel))
        page.div(class_='col-md-10 offset-md-1')
        page.add(htmlio.fancybox_img(img))
        page.div.close()  # col-md-10 offset-md-1
        himg = htmlio.FancyPlot(
            hpng, caption=HIST_CAPTION.format(CHANNEL=channel))
        page.div(class_='col-md-10 offset-md-1')
        page.add(htmlio.fancybox_img(himg))
        page.div.close()  # col-md-10 offset-md-1
        page.div.close()  # row
        segs = StringIO()
        if deadtime:
            page.p("%d segments were found predicting a scattering fringe "
                   "above %.2f Hz." % (
                       len(scatter_segments[channel].active),
                       fthresh))
            page.table(class_='table table-sm table-hover')
            page.tbody()
            page.tr()
            page.th('Deadtime')
            page.td('%.2f/%d seconds' % (deadtime, livetime))
            page.td('%.2f%%' % deadtimepc)
            page.tr.close()
            page.tr()
            page.th('Efficiency<br><small>(SNR&ge;8 and '
                    '%.2f Hz</sub>&ltf<sub>peak</sub>&lt;%.2f Hz)</small>'
                    % (args.fmin, multiplier * fthresh))
            page.td('%d/%d events' % (efficiency, len(highsnrtrigs)))
            page.td('%.2f%%' % efficiencypc)
            page.tr.close()
            page.tr()
            page.th('Efficiency/Deadtime')
            page.td()
            page.td('%.2f' % effdt)
            page.tr.close()
            page.tbody.close()
            page.table.close()
            scatter_segments[channel].active.write(segs, format='segwizard',
                                                   coltype=float)
            page.pre(segs.getvalue())
        else:
            page.p("No segments were found with scattering above %.2f Hz."
                   % fthresh)
        page.div.close()  # card-body
        page.div.close()  # collapse
        page.div.close()  # card

    if statea:  # close accordion
        page.div.close()  # osems-group

    # scattering based on transmon BLRMS
    if statea:
        page.h3('Transmons', class_='mt-3', id_='transmons')
        page.div(id_='transmons-group')
    logger.info('Searching for scatter based on band-limited RMS of transmons')

    for i, channel in enumerate(sorted(transmons)):
        logger.info("-- Processing %s --" % channel)
        optic = channel.split('-')[1][:6]
        flag = '%s:DCH-%s_SCATTERING_BLRMS:1' % (args.ifo, optic)
        scatter_segments[channel] = DataQualityFlag(
            flag,
            isgood=False,
            description="Evidence for scattering from whitened, band-limited "
                        "RMS trends of {0}".format(channel),
        )

        # loop over state segments and compute BLRMS
        for j, seg in enumerate(statea):
            logger.debug("Processing segment [%d .. %d)" % seg)
            wblrms = get_blrms(
                alldata[j][channel],
                flow=args.bandpass_flow,
                fhigh=args.bandpass_fhigh,
            )
            scatter = get_segments(
                wblrms,
                numpy.mean(wblrms) + args.sigma * numpy.std(wblrms),
                name=flag,
            )
            scatter_segments[channel] += scatter
            logger.debug(
                "    Found %d scattering segments" % (len(scatter.active)))
        logger.debug("Completed channel %s, found %d segments in total"
                     % (channel, len(scatter_segments[channel].active)))

        # calculate efficiency and deadtime of veto
        deadtime = abs(scatter_segments[channel].active)
        try:
            deadtimepc = deadtime / livetime * 100
        except ZeroDivisionError:
            deadtimepc = 0.
        logger.info("Deadtime: %.2f%% (%.2f/%ds)"
                    % (deadtimepc, deadtime, livetime))
        highsnrtrigs = trigs[trigs['snr'] <= 200]
        efficiency = in_segmentlist(highsnrtrigs[names[0]],
                                    scatter_segments[channel].active).sum()
        try:
            efficiencypc = efficiency / len(highsnrtrigs) * 100
        except ZeroDivisionError:
            efficiencypc = 0.
        logger.info("Efficiency (SNR>=8): %.2f%% (%d/%d)"
                    % (efficiencypc, efficiency, len(highsnrtrigs)))
        if deadtimepc == 0.:
            effdt = 0
        else:
            effdt = efficiencypc/deadtimepc
        logger.info("Efficiency/Deadtime: %.2f" % effdt)

        if abs(scatter_segments[channel].active):
            actives.extend(scatter_segments[channel].active)

        # write HTML
        if deadtime != 0 and effdt > 2:
            context = 'danger'
        elif deadtime != 0 and effdt < 2:
            context = 'warning'
        else:
            continue
        page.add(htmlio.write_flag_html(
            scatter_segments[channel], span, i, parent='transmons-group',
            title=channel, context=context, plotdir=''))

    if statea:  # close accordion
        page.div.close()  # transmons-group

    actives = actives.coalesce()  # merge contiguous segments
    if statea and not actives:
        page.add(htmlio.alert(
            'No evidence of scattering found in the channels analyzed',
            context=args.ifo.lower(), dismiss=False))

    # identify triggers during active segments
    logger.debug('Writing a summary CSV record')
    ind = [i for i, trigtime in enumerate(highsnrtrigs[names[0]])
           if trigtime in actives]
    gps = highsnrtrigs[names[0]][ind]
    freq = highsnrtrigs[names[1]][ind]
    snr = highsnrtrigs[names[2]][ind]
    segs = [y for x in gps for y in actives if x in y]
    table = EventTable(
        [gps, freq, snr, [seg[0] for seg in segs], [seg[1] for seg in segs]],
        names=('trigger_time', 'trigger_frequency', 'trigger_snr',
               'segment_start', 'segment_end'))
    logger.info('The following {} triggers fell within active scattering '
                'segments:\n\n'.format(len(table)))
    print(table)
    print('\n\n')
    table.write(summfile, overwrite=True)

    # -- launch omega scans -----------

    nscans = min(args.omega_scans, len(table))
    if nscans > 0:
        # launch scans
        scandir = 'scans'
        ind = random.sample(range(0, len(table)), nscans)
        omegatimes = [str(t) for t in table['trigger_time'][ind]]
        logger.debug('Collected {} event times to omega scan: {}'.format(
            nscans, ', '.join(omegatimes)))
        logger.info('Creating workflow for omega scans')
        flags = batch.get_command_line_flags(
            ifo=args.ifo, ignore_state_flags=True)
        condorcmds = batch.get_condor_arguments(timeout=4, gps=args.gpsstart)
        batch.generate_dag(omegatimes, flags=flags, submit=True,
                           outdir=scandir, condor_commands=condorcmds)
        logger.info('Launched {} omega scans to condor'.format(nscans))
        # render HTML
        page.h2('Omega scans', class_='mt-4', id_='omega-scans')
        msg = (
            'The following event times correspond to significant Omicron '
            'triggers that occur during the scattering segments found above. '
            'To compare these against fringe frequency projections, please '
            'use the "simple scattering" module:',
            markup.oneliner.pre(
                '$ python -m gwdetchar.scattering.simple --help',
            ),
        )
        page.add(htmlio.alert(msg, context=args.ifo.lower()))
        page.add(htmlio.scaffold_omega_scans(
            omegatimes, args.main_channel, scandir=scandir))
    elif args.omega_scans:
        logger.info('No events found during active scattering segments')

    # -- finalize ---------------------

    # write segments
    scatter_segments.write(segfile, path="segments", overwrite=True)
    logger.debug("%s written" % segfile)

    # write HTML
    htmlio.close_page(page, 'index.html')
    logger.info("-- index.html written, all done --")

    # return to original directory
    os.chdir(indir)


# -- run from command-line ----------------------------------------------------

if __name__ == '__main__':
    main()
