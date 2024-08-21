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

import fnmatch
import multiprocessing
import os
import re
import sys
from getpass import getuser

import numpy

from MarkupPy import markup

from astropy.table import Table

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler, scale

from pandas import set_option
from gwpy.io.datafind import find_urls
from gwpy.io.gwf import get_channel_names
from gwpy.segments import (
    Segment,
    SegmentList,
    DataQualityFlag,
)
from gwpy.time import tconvert
from gwpy.timeseries import TimeSeries

from .. import (cli, lasso as gwlasso)
from ..io.datafind import get_data
from ..io import html as htmlio

from matplotlib import (use, rcParams)
use('Agg')

# backend-dependent imports
from matplotlib.cm import get_cmap  # noqa: E402
from gwpy.plot import Plot  # noqa: E402
from gwdetchar.lasso import plot as gwplot  # noqa: E402
from gwdetchar.plot import texify  # noqa: E402

# LaTeX use
USETEX = rcParams["text.usetex"]

# default frametypes
DEFAULT_FRAMETYPE = {
    'GDS-CALIB_STRAIN': '{ifo}_HOFT_C00',
    'DMT-SNSH_EFFECTIVE_RANGE_MPC.mean': 'SenseMonitor_hoft_{ifo}_M',
    'DMT-SNSW_EFFECTIVE_RANGE_MPC.mean': 'SenseMonitor_CAL_{ifo}_M',
}

# set up logger
PROG = ('python -m gwdetchar.lasso' if sys.argv[0].endswith('.py')
        else os.path.basename(sys.argv[0]))
LOGGER = cli.logger(name=PROG.split('python -m ').pop())


# -- utilities ----------------------------------------------------------------

def _descaler(iterable, *coef):
    """Linear de-scaling of a data array
    """
    return (
        [((x * primary_std * coef[0]) + primary_mean) for x in iterable]
        if coef else [((x * primary_std) + primary_mean) for x in iterable]
    )


def _generate_cluster(input_):
    """Generate cluster data for use below
    """
    if USETEX:
        gwplot.configure_mpl_tex()
    currentchan = input_[1][0]
    currentts = input_[1][5]
    current = input_[0]
    p7 = (.135, .15, .95, .9)
    plot7 = None
    plot7_list = None

    if current < len(nonzerodata):
        cluster = []
        for i, otheritem in enumerate(list(auxdata.items())):
            chan_, ts_ = otheritem
            if chan_ != currentchan:
                pcorr = numpy.corrcoef(currentts.value, ts_.value)[0, 1]
                if abs(pcorr) >= cluster_threshold:
                    stub = re_delim.sub('_', chan_).replace('_', '-', 1)
                    cluster.append([i, ts_, pcorr, chan_, stub])

        if cluster:
            # write cluster table to file
            cluster = sorted(cluster, key=lambda x: abs(x[2]),
                             reverse=True)
            clustertab = Table(data=list(zip(*cluster))[2:4],
                               names=('Pearson Coefficient', 'Channel'))
            plot7_list = '%s_CLUSTER_LIST-%s.csv' % (
                re_delim.sub('_', str(currentchan)).replace('_', '-', 1),
                gpsstub)
            clustertab.write(plot7_list, format='csv', overwrite=True)

            ncluster = min(len(cluster), max_correlated_channels)
            colors2 = [cmap(i) for i in numpy.linspace(0, 1, ncluster+1)]

            # plot
            fig = Plot(figsize=(12, 4))
            fig.subplots_adjust(*p7)
            ax = fig.gca()
            ax.set_xscale('auto-gps')
            ax.plot(
                times, scale(currentts.value)*numpy.sign(input_[1][1]),
                label=texify(currentchan), linewidth=line_size_aux,
                color=colors[0])

            for i in range(0, ncluster):
                this = cluster[i]
                ax.plot(
                    times,
                    scale(this[1].value) * numpy.sign(input_[1][1]) *
                    numpy.sign(this[2]),
                    color=colors2[i+1],
                    linewidth=line_size_aux,
                    label=('{0}, r = {1:.2}'.format(
                        texify(cluster[i][3]), cluster[i][2])),
                )

            ax.margins(x=0)
            ax.set_ylabel('Scaled amplitude [arbitrary units]')
            ax.set_title('Highly Correlated Channels')
            ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
            plot7 = gwplot.save_figure(
                fig,
                os.path.join('%s_CLUSTER-%s.png' % (
                    re_delim.sub('_', str(currentchan)).replace('_', '-', 1),
                    gpsstub)), bbox_inches='tight')

    with counter.get_lock():
        counter.value += 1
        pc = 100 * counter.value / len(nonzerodata)
        LOGGER.info("Completed [%d/%d] %3d%% %-50s"
                    % (counter.value, len(nonzerodata), pc,
                       '(%s)' % str(currentchan)))
        sys.stdout.flush()
    return (plot7, plot7_list)


def _process_channel(input_):
    """Handle individual channels for multiprocessing
    """
    if USETEX:
        gwplot.configure_mpl_tex()
    p4 = (.1, .1, .9, .95)
    chan = input_[1][0]
    ts = input_[1][1]
    lassocoef = nonzerocoef[chan]
    zeroed = lassocoef == 0

    if zeroed:
        plot4 = None
        plot5 = None
        plot6 = None
        pcorr = None
    else:
        plot4 = None
        plot5 = None
        plot6 = None
        if trend_type == 'minute':
            pcorr = numpy.corrcoef(ts.value, primaryts.value)[0, 1]
        else:
            pcorr = 0.0
        if abs(lassocoef) < threshold:
            with counter.get_lock():
                counter.value += 1
            pc = 100 * counter.value / len(nonzerodata)
            LOGGER.info("Completed [%d/%d] %3d%% %-50s"
                        % (counter.value, len(nonzerodata), pc,
                           '(%s)' % str(chan)))
            sys.stdout.flush()
            return (chan, lassocoef, plot4, plot5, plot6, ts)

        # create time series subplots
        fig = Plot(figsize=(12, 8))
        fig.subplots_adjust(*p4)
        ax1 = fig.add_subplot(2, 1, 1, xscale='auto-gps', epoch=start)
        ax1.plot(primaryts, label=texify(primary), color='black',
                 linewidth=line_size_primary)
        ax1.set_xlabel(None)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1, xlim=xlim)
        ax2.plot(ts, label=texify(chan), linewidth=line_size_aux)
        if range_is_primary:
            ax1.set_ylabel('Sensitive range [Mpc]')
        else:
            ax1.set_ylabel('Primary channel units')
        ax2.set_ylabel('Channel units')
        for ax in fig.axes:
            ax.legend(loc='best')
        channelstub = re_delim.sub('_', str(chan)).replace('_', '-', 1)
        plot4 = gwplot.save_figure(
            fig,
            f'{channelstub}_TRENDS-{gpsstub}.png',
            bbox_inches='tight')

        # create scaled, sign-corrected, and overlayed timeseries
        tsscaled = scale(ts.value)
        if lassocoef < 0:
            tsscaled = numpy.negative(tsscaled)
        fig = Plot(figsize=(12, 4))
        fig.subplots_adjust(*p1)
        ax = fig.gca()
        ax.set_xscale('auto-gps')
        ax.set_epoch(start)
        ax.set_xlim(xlim)
        ax.plot(times, _descaler(target), label=texify(primary),
                color='black', linewidth=line_size_primary)
        ax.plot(times, _descaler(tsscaled), label=texify(chan),
                linewidth=line_size_aux)
        if range_is_primary:
            ax.set_ylabel('Sensitive range [Mpc]')
        else:
            ax.set_ylabel('Primary Channel Units')
        ax.legend(loc='best')
        plot5 = gwplot.save_figure(
            fig,
            f'{channelstub}_COMPARISON-{gpsstub}.png',
            bbox_inches='tight')

        # scatter plot
        tsCopy = ts.value.reshape(-1, 1)
        primarytsCopy = primaryts.value.reshape(-1, 1)
        primaryReg = linear_model.LinearRegression()
        primaryReg.fit(tsCopy, primarytsCopy)
        primaryFit = primaryReg.predict(tsCopy)
        fig = Plot(figsize=(12, 4))
        fig.subplots_adjust(*p1)
        ax = fig.gca()
        ax.set_xlabel(texify(chan) + ' [Channel units]')
        if range_is_primary:
            ax.set_ylabel('Sensitive range [Mpc]')
        else:
            ax.set_ylabel('Primary channel units')
        y_min = min(primaryts.value)
        y_max = max(primaryts.value)
        y_range = y_max - y_min
        ax.set_ylim(y_min - (y_range * 0.1), y_max + (y_range * 0.1))
        ax.text(.9, .1, 'r = ' + str('{0:.2}'.format(pcorr)),
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes, color='black', size=20,
                bbox=dict(boxstyle='square', facecolor='white', alpha=.75,
                          edgecolor='black'))
        ax.scatter(ts.value, primaryts.value, color='red')
        ax.plot(ts.value, primaryFit, color='black')
        ax.autoscale_view(tight=False, scalex=True, scaley=True)
        plot6 = gwplot.save_figure(
            fig,
            f'{channelstub}_SCATTER-{gpsstub}.png',
            bbox_inches='tight')

    # increment counter and print status
    with counter.get_lock():
        counter.value += 1
        pc = 100 * counter.value / len(nonzerodata)
        LOGGER.info("Completed [%d/%d] %3d%% %-50s"
                    % (counter.value, len(nonzerodata), pc,
                       '(%s)' % str(chan)))
        sys.stdout.flush()
    return (chan, lassocoef, plot4, plot5, plot6, ts)


def get_primary_ts(channel, start, end, filepath=None,
                   frametype=None, cache=None, nproc=1):
    """Retrieve primary channel timeseries
    by either reading a .gwf file or querying
    """
    if filepath is not None:
        LOGGER.info('Reading primary channel file')
        return TimeSeries.read(filepath, channel=channel, start=start, end=end,
                               format='gwf', nproc=nproc)
    else:
        LOGGER.info('Querying primary channel')
        return get_data(channel, start, end,
                        verbose='Reading primary:'.rjust(30),
                        frametype=frametype, source=cache, nproc=nproc,
                        on_gaps='error')


# -- parse command line -------------------------------------------------------

def create_parser():
    """Create a command-line parser for this entry point
    """
    # initialize argument parser
    parser = cli.create_parser(
        prog=PROG,
        description=__doc__,
        formatter_class=cli.argparse.ArgumentDefaultsHelpFormatter,
    )
    psig = parser.add_argument_group('Signal processing options')
    lsig = parser.add_argument_group('Lasso options')
    segm = parser.add_argument_group('Segment processing options')

    # required arguments
    cli.add_gps_start_stop_arguments(parser)
    cli.add_ifo_option(parser)

    # optional arguments
    cli.add_nproc_option(parser, default=1)
    parser.add_argument(
        '-J',
        '--nproc-plot',
        type=int,
        default=None,
        help='number of processes to use for plot rendering, '
             'will be ignored if not using LaTeX',
    )
    parser.add_argument(
        '-o',
        '--output-dir',
        default=os.curdir,
        type=os.path.abspath,
        help='output directory for plots',
    )
    parser.add_argument(
        '-s',
        '--summary-dir',
        default=os.curdir,
        type=os.path.abspath,
        help='output directory for html summary pages'
    )
    parser.add_argument(
        '-f',
        '--channel-file',
        type=os.path.abspath,
        help='path for channel file',
    )
    parser.add_argument(
        '-T',
        '--trend-type',
        default='minute',
        choices=['second', 'minute'],
        help='type of trend for correlation',
    )
    parser.add_argument(
        '-p',
        '--primary-channel',
        default='{ifo}:DMT-SNSH_EFFECTIVE_RANGE_MPC.mean',
        help='name of primary channel to use',
    )
    # primary channel filepath argument
    parser.add_argument(
        '-pf',
        '--primary-file',
        default=None,
        help='filepath of .gwf custom primary channel if using custom channel'
    )
    parser.add_argument(
        '-P',
        '--primary-frametype',
        default=None,
        help='frametype for --primary-channel, default: guess by channel name',
    )
    parser.add_argument(
        '--primary-cache',
        default=None,
        help='cache file for --primary-channel, default: None',
    )
    parser.add_argument(
        '-O',
        '--remove-outliers',
        type=float,
        default=None,
        help='Std. dev. limit for removing outliers',
    )
    parser.add_argument(
        '-R',
        '--remove-outliers-pf',
        type=float,
        default=None,
        help='Fractional limit for removing outliers between 0 and 1',
    )
    parser.add_argument(
        '-t',
        '--threshold',
        type=float,
        default=0.0001,
        help='threshold for making a plot',
    )
    parser.add_argument(
        '--remove-bad-chans',
        action='store_true',
        default=False,
        help='remove flat/bad channels',
    )

    # signal processing arguments
    psig.add_argument(
        '-b',
        '--band-pass',
        type=float,
        nargs=2,
        default=None,
        metavar="FLOW FHIGH",
        help='lower and upper frequencies for bandpass on h(t)',
    )
    psig.add_argument(
        '-x',
        '--filter-padding',
        type=float,
        default=3.,
        help='amount of time (seconds) to pad data for filtering',
    )

    # lasso arguments
    lsig.add_argument(
        '-a',
        '--alpha',
        default=None,
        type=float,
        help='alpha parameter for lasso fit',
    )
    lsig.add_argument(
        '-C',
        '--no-cluster',
        action='store_true',
        default=False,
        help='do not generate clustered channel plots',
    )
    lsig.add_argument(
        '-c',
        '--cluster-coefficient',
        default=.85,
        type=float,
        help='correlation coefficient threshold for clustering',
    )
    lsig.add_argument(
        '-L',
        '--line-size-primary',
        default=1,
        type=float,
        help='line width of primary channel',
    )
    lsig.add_argument(
        '-l',
        '--line-size-aux',
        default=0.75,
        type=float,
        help='line width of auxilary channel',
    )

    # segment options
    segm.add_argument(
        '--low-noise-state-flag',
        action='store_true',
        default=False,
        help='use low noise data quality flag for segments'
    )
    segm.add_argument(
        '--segment-padding',
        type=int,
        default=1200,
        help=('padding of time on either side of a data-quality segment to '
              'remove')
    )
    segm.add_argument(
        '--segment-min-length',
        default=10800,
        type=int,
        help='data-quality segments must be at least this many seconds'
    )
    segm.add_argument(
        '--intersect-data-segs',
        action='store_true',
        default=False,
        help='intersect data quality segments with available data'
    )

    # return the argument parser
    return parser


# -- main code block ----------------------------------------------------------

def main(args=None):
    """Run the lasso command-line interface
    """
    # declare global variables
    # this is needed for multiprocessing utilities
    global auxdata, cluster_threshold, cmap, colors, counter, gpsstub
    global line_size_aux, line_size_primary, max_correlated_channels
    global nonzerocoef, nonzerodata, p1, primary, primary_mean, primary_std
    global primaryts, range_is_primary, re_delim, start, target, times
    global threshold, trend_type, xlim
    parser = create_parser()
    args = parser.parse_args(args=args)

    # check for correct input
    if args.remove_outliers_pf:
        if args.remove_outliers_pf >= 1 or args.remove_outliers_pf <= 0:
            raise ValueError('Percent outlier limit must be between 0 and 1')

    # get run params
    start = int(args.gpsstart)
    end = int(args.gpsend)

    # set pertinent global variables
    cluster_threshold = args.cluster_coefficient
    line_size_aux = args.line_size_aux
    line_size_primary = args.line_size_primary
    threshold = args.threshold
    trend_type = args.trend_type

    # let's go
    LOGGER.info(f'{args.ifo} Lasso correlations {start}-{end}')

    # get primary channel frametype
    primary = args.primary_channel.format(ifo=args.ifo)
    range_is_primary = 'EFFECTIVE_RANGE_MPC' in args.primary_channel
    if args.primary_cache is not None:
        LOGGER.info("Using custom primary cache file")
    elif args.primary_frametype is None and args.primary_file is None:
        try:
            args.primary_frametype = DEFAULT_FRAMETYPE[
                primary.split(':')[1]].format(ifo=args.ifo)
        except KeyError as exc:
            raise type(exc)("Could not determine primary channel's frametype, "
                            "please specify with --primary-frametype")

    # Get auxiliary frametype
    if args.trend_type == 'minute':
        aux_frametype = f'{args.ifo}_M'  # for minute trends
    else:
        aux_frametype = f'{args.ifo}_T'  # for second trends

    # Remove channels regexp
    to_remove = ['.*\.n$', '.*.min$', '.*.max$', '.*.rms$']  # noqa: W605
    if range_is_primary:
        to_remove.extend([
            'OAF.*RANGE', 'SENSEMON', 'SENSMON',
            'CAL-CS_DARM', ':SUS-.*TM.*MODE[0-9]',
            'SQZ-DCPD_', 'CAL-CS_TDEP_*', 'PCAL.*COHERENCE',
            'ASC-ADS'])
    remove_regex = re.compile("({})".format("|".join(to_remove)))

    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # multiprocessing for plots
    nprocplot = (args.nproc_plot or args.nproc) if USETEX else 1

    # Get segments
    if args.low_noise_state_flag:
        state_flag = f'{args.ifo}:DMT-GRD_ISC_LOCK_NOMINAL:1'
        state_is_observing = False
        state_name = 'Locked'
    else:
        state_flag = f'{args.ifo}:DMT-ANALYSIS_READY:1'
        state_is_observing = True
        state_name = 'Observing'
    state = DataQualityFlag.query(state_flag, start, end)
    state.name = state_name
    segs = state.active.coalesce()

    # If requested, find data and check if available, then intersect with
    # segs
    if args.intersect_data_segs:
        # First the primary channel frametype: make a list of segments
        # that the data URLs cover. Join segments that are contiguous.
        primary_segs = SegmentList([])
        for i, seg in enumerate(segs):
            # on_gaps='warn' to avoid problems if there is a missing frame
            cache = find_urls(args.ifo[0], args.primary_frametype,
                              seg[0], seg[1], on_gaps='warn')
            for url in cache:
                filedata = os.path.splitext(
                    os.path.basename(url))[0].split('-')
                filestart = int(filedata[-2])
                fileend = filestart + int(filedata[-1])
                primary_segs.append(Segment(filestart, fileend))
        primary_segs.coalesce()

        # Same as above, but this time for the auxiliary frametype
        auxiliary_segs = SegmentList([])
        for i, seg in enumerate(segs):
            # on_gaps='warn' to avoid problems if there is a missing frame
            cache = find_urls(args.ifo[0], aux_frametype,
                              seg[0], seg[1], on_gaps='warn')
            for url in cache:
                filedata = os.path.splitext(
                    os.path.basename(url))[0].split('-')
                filestart = int(filedata[-2])
                fileend = filestart + int(filedata[-1])
                auxiliary_segs.append(Segment(filestart, fileend))
        auxiliary_segs.coalesce()

        data_segs = primary_segs & auxiliary_segs
        data_segs &= segs  # intersect with state segments
        data_segs.coalesce()  # just in case
    else:
        data_segs = segs

    # Loop over segments to create the "lasso segments", a perhaps slightly
    # reduced length of segments because of the segment_padding argument and
    # minimum length required
    # Note that if intersect-data-segs was requested, the code below will
    # implicitly do the intersection because data_segs are the segments of
    # available frames
    lasso_segs = SegmentList([])
    for i, seg in enumerate(data_segs):
        if abs(seg) >= args.segment_min_length:
            padded_seg = seg.contract(args.segment_padding)
            lasso_segs.append(padded_seg)
    lasso_segs.coalesce()  # just in case

    # Loop over lasso segments
    files = []
    gpsstubs = []
    gpsblocks = []
    subdirs = []
    for i, seg in enumerate(lasso_segs):
        gpsstub = f'{int(seg[0])}-{int(seg[1]-seg[0])}'
        gpsblock = f'{int(seg[0])}-{int(seg[1])}'
        subdir = os.path.join(args.output_dir, gpsblock)
        gpsstubs.append(gpsstub)
        gpsblocks.append(gpsblock)
        subdirs.append(subdir)

        os.makedirs(subdir, exist_ok=True)
        os.chdir(subdir)

        # Get auxiliary frames if there was no channel list provided
        # Get the channels of interest, removing repeats
        # Remove channels that are "bad" or "flat"
        # Save channels used
        if args.channel_file is None:
            # here we use the on_gaps='error' because we really shouldn't have
            # any more missing data at this point
            cache = find_urls(args.ifo[0], aux_frametype, seg[0], seg[1],
                              on_gaps='error')
            aux_channels = get_channel_names(cache[0])
            aux_channels = list(filter(
                lambda x: not remove_regex.search(x),
                aux_channels,
            ))
            if args.remove_bad_chans:
                LOGGER.debug(
                    f"Removing flat/bad channels in segment {gpsblock}")
                data = get_data(aux_channels, int(seg[0]), int(seg[0]) + 600,
                                frametype=aux_frametype, nproc=args.nproc,
                                pad=0, on_gaps='error')
                data = gwlasso.remove_flat(data)
                data = gwlasso.remove_bad(data)
                aux_channels = list(data.keys())
                filename = os.path.join(
                    subdir, f'{args.ifo}-CHANNELS-{gpsstub}.txt')
                files.append(filename)
                numpy.savetxt(filename, aux_channels, fmt='%s')
        else:
            with open(args.channel_file, 'r') as f:
                aux_channels = [name.rstrip('\n') for name in f]

        """ Run lasso over segment """

        # Get primary data and optionally bandpass
        if args.band_pass:
            try:
                flower, fupper = args.band_pass
            except TypeError:
                flower, fupper = None

            LOGGER.info("-- Loading primary channel data")
            bandts = get_primary_ts(channel=primary,
                                    start=int(seg[0]-args.filter_padding),
                                    end=int(seg[1]+args.filter_padding),
                                    filepath=args.primary_file,
                                    frametype=args.primary_frametype,
                                    cache=args.primary_cache,
                                    nproc=args.nproc)
            if flower < 0 or fupper >= float((bandts.sample_rate/2.).value):
                raise ValueError(
                    "bandpass frequency is out of range for this "
                    f"channel, band (Hz): {args.band_pass}, sample rate: "
                    f"{bandts.sample_rate}")

            # get darm BLRMS
            LOGGER.debug("-- Filtering data")
            if trend_type == 'minute':
                stride = 60
            else:
                stride = 1
            if flower:
                darmbl = (
                    bandts.highpass(flower/2., fstop=flower/4.,
                                    filtfilt=False, ftype='butter')
                    .notch(60, filtfilt=False)
                    .bandpass(flower, fupper, fstop=[flower/2., fupper*1.5],
                              filtfilt=False, ftype='butter')
                    .crop(int(seg[0]), int(seg[1])))
                darmblrms = darmbl.rms(stride)
                darmblrms.name = f'{primary} {flower}-{fupper} Hz BLRMS'
            else:
                darmbl = bandts.notch(60).crop(int(seg[0]), int(seg[1]))
                darmblrms = darmbl.rms(stride)
                darmblrms.name = f'{primary} RMS'

            primaryts = darmblrms

            bandts_asd = bandts.asd(4, 2, method='median')
            darmbl_asd = darmbl.asd(4, 2, method='median')

            spectrum_plots = gwplot.make_spectrum_plots(
                int(seg[0]), int(seg[1]), flower, fupper, primary,
                bandts_asd, darmbl_asd)
            spectrum_plot_zoomed_out = spectrum_plots[0]
            spectrum_plot_zoomed_in = spectrum_plots[1]
        else:
            # load primary channel data
            LOGGER.info("-- Loading primary channel data")
            primaryts = get_primary_ts(
                channel=primary,
                start=int(seg[0]),
                end=int(seg[1]),
                filepath=args.primary_file,
                frametype=args.primary_frametype,
                cache=args.primary_cache,
                nproc=args.nproc)

        if args.remove_outliers:
            LOGGER.debug(
                f"-- Removing outliers above {args.remove_outliers} sigma")
            gwlasso.remove_outliers(primaryts, args.remove_outliers)
        elif args.remove_outliers_pf:
            LOGGER.debug("-- Removing outliers in the bottom "
                         f"{args.remove_outliers_pf*100} percent of data")
            gwlasso.remove_outliers(
                primaryts, args.remove_outliers_pf, method='pf')
            start = int(primaryts.span[0])
            end = int(primaryts.span[1])

        primary_mean = numpy.mean(primaryts.value)
        primary_std = numpy.std(primaryts.value)

        # -- get aux data
        LOGGER.info("-- Loading auxiliary channel data")
        nchan = len(aux_channels)
        LOGGER.debug(f"Identified {nchan} channels")

        # read aux channels
        auxdata = get_data(
            aux_channels,
            int(seg[0]), int(seg[1]),
            verbose='Reading:'.rjust(30),
            frametype=aux_frametype,
            nproc=args.nproc,
            pad=0,
            on_gaps='error')

        # re-order aux channels to the same order as channel list

        [auxdata.move_to_end(key=ch) for ch in aux_channels if ch]

        # -- removes flat data to be re-introdused later

        LOGGER.info('-- Pre-processing auxiliary channel data')
        auxdata = gwlasso.remove_flat(auxdata)
        flatable = Table(data=(list(set(aux_channels) - set(auxdata.keys())),),
                         names=('Channels',))
        LOGGER.debug(f'Removed {len(flatable)} channels with flat data')
        LOGGER.debug(f'{len(auxdata)} channels remaining')

        # -- remove bad data

        LOGGER.info("Removing any channels with bad data...")
        nbefore = len(auxdata)
        auxdata = gwlasso.remove_bad(auxdata)
        nafter = len(auxdata)
        LOGGER.debug(f'Removed {nbefore - nafter} channels with bad data')
        LOGGER.debug(f'{nafter} channels remaining'.format())
        data = numpy.array([ts.value for ts in auxdata.values()]).T
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        # -- perform lasso regression -------------------

        # create model
        LOGGER.info('-- Fitting data to target')
        target = scale(primaryts.value)
        model = gwlasso.fit(data, target, alpha=args.alpha)
        LOGGER.info(f'Alpha: {model.alpha}')

        # restructure results for convenience
        allresults = Table(
            data=(list(auxdata.keys()), model.coef_, numpy.abs(model.coef_)),
            names=('Channel', 'Lasso coefficient', 'rank'))
        allresults.sort('rank')
        allresults.reverse()
        useful = allresults['rank'] > 0
        allresults.remove_column('rank')
        results = allresults[useful]  # non-zero coefficient
        zeroed = allresults[numpy.invert(useful)]  # zero coefficient

        # extract data for useful channels
        nonzerodata = {name: auxdata[name] for name in results['Channel']}
        nonzerocoef = {name: coeff for name, coeff in results.as_array()}

        # print results
        LOGGER.info(f'Found {len(results)} channels with |Lasso coefficient| '
                    f'>= {threshold}:\n\n')
        print(results)
        print('\n\n')

        # convert to pandas
        set_option('max_colwidth', None)
        df = results.to_pandas()
        df.index += 1

        # write results to files
        resultsfile = f'{args.ifo}-LASSO_RESULTS-{gpsstub}.csv'
        results.write(resultsfile, format='csv', overwrite=True)
        zerofile = f'{args.ifo}-ZERO_COEFFICIENT_CHANNELS-{gpsstub}.csv'
        zeroed.write(zerofile, format='csv', overwrite=True)
        flatfile = f'{args.ifo}-FLAT_CHANNELS-{gpsstub}.csv'
        flatable.write(flatfile, format='csv', overwrite=True)

        # -- generate lasso plots

        modelFit = model.predict(data)

        re_delim = re.compile(r'[:_-]')
        p1 = (.1, .15, .9, .9)  # global plot defaults for plot1, lasso model

        times = primaryts.times.value
        xlim = primaryts.span
        cmap = get_cmap('tab20')
        colors = [cmap(i) for i in numpy.linspace(0, 1, len(nonzerodata)+1)]

        plot = Plot(figsize=(12, 4))
        plot.subplots_adjust(*p1)
        ax = plot.gca()
        ax.set_xscale('auto-gps')
        ax.set_epoch(seg[0])
        ax.set_xlim(xlim)
        ax.plot(times, _descaler(target), label=texify(primary),
                color='black', linewidth=line_size_primary)
        ax.plot(times, _descaler(modelFit), label='Lasso model',
                linewidth=line_size_aux)
        if range_is_primary:
            ax.set_ylabel('Sensitive range [Mpc]')
            ax.set_title('Lasso Model of Range')
        else:
            ax.set_ylabel('Primary Channel Units')
            ax.set_title('Lasso Model of Primary Channel')
        ax.legend(loc='best')
        plot1 = gwplot.save_figure(
            plot,
            f'{args.ifo}-LASSO_MODEL-{gpsstub}.png',
            bbox_inches='tight')

        # summed contributions
        plot = Plot(figsize=(12, 4))
        plot.subplots_adjust(*p1)
        ax = plot.gca()
        ax.set_xscale('auto-gps')
        ax.set_epoch(seg[0])
        ax.set_xlim(xlim)
        ax.plot(times, _descaler(target), label=texify(primary),
                color='black', linewidth=line_size_primary)
        summed = 0
        for i, name in enumerate(results['Channel']):
            summed += scale(nonzerodata[name].value) * nonzerocoef[name]
            if i:
                label = f'Channels 1-{i+1}'
            else:
                label = 'Channel 1'
            ax.plot(times, _descaler(summed), label=label, color=colors[i],
                    linewidth=line_size_aux)
        if range_is_primary:
            ax.set_ylabel('Sensitive range [Mpc]')
        else:
            ax.set_ylabel('Primary Channel Units')
        ax.set_title('Summations of Channel Contributions to Model')
        ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        plot2 = gwplot.save_figure(
            plot,
            f'{args.ifo}-LASSO_CHANNEL_SUMMATION-{gpsstub}.png',
            bbox_inches='tight')

        # individual contributions
        plot = Plot(figsize=(12, 4))
        plot.subplots_adjust(*p1)
        ax = plot.gca()
        ax.set_xscale('auto-gps')
        ax.set_epoch(seg[0])
        ax.set_xlim(xlim)
        ax.plot(times, _descaler(target), label=texify(primary),
                color='black', linewidth=line_size_primary)
        for i, name in enumerate(results['Channel']):
            this = _descaler(
                scale(nonzerodata[name].value) * nonzerocoef[name])
            if i:
                label = 'Channels 1-{0}'.format(i+1)
            else:
                label = 'Channel 1'
            ax.plot(times, this, label=texify(name), color=colors[i],
                    linewidth=line_size_aux)
        if range_is_primary:
            ax.set_ylabel('Sensitive range [Mpc]')
        else:
            ax.set_ylabel('Primary Channel Units')
        ax.set_title('Individual Channel Contributions to Model')
        ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        plot3 = gwplot.save_figure(
            plot,
            f'{args.ifo}-LASSO_CHANNEL_CONTRIBUTIONS-{gpsstub}.png',
            bbox_inches='tight')

        # -- process aux channels, making plots

        LOGGER.info("-- Processing channels")
        counter = multiprocessing.Value('i', 0)

        # process channels
        pool = multiprocessing.Pool(nprocplot)
        results = pool.map(_process_channel,
                           enumerate(list(nonzerodata.items())))
        results = sorted(results, key=lambda x: abs(x[1]), reverse=True)
        pool.close()

        #  generate clustered time series plots
        counter = multiprocessing.Value('i', 0)
        max_correlated_channels = 20

        if args.no_cluster is False:
            LOGGER.info("-- Generating clusters")
            pool = multiprocessing.Pool(nprocplot)
            clusters = pool.map(_generate_cluster, enumerate(results))
            pool.close()

        channelsfile = f'{args.ifo}-CHANNELS-{gpsstub}.csv'
        numpy.savetxt(channelsfile, aux_channels, delimiter=',', fmt='%s')

        # -- Generate a segment summary page

        # write html
        title = f'{args.ifo} Lasso Correlation: {gpsblock}'
        if args.band_pass:
            links = [gpsblock] + [(s, f'#{s.lower()}') for s in
                                  ['Parameters', 'Spectra', 'Model',
                                   'Results']]
        else:
            links = [gpsblock] + [(s, f'#{s.lower()}') for s in
                                  ['Parameters', 'Model', 'Results']]
        (brand, class_) = htmlio.get_brand(args.ifo, 'Lasso', seg[0])
        navbar = htmlio.navbar(links, class_=class_, brand=brand)
        page = htmlio.new_bootstrap_page(
            title=f'{args.ifo} Lasso | {gpsblock}',
            navbar=navbar)
        page.h1(title, class_='pb-2 mt-3 mb-2 border-bottom')

        # -- summary table
        content = [
            ('Primary channel', markup.oneliner.code(primary)),
            ('Primary frametype', markup.oneliner.code(
                args.primary_frametype) or '-'),
            ('Primary cache file', markup.oneliner.code(
                args.primary_cache) or '-'),
            ('Outlier threshold', f'{args.remove_outliers} sigma'),
            ('Lasso coefficient threshold', str(threshold)),
            ('Cluster coefficient threshold', str(args.cluster_coefficient)),
            ('Non-zero coefficients', str(numpy.count_nonzero(model.coef_))),
            ('&alpha; (model)', '%.4f' % model.alpha)]
        if args.band_pass:
            content.insert(2, ('Primary bandpass',
                               f'{flower}-{fupper} Hz'))
        page.h2('Parameters', class_='mt-4 mb-4', id_='parameters')
        page.div(class_='row')
        page.div(class_='col-md-9 col-sm-12')
        page.add(htmlio.parameter_table(
            content, start=int(seg[0]), end=int(seg[1])))
        page.div.close()  # col-md-9 col-sm-12

        # -- download button
        files = [
            (f'{nchan} analyzed channels (CSV)', channelsfile),
            (f'{len(flatable)} flat channels (CSV)', flatfile),
            (f'{len(zeroed)} zeroed channels (CSV)', zerofile)]
        page.div(class_='col-md-3 col-sm-12')
        page.add(htmlio.download_btn(
            files, label='Channel information',
            btnclass=f'btn btn-{args.ifo.lower()} dropdown-toggle'))
        page.div.close()  # col-md-3 col-sm-12
        page.div.close()  # rowa

        # -- command-line
        page.h5('Command-line:')
        page.add(htmlio.get_command_line(about=False, prog=PROG))

        if args.band_pass:
            page.h2('Primary channel spectra', class_='mt-4', id_='spectra')
            page.div(class_='card border-light card-body shadow-sm')
            page.div(class_='row')
            page.div(class_='col-md-6')
            spectra_img1 = htmlio.FancyPlot(spectrum_plot_zoomed_out)
            page.add(htmlio.fancybox_img(spectra_img1))
            page.div.close()  # col-md-6
            page.div(class_='col-md-6')
            spectra_img2 = htmlio.FancyPlot(spectrum_plot_zoomed_in)
            page.add(htmlio.fancybox_img(spectra_img2))
            page.div.close()  # col-md-6
            page.div.close()  # row
            page.div.close()  # card border-light card-body shadow-sm

        # -- model information
        page.h2('Model information', class_='mt-4', id_='model')

        page.div(class_='card card-%s card-body shadow-sm' % args.ifo.lower())
        page.div(class_='row')
        page.div(class_='col-md-8 offset-md-2', id_='results-table')
        page.p('Below are the top {} mean minute-trend channels, ranked by '
               'Lasso correlation with the primary.'.format(df.shape[0]))
        page.add(df.to_html(
            classes=('table', 'table-sm', 'table-hover'),
            formatters={
                'Lasso coefficient': lambda x: "%.4f" % x,
                'Channel': lambda x: str(htmlio.cis_link(x.split('.')[0])),
                '__index__': lambda x: str(x)
            },
            escape=False,
            border=0).replace(' style="text-align: right;"', ''))
        page.div.close()  # col-md-10 offset-md-1
        page.div.close()  # row

        page.div(class_='row', id_='primary-lasso')
        page.div(class_='col-md-8 offset-md-2')
        img1 = htmlio.FancyPlot(plot1)
        page.add(htmlio.fancybox_img(img1))  # primary lasso plot
        page.div.close()  # col-md-8 offset-md-2
        page.div.close()  # primary-lasso

        page.div(class_='row', id_='channel-summation')
        img2 = htmlio.FancyPlot(plot2)
        page.div(class_='col-md-8 offset-md-2')
        page.add(htmlio.fancybox_img(img2))
        page.div.close()  # col-md-8 offset-md-2
        page.div.close()  # channel-summation

        page.div(class_='row', id_='channels-and-primary')
        img3 = htmlio.FancyPlot(plot3)
        page.div(class_='col-md-8 offset-md-2')
        page.add(htmlio.fancybox_img(img3))
        page.div.close()  # col-md-8 offset-md-2
        page.div.close()  # channels-and-primary

        page.div.close()  # card card-<ifo> card-body shadow-sm

        # -- results
        page.h2('Top channels', class_='mt-4', id_='results')
        page.div(id_='results')
        # for each aux channel, create information container and put
        # plots in it
        for i, (ch, lassocoef, plot4, plot5, plot6, ts) in enumerate(results):
            # set container color/context based on lasso coefficient
            if lassocoef == 0:
                break
            elif abs(lassocoef) < threshold:
                h = '%s [lasso coefficient = %.4f] (Below threshold)' % (
                    ch, lassocoef)
            else:
                h = '%s [lasso coefficient = %.4f]' % (ch, lassocoef)
            if ((lassocoef is None) or (lassocoef == 0)
                    or (abs(lassocoef) < threshold)):
                card = 'card border-light mb-1 shadow-sm'
                card_header = 'card-header bg-light'
            elif abs(lassocoef) >= .5:
                card = 'card border-danger mb-1 shadow-sm'
                card_header = 'card-header text-white bg-danger'
            elif abs(lassocoef) >= .2:
                card = 'card border-warning mb-1 shadow-sm'
                card_header = 'card-header text-white bg-warning'
            else:
                card = 'card border-info mb-1 shadow-sm'
                card_header = 'card-header text-white bg-info'
            page.div(class_=card)

            # heading
            page.div(class_=card_header)
            page.a(h, class_='collapsed card-link cis-link',
                   href=f'#channel{i}', **{'data-bs-toggle': 'collapse'})
            page.div.close()  # card-header
            # body
            page.div(id_='channel%d' % i, class_='collapse',
                     **{'data-parent': '#results'})
            page.div(class_='card-body')
            if lassocoef is None:
                page.p('The amplitude data for this channel is flat (does not '
                       'change) within the chosen time period.')
            elif abs(lassocoef) < threshold:
                page.p('Lasso coefficient below the threshold of %g.'
                       % (threshold))
            else:
                for image in [plot4, plot5, plot6]:
                    img = htmlio.FancyPlot(image)
                    page.div(class_='row')
                    page.div(class_='col-md-8 offset-md-2')
                    page.add(htmlio.fancybox_img(img))
                    page.div.close()  # col-md-8 offset-md-2
                    page.div.close()  # row
                    page.add('<hr class="row-divider">')
                if args.no_cluster is False:
                    if clusters[i][0] is None:
                        page.p("<font size='3'><br />No channels were highly "
                               "correlated with this channel.</font>")
                    else:
                        page.div(class_='row', id_='clusters')
                        page.div(class_='col-md-12')
                        cimg = htmlio.FancyPlot(clusters[i][0])
                        page.add(htmlio.fancybox_img(cimg))
                        page.div.close()  # col-md-12
                        page.div.close()  # clusters
                        if clusters[i][1] is not None:
                            corr_link = markup.oneliner.a(
                                'Export {} channels (CSV)'.format(
                                    max_correlated_channels),
                                href=clusters[i][1], download=clusters[i][1],
                            )
                            page.button(
                                corr_link,
                                class_='btn btn-%s' % args.ifo.lower(),
                            )
            page.div.close()  # card-body
            page.div.close()  # collapse
            page.div.close()  # card
        page.div.close()  # results
        htmlio.close_page(page, 'index.html')  # save and close

    # -- Generate an overview summary page

    LOGGER.info("Generating summary page")

    os.makedirs(args.summary_dir, exist_ok=True)
    os.chdir(args.summary_dir)

    external_link = args.output_dir.replace(
        os.path.join(os.environ.get('HOME'), 'public_html'),
        f'/~{getuser()}'
    )

    # get primary channel data
    # gds_channel = '%s:DMT-SNSH_EFFECTIVE_RANGE_MPC.mean' % ifo
    # frontend_channel = '%s:DMT-SNSW_EFFECTIVE_RANGE_MPC.mean' % ifo

    primary_ts = get_data(primary, start, end, pad=numpy.nan,
                          frametype=args.primary_frametype,
                          on_gaps='warn')

    """gds_timeseries = get_data(
        gds_channel, gpsstart, gpsend, pad=numpy.nan,
        frametype='SenseMonitor_hoft_%s' % frametype)
    frontend_timeseries = get_data(
        frontend_channel, gpsstart, gpsend, pad=numpy.nan,
        frametype='SenseMonitor_CAL_%s' % frametype)"""

    # construct primary channel timseries plot
    range_plot = Plot(figsize=(12, 6))
    ax = range_plot.gca()
    if args.ifo == 'H1':
        ax.plot(primary_ts, color='gwpy:ligo-hanford', label='Primary',
                linewidth=1)
    else:
        ax.plot(primary_ts, color='gwpy:ligo-livingston', label='Primary',
                linewidth=1)
    ax.set_xscale('hours')
    ax.set_epoch(start)
    ax.set_xlim(start, end)
    if range_is_primary:
        ax.set_ylim(-5, 180)
        ax.set_ylabel('Angle-averaged range [Mpc]')
        ax.set_title(f'{args.ifo} binary neutron star inspiral range '
                     '(DMT SenseMon)')
    else:
        ax.set_label('Primary channel units')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    range_plot.add_segments_bar(state, ax=ax, pad=.1, height=.14)
    gpsstub_day = f'{start}-86400'
    if state_is_observing:
        range_filename = (f'{args.ifo}-OBSERVING_TIMESERIES-{gpsstub_day}.png')
    else:
        range_filename = (f'{args.ifo}-LOCKED-TIMESERIES-{gpsstub_day}.png')
    range_plot.savefig(range_filename, bbox_inches='tight')

    # generate summaries of lasso models for each data-quality segment
    set_option('max_colwidth', None)
    seg_page_links = []
    dataframes = []
    seg_plot_links = []
    for i, subdir in enumerate(subdirs):
        seg_page_links.append(os.path.join(external_link, gpsblocks[i]))
        dataframes.append(None)
        seg_plot_links.append(None)
        for file_ in os.listdir(subdir):
            if fnmatch.fnmatch(file_, f'{args.ifo}-LASSO_RESULTS-*'):
                file_ = '/'.join([subdir, file_])
                usefultab = Table.read(file_, format='csv')
                df = usefultab.to_pandas()
                df.index += 1
                if len(df) == 0:
                    df = None
                dataframes[i] = df

            elif fnmatch.fnmatch(file_,
                                 f'{args.ifo}-LASSO_CHANNEL_SUMMATION-*'):
                summations_plot_link = (os.path.join(
                    external_link,
                    gpsblocks[i],
                    f'{args.ifo}-LASSO_CHANNEL_SUMMATION-{gpsstubs[i]}.png'))
                seg_plot_links[i] = summations_plot_link

    # generate HTML page
    title = f'Lasso Slow Correlations: {start}-{end}'
    sections = [[f'{i+1}: {gpsblocks[i]}', f'#{gpsblocks[i]}']
                for i, seg in enumerate(gpsblocks)]
    date = tconvert(start)
    links = [date.strftime('%Y-%m-%d'),
             ['Summary', '#'],
             ['Segments', [['Analyses', sections]]]]
    (brand, class_) = htmlio.get_brand(args.ifo, 'Daily Lasso', start)
    navbar = htmlio.navbar(links, class_=class_, brand=brand)
    page = htmlio.new_bootstrap_page(title=title, navbar=navbar)

    # page header
    page.div(class_='page-header')
    page.h1(title)
    lasso_link = 'https://www.jstor.org/stable/2346178'
    paper_link = 'http://dx.doi.org/10.1088/1361-6382/aae593'
    if range_is_primary:
        primary_nickname = f'BNS range (<code>{args.primary_channel}</code>)'
    else:
        primary_nickname = f'<code>{args.primary_channel}</code> channel'
    page.add(htmlio.alert(
        'This algorithm uses <a href="{0}" class="alert-link" target="_blank">'
        'lasso regression</a> to create a model for fluctuations in the {1} '
        'based on information in auxiliary channels. Using minute trend data, '
        'a minimal set of channels is selected and then clustering is '
        'performed to find closely-related channels. For more information, '
        'see <a href="{2}" class="alert-link" target="_blank">Walker et al. '
        '(2018)</a>.'.format(
            lasso_link, primary_nickname, paper_link),
        ))
    page.add(htmlio.get_command_line(prog=PROG))
    page.div.close()  # page-header

    img = htmlio.FancyPlot(
        '/'.join(
            [external_link.replace('full-results', 'summary'),
             range_filename]),
        caption='{ifo} binary neutron star inspiral range on {date}. The '
                'colorful trace represents the range based on <code>'
                '{ifo}:GDS-CALIB_STRAIN</code>, while the gray trace is based '
                'on <code>{ifo}:CAL-DELTAL_EXTERNAL_DQ</code>.'.format(
                    ifo=args.ifo, date=date.strftime('%Y%m%d')))
    page.div(class_='row')
    page.div(class_='col-md-8 offset-md-2')
    page.add(htmlio.fancybox_img(img))  # primary lasso plot
    page.div.close()  # col-md-8 offset-md-2
    page.div.close()  # row

    page.div(id_='main')
    page.div(class_='banner')
    page.h2('Results Summary', id_='summary')
    page.div.close()  # banner

    if len(lasso_segs) == 0:
        page.add(htmlio.alert(
            f'No segments of length {args.segment_min_length/3600} hours or '
            f'longer were found in state <code>{state_flag}</code> on '
            f"{date.strftime('%Y-%m-%d')}. State <code>{state_flag}</code> "
            f"{abs(segs)} s, frame coverage {abs(data_segs)} s",
            dismiss=False,
        ))
    else:
        page.add(htmlio.alert(
            f'A total of {len(segs)} segments were found in state '
            f"<code>{state_flag}</code> on {date.strftime('%Y-%m-%d')} "
            f'of which {len(lasso_segs)} are at least '
            f'{args.segment_min_length/3600} hours long. These are '
            'summarized and linked below. The list of channels for each '
            'segment represents the largest contributors in modeling '
            f'{primary_nickname}.',
        ))

    for i, link in enumerate(seg_page_links):
        page.div(class_=f'card card-{args.ifo.lower()} mb-4 shadow-sm')
        page.div(class_='card-header pb-0')
        page.h5(gpsblocks[i], class_='card-title', id_=gpsblocks[i])
        page.div.close()  # card-header pb-0
        page.div(class_='card-body')
        tableid = f'{args.ifo.lower()}_lasso_{gpsblocks[i]}'
        page.div(class_='container')
        if dataframes[i] is None:
            page.div('No page was generated for this segment',
                     class_='alert alert-danger')
        else:
            page.div(class_='row', id_='results-table')
            page.div(class_='col-md-9 col-sm-12')
            page.p(f'Below are the top {dataframes[i].shape[0]} mean minute-'
                   'trend channels, ranked by Lasso correlation with the '
                   'primary.')
            page.add(dataframes[i].to_html(
                classes=('table', 'table-sm', 'table-hover'),
                formatters={
                    'Lasso coefficient': lambda x: "%.4f" % x,
                    'Channel': lambda x: str(htmlio.cis_link(x.split('.')[0])),
                    '__index__': lambda x: "%d" % x
                },
                escape=False,
                border=0,
                table_id=tableid).replace(' style="text-align: right;"', ''))
            page.div.close()  # col-md-9 col-sm-12
            page.div(class_='col-md-3 col-sm-12')
            page.div(class_='float-right d-none d-md-block')
            page.a('Full results', class_=f'btn btn-{args.ifo.lower()}',
                   href=link, role='button')
            page.div.close()  # float-right d-none d-md-block
            page.div.close()  # col-md-3 col-sm-12
            page.div.close()  # results-table
            page.hr(class_='row-divider')
            # no summary plot available
            if seg_plot_links[i] is None:
                page.div('This segment failed to generate a summary plot',
                         class_='alert alert-warning')
            else:
                page.div(class_='container', id_='channel-summation')
                img2 = htmlio.FancyPlot(seg_plot_links[i])
                page.div(class_='row')
                page.div(class_='col-md-12')
                page.add(htmlio.fancybox_img(img2))
                page.div.close()  # col-md-12
                page.div.close()  # row
                page.div.close()  # channel-summation
        page.div.close()  # container
        page.div.close()  # card-body
        page.div.close()  # card well

    # close page
    page.div.close()  # main
    htmlio.close_page(page, 'index.html')

    LOGGER.info("-- Process Completed")


# -- run from command-line ----------------------------------------------------

if __name__ == "__main__":
    main()
