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

"""Utilities for analysing optical scattering
"""

import numpy
from scipy.signal import savgol_filter

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

OPTIC_MOTION_CHANNELS = {
    'BS': ['SUS-BS_M1_DAMP_L_IN1_DQ',
           'SUS-BS_M2_WIT_L_DQ'],
    'ETMX': ['SUS-ETMX_M0_DAMP_L_IN1_DQ',
             'SUS-ETMX_R0_DAMP_L_IN1_DQ',
             'SUS-ETMX_L2_WIT_L_DQ'],
    'ETMY': ['SUS-ETMY_M0_DAMP_L_IN1_DQ',
             'SUS-ETMY_R0_DAMP_L_IN1_DQ',
             'SUS-ETMY_L2_WIT_L_DQ'],
    'IM1': ['SUS-IM1_M1_DAMP_L_IN1_DQ'],
    'IM2': ['SUS-IM2_M1_DAMP_L_IN1_DQ'],
    'IM3': ['SUS-IM3_M1_DAMP_L_IN1_DQ'],
    'IM4': ['SUS-IM4_M1_DAMP_L_IN1_DQ'],
    'ITMX': ['SUS-ITMX_M0_DAMP_L_IN1_DQ',
             'SUS-ITMX_R0_DAMP_L_IN1_DQ',
             'SUS-ITMX_L2_WIT_L_DQ'],
    'ITMY': ['SUS-ITMY_M0_DAMP_L_IN1_DQ',
             'SUS-ITMY_R0_DAMP_L_IN1_DQ',
             'SUS-ITMY_L1_WIT_L_DQ'],
    'MC1': ['SUS-MC1_M1_DAMP_L_IN1_DQ',
            'SUS-MC1_M3_WIT_L_DQ'],
    'MC2': ['SUS-MC2_M1_DAMP_L_IN1_DQ',
            'SUS-MC2_M3_WIT_L_DQ'],
    'MC3': ['SUS-MC3_M1_DAMP_L_IN1_DQ',
            'SUS-MC3_M3_WIT_L_DQ'],
    'OM1': ['SUS-OM1_M1_DAMP_L_IN1_DQ'],
    'OM2': ['SUS-OM2_M1_DAMP_L_IN1_DQ'],
    'OM3': ['SUS-OM3_M1_DAMP_L_IN1_DQ'],
    'OMC': ['SUS-OMC_M1_DAMP_L_IN1_DQ'],
    'PR2': ['SUS-PR2_M1_DAMP_L_IN1_DQ',
            'SUS-PR2_M3_WIT_L_DQ'],
    'PR3': ['SUS-PR3_M1_DAMP_L_IN1_DQ',
            'SUS-PR3_M3_WIT_L_DQ'],
    'PRM': ['SUS-PRM_M1_DAMP_L_IN1_DQ',
            'SUS-PRM_M3_WIT_L_DQ'],
    'RM1': ['SUS-RM1_M1_DAMP_L_IN1_DQ'],
    'RM2': ['SUS-RM2_M1_DAMP_L_IN1_DQ'],
    'ZM1': ['SUS-ZM1_M1_DAMP_L_IN1_DQ'],
    'ZM2': ['SUS-ZM2_M1_DAMP_L_IN1_DQ'],
    'OPO': ['SUS-OPO_M1_DAMP_L_IN1_DQ'],
    'OFI': ['SUS-OFI_M1_DAMP_L_IN1_DQ',
            'SUS-OFI_M1_DAMP_T_IN1_DQ'],
    'SR2': ['SUS-SR2_M1_DAMP_L_IN1_DQ',
            'SUS-SR2_M3_WIT_L_DQ'],
    'SR3': ['SUS-SR3_M1_DAMP_L_IN1_DQ',
            'SUS-SR3_M3_WIT_L_DQ'],
    'SRM': ['SUS-SRM_M1_DAMP_L_IN1_DQ',
            'SUS-SRM_M3_WIT_L_DQ'],
    'TMSX': ['SUS-TMSX_M1_DAMP_L_IN1_DQ'],
    'TMSY': ['SUS-TMSY_M1_DAMP_L_IN1_DQ'],
}

TRANSMON_CHANNELS = ['ASC-X_TR_B_NSUM_OUT_DQ', 
                     'ASC-Y_TR_B_NSUM_OUT_DQ']

FREQUENCY_MULTIPLIERS = range(1, 5)


def get_fringe_frequency(series, multiplier=2.0):
    """Predict scattering fringe frequency from the derivative of a timeseries

    Parameters
    ----------
    series : `~gwpy.timeseries.TimeSeries`
        timeseries record of relative motion

     multiplier : `float`
        harmonic number of fringe frequency

    Returns
    -------
    fringef : `~gwpy.timeseries.TimeSeries`
        timeseries record of fringe frequency

    See Also
    --------
    scipy.signal.savgol_filter
        for an implementation of the Savitzky-Golay filter
    """
    velocity = type(series)(savgol_filter(series.value, 5, 2, deriv=1))
    velocity.__array_finalize__(series)
    fringef = numpy.abs(multiplier * 2. / 1.064 * velocity *
                        velocity.sample_rate.value)
    fringef.override_unit('Hz')
    return fringef


def get_blrms(series, flow=4.0, fhigh=10.0, stride=1, fftlength=4,
             overlap=2, **kwargs):
    """ Generate blrms for a TimeSeries

    Parameters
    ----------
    series  : `~gwpy.timeseries.TimeSeries` of the channel
            for which blrms are to be computed.
    low : `float`
        lower limit of the bandpass frequency.
    high : `float`
         upper limit of the bandpass frequency.
    stride: `float`
           time duration of the stride for rms

    Returns
    -------
    blrms : A whitened band passed time series.
     """
    wseries = series.whiten(fftlength=fftlength, overlap=overlap, **kwargs)
    bpseries = wseries.bandpass(flow, fhigh)
    return bpseries.rms(stride)


def get_segments(series, threshold, name=None):
    """ Generates segments from a series that are above
        some threshold

    Parameters
    ----------
    series  : `~gwpy.timeseries.TimeSeries` of the channel
            for which the segments are to be computed.
    name : name of the flag for the segments.

    Returns
    --------
    DataQualityFlag
    """
    thresh = series > threshold
    name = name or series.name
    threshflag = thresh.to_dqflag(name, round=True)
    return threshflag
