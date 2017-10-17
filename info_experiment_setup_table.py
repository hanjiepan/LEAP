"""
Copyright 2017 Hanjie Pan

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import division
import setup
import os
import subprocess
import sys
from astropy import units
from astropy.coordinates import SkyCoord
import numpy as np
import pandas as pd
import scipy.constants as sc
from astropy.time import Time
from utils import planar_compute_all_baselines
from formDataStructures import openHDF5, dataKey

if sys.version_info[0] > 2:
    sys.exit('Sorry casacore only runs on Python 2.')
else:
    from casacore import tables as casa_tables

def loadData(timeIndex, freqIndex, args):
    """
    Load data from the input HDF5 file and transform relevant fields from UVW to XYZ coordinates

    :param timeIndex: time index
    :param freqIndex: freq index
    :param args: output of parseArgs()
    :return: (S, STATION_ID, STATION_XYZ, gridPoints_XYZ, wavelength, pointing_direction)
    """
    store = openHDF5(args)

    S = store[dataKey('S', timeIndex, freqIndex)].iloc[:args['stationCount'], :args['stationCount']]

    wavelength = sc.speed_of_light / store['FREQ_MAP'].loc[freqIndex].values

    STATION_ID = store[dataKey('STATION_ID', timeIndex, freqIndex)][:args['stationCount']]

    STATION_UVW = store[dataKey('STATION_UVW', timeIndex, freqIndex)]
    STATION_UVW = pd.concat(
        [station for (_, station) in STATION_UVW.groupby(by='stationID')][:args['stationCount']],
        ignore_index=True
    )

    pointing_direction = np.array([0, np.pi / 2])

    store.close()
    return S, STATION_ID, STATION_UVW, wavelength, pointing_direction


if __name__ == '__main__':
    dataset = 'bootes'  # either 'bootes' or 'toothbrush'
    '''
    list of WORKING stations:
        - bootes (station 12, 13, 16, 17 are flagged): 
            [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 14, 15, 18, 19, 20, 
              21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
              38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 54, 55,
              56, 57, 58, 59, 60, 61]
        
        - toothbrush:
            [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 
              17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
              34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
              52, 53, 54]
    '''
    parameter_set = {}
    if dataset == 'bootes':
        parameter_set = {
            'basefile_name': 'BOOTES24_SB180-189.2ch8s_SIM',
            'freq_channel_min': 0,
            'num_station': 28,  # <- 24 working stations
        }
    elif dataset == 'toothbrush':
        parameter_set = {
            'basefile_name': 'RX42_SB100-109.2ch10s',
            'freq_channel_min': 19,
            'num_station': 36,
        }
    else:
        RuntimeError('Unknown dataset: {}'.format(dataset))

    data_root_path = os.environ['DATA_ROOT_PATH']
    hdf5_root_path = os.environ['BLUEBILD_ROOT_PATH']
    basefile_name = parameter_set['basefile_name']
    ms_file_name = data_root_path + basefile_name + '.ms'
    num_channel = 1
    sti_step = 50 * num_channel  # step size for the integration time
    time_end = None  # the end of the observation
    # frequency channels in [freq_channel_min, freq_channel_max) are selected
    freq_channel_min = parameter_set['freq_channel_min']
    freq_channel_max = freq_channel_min + num_channel
    freq_channel_step = 1  # <- if non-consecutive, then wsclean has difficulty for the input spec.
    num_station = parameter_set['num_station']
    num_antenna = 24  # <= at each time at most 24 out of 48 antennas are working

    freq_sel_idx = np.r_[freq_channel_min:freq_channel_max:freq_channel_step]
    num_subband = freq_sel_idx.size

    # get pointing direction
    taql_cmd_str = 'select REFERENCE_DIR from {ms_file_name}::FIELD'.format(
        ms_file_name=ms_file_name
    )
    direction = casa_tables.taql(taql_cmd_str).getcol('REFERENCE_DIR').squeeze()
    RA_rad, DEC_rad = direction
    RA_hms, DEC_dms = SkyCoord(
        ra=RA_rad, dec=DEC_rad, unit=units.radian
    ).to_string('hmsdms').split(' ')
    print('Telesceope focus (RA, DEC): ({RA_hms}, {DEC_dms})'.format(
        RA_hms=RA_hms, DEC_dms=DEC_dms
    ))
    print('--------------------------------------')

    # Subband frequencies
    taql_cmd_str = 'select CHAN_FREQ from {ms_file_name}::SPECTRAL_WINDOW'.format(
        ms_file_name=ms_file_name
    )
    freq_subbands_hz = casa_tables.taql(taql_cmd_str).getcol('CHAN_FREQ').squeeze()
    # for band_count, subband_freq in enumerate(freq_subbands_hz):
    for band_number in freq_sel_idx:
        print('Subband {band_count} frequency {subband_freq}MHz'.format(
            band_count=band_number,
            subband_freq=repr(freq_subbands_hz[band_number] / 1e6)
        ))
    freq_subbands_hz = freq_subbands_hz[freq_sel_idx]
    print('--------------------------------------')

    # Observation time
    taql_cmd_str = 'select distinct TIME from {ms_file_name}'.format(
        ms_file_name=ms_file_name
    )
    # get TIME column description
    time_column_desc = casa_tables.table(ms_file_name).getcoldesc('TIME')
    for dict_key in time_column_desc.keys():
        print('{0}: {1}'.format(dict_key, time_column_desc[dict_key]))
    print('--------------------------------------')
    # the observation time is in Modified Julian Day format
    # note the value unit (second or day)
    observation_time = casa_tables.taql(taql_cmd_str).getcol('TIME')[:time_end]
    # total number of integration time
    num_sti = int((observation_time.size - 1) // sti_step) + 1
    print('num sti {}'.format(num_sti))

    observation_start_time = Time(observation_time[0] / (24 * 3600),
                                  scale='utc', format='mjd')
    observation_start_time.format = 'iso'  # convert to iso format
    observation_end_time = Time(observation_time[(num_sti - 1) * sti_step] / (24 * 3600),
                                scale='utc', format='mjd')
    observation_end_time.format = 'iso'  # convert to iso format
    print('Observation start time (UTC): {}'.format(observation_start_time))
    print('Observation end time (UTC): {}'.format(observation_end_time))
    print('Time resolution: {0}sec'.format(observation_time[sti_step] - observation_time[0]))

    args = {
        'freqRange': freq_sel_idx,
        'timeRange': np.r_[0:(num_sti - 1) * sti_step:sti_step],
        'stationCount': num_station,
        'dataFile': hdf5_root_path + basefile_name + '.hdf5'
    }
    # get antenna layout
    # since not all antennas are always working, we initialise the matrix filled with nan.
    # later, we can use np.isnan to determine which antenna are involved.
    array_coordinate = np.full((num_antenna, num_station, num_sti, 3),
                               np.nan, dtype=float)
    for freq_count, freqIndex in enumerate(args['freqRange']):
        for time_count, timeIndex in enumerate(args['timeRange']):
            S, STATION_ID, STATION_UVW, \
            wavelength, pointing_direction = \
                loadData(int(timeIndex), int(freqIndex), args)

            # frequencies of different subbands
            freq_subbands_hz[freq_count] = sc.speed_of_light / wavelength

            # antenna coordinates
            antenna_idx = np.mod(STATION_UVW.loc[:, 'antennaID'].values, num_antenna)

            '''
            because some stations may not be working, we change the station_id to a
            sequentially increasing sequence -- we will use station id later to store
            antenna coordinates
            '''
            for staion_id_count, station_id_loop in enumerate(STATION_ID.values):
                STATION_UVW['stationID'].replace(station_id_loop, staion_id_count, inplace=True)

            station_idx = STATION_UVW.loc[:, 'stationID'].values

            array_coordinate[antenna_idx, station_idx, time_count, 0] = \
                STATION_UVW.loc[:, 'u'].values * wavelength
            array_coordinate[antenna_idx, station_idx, time_count, 1] = \
                STATION_UVW.loc[:, 'v'].values * wavelength
            array_coordinate[antenna_idx, station_idx, time_count, 2] = \
                STATION_UVW.loc[:, 'w'].values * wavelength

    light_speed = sc.speed_of_light  # speed of light
    wave_length = np.reshape(light_speed / freq_subbands_hz, (1, 1, 1, -1), order='F')

    r_antenna_x = array_coordinate[:, :, :num_sti, 0]
    r_antenna_y = array_coordinate[:, :, :num_sti, 1]
    r_antenna_z = array_coordinate[:, :, :num_sti, 2]

    # normalised antenna coordinates
    p_y_normalised = np.reshape(
        r_antenna_y, (-1, num_station, num_sti, 1), order='F') / wave_length
    p_x_normalised = np.reshape(
        r_antenna_x, (-1, num_station, num_sti, 1), order='F') / wave_length
    p_z_normalised = np.reshape(
        r_antenna_z, (-1, num_station, num_sti, 1), order='F') / wave_length

    # compute all the baselines
    all_baselines_x, all_baselines_y = \
        planar_compute_all_baselines(p_x_normalised, p_y_normalised, num_antenna,
                                     num_station, num_subband, num_sti)
    norm_baselines = np.sqrt(all_baselines_x ** 2 + all_baselines_y ** 2)
    print('Longest baseline {:.1f}wavelength'.format(np.max(norm_baselines)))
    # calculate instrument resolution
    instrument_resolution = 1. / np.max(norm_baselines)
    resolution_dms = SkyCoord(
        ra=instrument_resolution, dec=0, unit=units.radian
    ).to_string('dms').split(' ')[0]
    print('Telesceope instrument resolution: {resolution_dms}'.format(
        resolution_dms=resolution_dms
    ))
