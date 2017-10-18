"""
real_data.py: takes the output of formDataStructures.py and puts it in aformat suitable for FRI experiments.
Copyright (C) 2017  Hanjie Pan

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Correspondence concerning LEAP should be addressed as follows:
             Email: hanjie [Dot] pan [At] epfl [Dot] ch
    Postal address: EPFL-IC-LCAV
                    Station 14
                    1015 Lausanne
                    Switzerland
"""
import re
import argparse
import itertools
import numpy as np
import os, sys
import pandas as pd
import scipy.constants as sc

from formDataStructures import openHDF5, dataKey, printProgress


def extract_off_diag(mtx):
    """
    extract off-diagonal entries in mtx
    The output vector is order in a column major manner
    :param mtx: input matrix to extract the off-diagonal entries
    :return:
    """
    Q = mtx.shape[0]
    extract_cond = np.reshape((1 - np.eye(Q)).astype(bool), (-1, 1), order='F')
    return np.extract(extract_cond, mtx[:, :])


def parseArgs():
    """
    Parse command-line arguments.

    :return: dictionary of valid arguments
    """
    printProgress()

    def parseRange(code):
        if code is None:
            return None
        else:
            range = eval(code)
            return np.sort(range)

    parser = argparse.ArgumentParser(
        description="""
Read HDF5 file produced by formDataStructures.py and <DO SOMETHING WITH FRI>
            """,
        epilog="""
Example usage:
python real_data.py --dataFile '/Users/pan/Google Drive/RadioAstData/BOOTES24_SB180-189.2ch8s_SIM.hdf5'
                    --timeRange np.r_[0:2500:50]
                    --freqRange np.r_[0]
                    --stationCount 12
                    --FoV 5
                    --imageWidth 505
                    --lsqImage '/Users/pan/Google Drive/RadioAstData/bootes_background_eig48_station48.hdf5'
                    --catalog '/Users/pan/Google Drive/RadioAstData/skycatalog.npz'
                    --cleanData './data/CLEAN_data.npz'
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--dataFile', type=str, required=True, help='HDF5 file produced by formDataStructures.py')
    parser.add_argument('--timeRange', type=str, required=True, help="""
List of (integer) time indices to process.
The format is np.r_[<write all indices here>].
    """)
    parser.add_argument('--freqRange', type=str, required=True, help="""
List of (integer) frequency indices to process.
The format is np.r_[<write all indices here>].
    """)
    parser.add_argument('--stationCount', type=int, required=True, help="""
Integer K specifying that only the first K stations should be used.
If K is small, then only the core stations are being used.
    """)
    parser.add_argument('--FoV', type=float, required=True,
                        help='Field of View (degrees)')
    parser.add_argument('--imageWidth', type=int, required=True,
                        help='Width of image (pixels)')
    parser.add_argument('--lsqImage', type=str, default=None, required=False,
                        help='HDF5 file produced by generateImages.py')
    parser.add_argument('--catalog', type=str, default=None, required=False,
                        help='(Optional) Catalog data file')
    parser.add_argument('--nvss_catalog', type=str, default=None, required=False,
                        help='(Optional) NVSS catalog data file')
    parser.add_argument('--cleanData', type=str, default=None, required=False,
                        help='(Optional) CLEAN image with wsclean')
    parser.add_argument('--csData', required=False,
                        help='(Optional) CS image with wsclean')
    parser.add_argument('--trim_data', default=False, action='store_true',
                        help='If present, then the data is trimmed (due to failed stations)')

    args = vars(parser.parse_args())

    if args['dataFile'] == 'None':
        args['dataFile'] = None
    if args['lsqImage'] == 'None':
        args['lsqImage'] = None
    if args['catalog'] == 'None':
        args['catalog'] = None
    if args['nvss_catalog'] == 'None':
        args['nvss_catalog'] = None

    if args['dataFile'] is not None:
        args['dataFile'] = os.path.abspath(args['dataFile'])
    if args['lsqImage'] is not None:
        args['lsqImage'] = os.path.abspath(args['lsqImage'])
    args['timeRange'] = parseRange(args['timeRange'])
    args['freqRange'] = parseRange(args['freqRange'])

    return args


def getPointingDirection(args):
    """
    Returns the pointing direction.

    :param args: output of parseArgs()
    :return: (longitude [-pi,pi], latitude [-pi/2,pi/2])
    """
    store = openHDF5(args)
    pointing_direction = store['POINTING_DIRECTION']
    store.close()
    return pointing_direction


def computeGridPoints(args):
    """
    Calculate the grid-points on which the random field must be drawn.
    :param args: output of parseArgs()
    :return: (args['imageWidth']**2,3) array of XYZ grid-points
    """
    FoV = args['FoV'] * np.pi / 180.

    x = y = np.linspace(-np.sin(FoV / 2.), np.sin(FoV / 2.), args['imageWidth'])
    [X, Y] = np.meshgrid(x, y)
    Z = np.sqrt(1 - X ** 2 - Y ** 2)

    gridPoints = np.column_stack((
        X.reshape(-1),
        Y.reshape(-1),
        Z.reshape(-1)
    ))
    return gridPoints


def loadData(timeIndex, freqIndex, args):
    """
    Load data from the input HDF5 file and transform relevant fields from UVW to XYZ coordinates

    :param timeIndex: time index
    :param freqIndex: freq index
    :param args: output of parseArgs()
    :return: (S, STATION_ID, STATION_XYZ, gridPoints_XYZ, wavelength, pointing_direction)
    """
    store = openHDF5(args)

    FoV_radian = args['FoV'] * np.pi / 180

    S = store[dataKey('S', timeIndex, freqIndex)].iloc[:args['stationCount'], :args['stationCount']]

    wavelength = sc.speed_of_light / store['FREQ_MAP'].loc[freqIndex].values

    STATION_ID = store[dataKey('STATION_ID', timeIndex, freqIndex)][:args['stationCount']]

    STATION_UVW = store[dataKey('STATION_UVW', timeIndex, freqIndex)]
    STATION_UVW = pd.concat(
        [station for (_, station) in STATION_UVW.groupby(by='stationID')][:args['stationCount']],
        ignore_index=True
    )

    pointing_direction = store['POINTING_DIRECTION'].values
    gridPoints_UVW = computeGridPoints(args)

    store.close()
    return S, STATION_ID, STATION_UVW, gridPoints_UVW, wavelength, pointing_direction, FoV_radian


if __name__ == '__main__':
    args = parseArgs()
    if args['lsqImage'] is None:
        lsqImg_available = False
    else:
        lsqImg_available = True

    if args['catalog'] is None:
        catalog_available = False
    else:
        catalog_available = True

    if args['nvss_catalog'] is None:
        nvss_catalog_available = False
    else:
        nvss_catalog_available = True

    if args['cleanData'] is None:
        clean_data_availabe = False
    else:
        clean_data_availabe = True

    if args['csData'] is None:
        cs_data_available = False
    else:
        cs_data_available = True

    # print(type(args['stationCount']), args['stationCount'])
    num_subband = args['freqRange'].size
    num_sti = args['timeRange'].size
    num_station = args['stationCount']
    num_antenna = 24  # <= at each time at most 24 out of 48 antennas are working

    # the station count is not always consecutive (some stations are not working)
    max_station_num = loadData(0, 0, args)[1].size
    num_station = min(num_station, max_station_num)
    args['stationCount'] = num_station

    freq_subbands_hz = np.zeros(num_subband, dtype=float)
    # since not all antennas are always working, we initialise the matrix filled with nan.
    # later, we can use np.isnan to determine which antenna are involved.
    array_coordinate = np.full((num_antenna, num_station, num_sti, 3), np.nan, dtype=float)
    visi_noisy = np.zeros((num_station * (num_station - 1), num_sti, num_subband), dtype=complex)

    for freq_count, freqIndex in enumerate(args['freqRange']):
        for time_count, timeIndex in enumerate(args['timeRange']):
            S, STATION_ID, STATION_UVW, gridPoints_UVW, \
            wavelength, pointing_direction, FoV_radian = \
                loadData(int(timeIndex), int(freqIndex), args)
            if args['trim_data']:
                # find failed stations
                validStationIDs = np.where(~np.all(S == 0, axis=0))
                # trim data
                STATION_UVW = STATION_UVW[STATION_UVW['stationID'].isin(*validStationIDs)]

            # frequencies of different subbands
            freq_subbands_hz[freq_count] = sc.speed_of_light / wavelength

            # antenna coordinates
            antenna_idx = np.mod(STATION_UVW.loc[:, 'antennaID'].values, num_antenna)

            '''
            because some stations may not be working, we change the station_id to a
            sequentially increasing sequence -- we will use station id later to store
            antenna coordinates
            '''
            '''
            for staion_id_count, station_id_loop in enumerate(STATION_ID.values):
                STATION_UVW['stationID'].replace(station_id_loop, staion_id_count, inplace=True)
            '''

            station_idx = STATION_UVW.loc[:, 'stationID'].values

            array_coordinate[antenna_idx, station_idx, time_count, 0] = \
                STATION_UVW.loc[:, 'u'].values * wavelength
            array_coordinate[antenna_idx, station_idx, time_count, 1] = \
                STATION_UVW.loc[:, 'v'].values * wavelength
            array_coordinate[antenna_idx, station_idx, time_count, 2] = \
                STATION_UVW.loc[:, 'w'].values * wavelength

            # noisy visibility measurements
            visi_noisy[:, time_count, freq_count] = extract_off_diag(S.as_matrix())

            # plotting grid point
            x_plt = gridPoints_UVW[:, 0].reshape(args['imageWidth'], args['imageWidth'])
            y_plt = gridPoints_UVW[:, 1].reshape(args['imageWidth'], args['imageWidth'])
            z_plt = gridPoints_UVW[:, 2].reshape(args['imageWidth'], args['imageWidth'])

            # telescope focusing point
            sky_focus = pointing_direction.squeeze()
            sky_ra = sky_focus[0]
            sky_dec = sky_focus[1]

    if lsqImg_available:
        # load least square image
        lsqImg_store = pd.HDFStore(args['lsqImage'], mode='r')
        # some frames are missing from the hdf5 file
        indexing_keys = lsqImg_store.keys()
        pattern = r'/DATA/t(?P<time>\d+)/IMAGE'
        valid_indices = [int(re.match(pattern, key).group('time'))
                         for key in indexing_keys if re.match(pattern, key) != None]

        img_lsq = np.zeros(lsqImg_store['/DEC'].shape)
        for loop_count in filter(lambda x: x in args['timeRange'], valid_indices):
            loop_file_name = '/DATA/t{t:=04d}/IMAGE'.format(t=loop_count)
            img_lsq += lsqImg_store[loop_file_name]

    # (optional) catalog
    if catalog_available:
        catalog_data = np.load(args['catalog'])
        skycatalog_intensities = catalog_data['Intensities_skyctalog']
        skycatalog_U = catalog_data['U_skycatalog']
        skycatalog_V = catalog_data['V_skycatalog']
        skycatalog_W = catalog_data['W_skycatalog']
    else:
        skycatalog_intensities = None
        skycatalog_U = None
        skycatalog_V = None
        skycatalog_W = None

    if nvss_catalog_available:
        nvss_catalog_data = np.load(args['nvss_catalog'])
        nvss_skycatalog_intensities = nvss_catalog_data['Intensities_skyctalog']
        nvss_skycatalog_U = nvss_catalog_data['U_skycatalog']
        nvss_skycatalog_V = nvss_catalog_data['V_skycatalog']
        nvss_skycatalog_W = nvss_catalog_data['W_skycatalog']
    else:
        nvss_skycatalog_intensities = None
        nvss_skycatalog_U = None
        nvss_skycatalog_V = None
        nvss_skycatalog_W = None

    # (optional) CLEAN image
    if clean_data_availabe:
        clean_data = np.load(args['cleanData'])
        img_clean = clean_data['img_clean']
        img_dirty = clean_data['img_dirty']
        x_plt_CLEAN = clean_data['x_plt_CLEAN_rad']
        y_plt_CLEAN = clean_data['y_plt_CLEAN_rad']

    # (optional) CS image
    if cs_data_available:
        cs_data = np.load(args['csData'])
        img_cs = cs_data['img_clean']

    # save extracted data
    data_file_name = ('./data/' +
                      os.path.splitext(os.path.basename(args['dataFile']))[0] +
                      '_{0}STI_{1:.0f}MHz_{2}Station_{3}Subband.npz'
                      ).format(num_sti, np.mean(freq_subbands_hz) / 1e6,
                               num_station, num_subband)

    npz_data_dict = {
        'freq_subbands_hz': freq_subbands_hz,
        'array_coordinate': array_coordinate,
        'visi_noisy': visi_noisy,
        'RA_rad': sky_ra,
        'DEC_rad': sky_dec,
        'FoV': np.degrees(FoV_radian),
        'skycatalog_intensities': skycatalog_intensities,
        'skycatalog_U': skycatalog_U,
        'skycatalog_V': skycatalog_V,
        'skycatalog_W': skycatalog_W,
        'nvss_skycatalog_intensities': nvss_skycatalog_intensities,
        'nvss_skycatalog_U': nvss_skycatalog_U,
        'nvss_skycatalog_V': nvss_skycatalog_V,
        'nvss_skycatalog_W': nvss_skycatalog_W,
        'x_plt': x_plt_CLEAN if clean_data_availabe else x_plt,
        'y_plt': y_plt_CLEAN if clean_data_availabe else y_plt,
        'z_plt': z_plt,
    }

    if clean_data_availabe:
        npz_data_dict['img_clean'] = img_clean
        npz_data_dict['img_dirty'] = img_dirty

    if cs_data_available:
        npz_data_dict['img_cs'] = img_cs

    if lsqImg_available:
        npz_data_dict['img_lsq'] = img_lsq

    np.savez(data_file_name, **npz_data_dict)
