"""
formDataStructures.py: reads the provided MS file and generates a HDF5 file
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
------------------------------------------------------------
formDataStructures.py reads the provided MS file and generates a HDF5 file containing the following data structures:
    - TIME_MAP: pd.DataFrame with columns (timeIndex [int], time [UTC])
    - FREQ_MAP: pd.DataFrame with columns (freqIndex [int], freq [Hz])
    - STATION_INFO: pd.DataFrame with columns (stationID [int], antennaID [int], x [m], y [m], z [m]) in ITRF format
    - POINTING_DIRECTION: pd.DataFrame with columns (longitude [rad], latitude [rad]) in J2000 format
    - DATA: directory-like structure containing several files for each (timeIndex,freqIndex) pair:
      Files are named as DATA/<timeIndex>/<freqIndex>/<name-of-data-structure>.

      - STATION_UVW: pd.DataFrame with columns (stationID [int], antennaID [int], u [m], v [m], w [m]) in UVW format
      - STATION_ID: pd.Series with station ordering used in S,G, and W
      - G: Gram matrix of the different stations.
      - S: Covariance matrix of the different stations WITHOUT the diagonal terms (which are set to 0)
      - W: pd.DataFrame with columns (stationID [int], antennaCount [int], weights [complex])
        For performance reasons, the column 'antennaCount' is labeled '-1' and is stored as a complex number.
"""

from __future__ import print_function, division

import argparse
import datetime
import itertools
import os
import sys

import casacore.tables as ct
import joblib
import numpy as np
import pandas as pd
import scipy.constants as sc

import coordinateConversion as cc


def printProgress():
    """
    Print to sys.stdout the name of the calling function along with a timestamp.
    """
    dt = datetime.datetime.now()
    info = '{year:=04d}-{month:=02d}-{day:=02d}/{hour:02d}:{minute:=02d}:{second:=02d}  {function}'.format(
        year=dt.year, month=dt.month, day=dt.day,
        hour=dt.hour, minute=dt.minute, second=dt.second,
        function=sys._getframe(1).f_code.co_name
    )
    print(info)


def openHDF5(args):
    """
    Open the output HDF5 file in append mode.

    :param args: output of parseArgs()
    :return: file-descriptor to HDF5 file
    """
    store = pd.HDFStore(
        path=args['dataFile'],
        mode='r+'
    )
    return store


def dataKey(fileName, timeIndex, freqIndex):
    """
    Return filename that should be used to write data to a HDF5 file.

    :param fileName: basename of the file
    :param timeIndex: time index
    :param freqIndex: frequency index
    :return: string with HDF5 key name
    """
    key = 'DATA/t{t:=04d}/f{f:=02d}/{n}'.format(t=timeIndex, f=freqIndex, n=fileName)
    return key


def parseArgs():
    """
    Parse command-line arguments.

    :return: dictionary of valid arguments
    """
    printProgress()

    def parseRange(code):
        range = eval(code)
        return np.sort(range)

    parser = argparse.ArgumentParser(
        description="""
Read MS file and produce HDF5 file to be fed into generateImages.py
            """,
        epilog="""
Example usage: python2 formDataStructures.py --msFile BOOTES24_SB180-189.2ch8s.ms
                                             --dataFile BOOTES24_SB180-189.2ch8s.hdf5
                                             --timeRange np.r_[0:5, 10:15, 18, 359]
                                             --freqRange np.r_[3, 5, 17]
                                             --coreCount -3
                                             --verbosity high
                                             --modi_data_col 1  # <- for bootes field
            """,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--msFile', type=str, required=True, help='MS File')
    parser.add_argument('--dataFile', type=str, required=True, help='Destination File')
    parser.add_argument('--timeRange', type=str, required=True, help="""
List of (integer) time indices to process.
The format is np.r_[<write all indices here>].
    """)
    parser.add_argument('--freqRange', type=str, required=True, help="""
List of (integer) frequency indices to process.
The format is np.r_[<write all indices here>].
    """)
    parser.add_argument('--coreCount', type=int, required=True, help="""
Number of processor cores to use for parallelizable steps.
Positive and negative values can be given.
Ex: 1 = 1 core, 2 = 2 core, ..., -1 = all cores, -2 = all but one, ...
    """)
    parser.add_argument('--verbosity', type=str, required=True, choices=['low', 'high'], help="""
Amount of logging to do.
'low' = just log the different stages.
'high' = log stages & all parallel steps.
    """)
    parser.add_argument('--modi_data_col', type=bool, required=False,
                        default=False, help='Whether to modify the DATA column or not.')

    args = vars(parser.parse_args())

    args['msFile'] = os.path.abspath(args['msFile'])
    args['dataFile'] = os.path.abspath(args['dataFile'])
    args['timeRange'] = parseRange(args['timeRange'])
    args['freqRange'] = parseRange(args['freqRange'])
    if args['verbosity'] == 'low':
        args['verbosity'] = 5
    else:
        args['verbosity'] = 11

    return args


def prepareFileSystem(args):
    """
    Create output HDF5 file.

    :param args: output of parseArgs()
    """
    printProgress()

    store = pd.HDFStore(
        path=args['dataFile'],
        mode='w'
    )
    store.close()


def create_POINT_DIR(args):
    """
    Write POINT_DIR to HDF5 file.

    :param args: output of parseArgs()
    """
    printProgress()

    direction = ct.taql(
        'SELECT REFERENCE_DIR from {msFile}::FIELD'.format(msFile=args['msFile'])
    ).getcell('REFERENCE_DIR', 0)

    POINTING_DIRECTION = pd.DataFrame(
        data=direction,
        columns=['longitude', 'latitude']
    )

    store = openHDF5(args)
    store['POINTING_DIRECTION'] = POINTING_DIRECTION
    store.close()


def create_TIME_MAP(args):
    """
    Write TIME_MAP to the HDF5 file.

    :param args: output of parseArgs()
    """
    printProgress()

    time = ct.taql(
        'SELECT DISTINCT TIME_CENTROID FROM {msFile}'.format(msFile=args['msFile'])
    ).getcol('TIME_CENTROID')
    TIME_MAP = pd.DataFrame({
        'timeIndex': np.arange(len(time)),
        'time': np.sort(time)
    }).set_index('timeIndex')
    TIME_MAP = TIME_MAP.loc[args['timeRange']]

    store = openHDF5(args)
    store['TIME_MAP'] = TIME_MAP
    store.close()


def create_FREQ_MAP(args):
    """
    Write FREQ_MAP to the HDF5 file.

    :param args: output of parseArgs()
    """
    printProgress()

    frequency = ct.taql(
        'SELECT CHAN_FREQ FROM {msFile}::SPECTRAL_WINDOW'.format(msFile=args['msFile'])
    ).getcol('CHAN_FREQ').flatten()
    FREQ_MAP = pd.DataFrame({
        'freqIndex': np.arange(len(frequency)),
        'frequency': np.sort(frequency)
    }).set_index('freqIndex')
    FREQ_MAP = FREQ_MAP.loc[args['freqRange']]

    store = openHDF5(args)
    store['FREQ_MAP'] = FREQ_MAP
    store.close()


def create_STATION_INFO(args):
    """
    Write STATION_INFO to the HDF5 file.

    :param args: parseArgs()
    """
    printProgress()

    masterTable = ct.taql(
        'SELECT ANTENNA_ID, POSITION, ELEMENT_OFFSET, ELEMENT_FLAG FROM {msFile}::LOFAR_ANTENNA_FIELD'.format(
            msFile=args['msFile'])
    )

    def process(table):
        stationID = table.getcol('ANTENNA_ID')[0]
        stationCentroid = table.getcol('POSITION')
        antennaOffsets = table.getcol('ELEMENT_OFFSET').squeeze()
        antennaFlags = table.getcol('ELEMENT_FLAG').squeeze()

        antennaPositions = stationCentroid + antennaOffsets
        antennaFlagged = np.any(antennaFlags, axis=1)

        station_info = pd.DataFrame({
            'stationID': stationID,
            'antennaID': np.arange(len(antennaFlagged)),
            'x': antennaPositions[:, 0],
            'y': antennaPositions[:, 1],
            'z': antennaPositions[:, 2],
            'flagged': antennaFlagged
        })
        station_info = station_info[['stationID', 'antennaID', 'x', 'y', 'z', 'flagged']]
        return station_info[np.logical_not(station_info['flagged'])].drop('flagged', axis=1)

    STATION_INFO = pd.concat(
        [process(subTable) for subTable in ct.tableiter(masterTable, 'ANTENNA_ID', sort=True)],
        ignore_index=True
    )

    store = openHDF5(args)
    store['STATION_INFO'] = STATION_INFO
    store.close()


def create_STATION_UVW_process(timeIndex, freqIndex, args):
    """
    Inner-loop of create_STATION_UVW().
    This function should be placed in create_STATION_UVW(), but due to limitations of joblib, it has been placed here.

    :param timeIndex: time index
    :param freqIndex: frequency index
    :param args: output of parseArgs()
    :return: tuple (timeIndex, freqIndex, STATION_INFO_UVW)
    """
    import casacore.measures as cm
    import casacore.quanta as cq

    def ITRF_to_J2000(time, x, y, z):
        dm = cm.measures()
        dm.do_frame(dm.epoch('UTC', cq.quantity(time, 's')))

        ITRF_position = dm.position(
            rf='ITRF',
            v0=cq.quantity(x, 'm'),
            v1=cq.quantity(y, 'm'),
            v2=cq.quantity(z, 'm')
        )
        dm.do_frame(ITRF_position)

        ITRFLL_position = dm.measure(ITRF_position, 'ITRFLL')
        height = ITRFLL_position['m2']

        ITRFLL_direction = dm.direction('ITRFLL', v0=ITRFLL_position['m0'], v1=ITRFLL_position['m1'])

        J2000_direction = dm.measure(ITRFLL_direction, 'J2000')
        J2000_position = dm.position(rf='ITRF', v0=J2000_direction['m0'], v1=J2000_direction['m1'], v2=height)

        (az, el, r) = (J2000_position['m0']['value'], J2000_position['m1']['value'], J2000_position['m2']['value'])
        return (az, el, r)

    ITRF_to_J2000_vec = np.vectorize(ITRF_to_J2000)

    store = openHDF5(args)
    time = store['TIME_MAP'].loc[timeIndex].values
    wavelength = sc.speed_of_light / store['FREQ_MAP'].loc[freqIndex].values
    STATION_INFO_CART = store['STATION_INFO']
    store.close()

    (az, el, r) = ITRF_to_J2000_vec(
        time,
        STATION_INFO_CART['x'].values,
        STATION_INFO_CART['y'].values,
        STATION_INFO_CART['z'].values
    )
    (x, y, z) = cc.sph2cart(az, el, r)

    # pointing direction in radian (RA, DEC)
    pointingDirection = ct.taql('select REFERENCE_DIR from {msFile}::FIELD'.format(msFile=args['msFile'])).getcol(
        'REFERENCE_DIR').squeeze()
    M = cc.J2000_to_UVW_operator(*pointingDirection)
    # TODO: different normalization for other datasets. Add a switch to control this.
    (u, v, w) = M.dot(np.vstack((x, y, z))) / wavelength

    STATION_INFO_UVW = pd.DataFrame({
        'stationID': STATION_INFO_CART['stationID'],
        'antennaID': STATION_INFO_CART['antennaID'],
        'u': u,
        'v': v,
        'w': w
    })
    STATION_INFO_UVW = STATION_INFO_UVW[['stationID', 'antennaID', 'u', 'v', 'w']]
    return timeIndex, freqIndex, STATION_INFO_UVW


def create_STATION_UVW(args):
    """
    Write /DATA/<>/<>/STATION_UVW to HDF5 file.
    The computations are done in parallel, hence it may be necessary to tune the --freqIndex option to limit the data
    output and eventual crashes.

    :param args: output of parseArgs()
    """
    printProgress()

    data = joblib.Parallel(n_jobs=args['coreCount'], verbose=args['verbosity'])(
        joblib.delayed(create_STATION_UVW_process)(timeIndex, freqIndex, args)
        for (timeIndex, freqIndex) in itertools.product(args['timeRange'], args['freqRange'])
    )

    store = openHDF5(args)
    for (timeIndex, freqIndex, STATION_UVW) in data:
        store[dataKey('STATION_UVW', timeIndex, freqIndex)] = STATION_UVW
    store.close()


def create_S(args):
    """
    Write Covariance matrices to the HDF5 file.

    :param args: output of parseArgs()
    """
    printProgress()

    store = openHDF5(args)
    TIME_MAP = store['TIME_MAP']

    if args['modi_data_col']:
        def process(table):
            # Flag Parameter
            flagCol = table.getcol('FLAG')
            flags = np.sign(np.sum(flagCol[:, :, [0, 3]], axis=2))

            # Stokes Intensity Parameter #################################
            # Code used for TammoJan's specially crafted MS file
            dataColSimulated = table.getcol('DATA_SIMULATED')
            dataColResidual = table.getcol('DATA')
            dataCol = dataColResidual + 0.001 * dataColSimulated

            stokesIntensity = np.average(dataCol[:, :, [0, 3]], axis=2)
            stokesIntensity[flags != 0] = 0

            # Visibility Cube
            antenna1s = table.getcol('ANTENNA1')
            antenna2s = table.getcol('ANTENNA2')
            subbandCount = np.shape(stokesIntensity)[-1]
            cube = 1j * np.zeros([
                np.size(np.unique(antenna1s)),
                np.size(np.unique(antenna2s)),
                subbandCount
            ])
            for freq in np.arange(0, subbandCount):
                cube[antenna1s, antenna2s, freq] = stokesIntensity[:, freq]
                # Making cube slices Hermitian (not the case in the MS files)
                cube[:, :, freq] += cube[:, :, freq].conj().transpose()
                # NOTE FOR FUTURE: if autocorrelations are not set to 0, the diagonal must be subtracted.
            return cube
    else:
        def process(table):
            # Flag Parameter
            flagCol = table.getcol('FLAG')
            flags = np.sign(np.sum(flagCol[:, :, [0, 3]], axis=2))

            # Stokes Intensity Parameter #################################
            # Code used for regular MS files
            dataCol = table.getcol('DATA')
            # ############################################################

            stokesIntensity = np.average(dataCol[:, :, [0, 3]], axis=2)
            stokesIntensity[flags != 0] = 0

            # Visibility Cube
            antenna1s = table.getcol('ANTENNA1')
            antenna2s = table.getcol('ANTENNA2')
            subbandCount = np.shape(stokesIntensity)[-1]
            cube = 1j * np.zeros([
                np.size(np.unique(antenna1s)),
                np.size(np.unique(antenna2s)),
                subbandCount
            ])
            for freq in np.arange(0, subbandCount):
                cube[antenna1s, antenna2s, freq] = stokesIntensity[:, freq]
                # Making cube slices Hermitian (not the case in the MS files)
                cube[:, :, freq] += cube[:, :, freq].conj().transpose()
                # NOTE FOR FUTURE: if autocorrelations are not set to 0, the diagonal must be subtracted.
            return cube

    masterTable = ct.table(args['msFile'], ack=False)
    subTables = (
        subTable
        for subTable in ct.tableiter(masterTable, 'TIME', sort=True)
        if subTable.getcell('TIME', 0) in TIME_MAP.values
    )
    cubes = [process(subTable) for subTable in subTables]

    for (timeIndex, cube) in zip(args['timeRange'], cubes):
        for freqIndex in args['freqRange']:
            S = pd.DataFrame(
                data=cube[:, :, freqIndex],
            )
            store[dataKey('S', timeIndex, freqIndex)] = S
    store.close()


def filter_STATION_INFO_and_S(args):
    """
    Once the STATION_UVWs and Ss are created, we have to filter them to keep the stations that are in common to both
    data structures.
    In essence, we are removing both flagged stations (in STATION_UVW) and stations who have 'no data'
    (0-filled lines & columns in S).

    Also write the station identifiers that are left at the end in /DATA/<>/<>/STATION_ID.

    :param args: output of parseArgs()
    """
    printProgress()

    def process(timeIndex, freqIndex, args):
        store = openHDF5(args)

        # Shrink S
        # S = store[dataKey('S', timeIndex, freqIndex)]
        # TODO: confirm with Sepand
        # validStationIDs = np.where(~np.all(S == 0, axis=0))
        # S = S.loc[np.r_[validStationIDs], np.r_[validStationIDs]]

        # Shrink STATION_UVW
        STATION = store[dataKey('STATION_UVW', timeIndex, freqIndex)]
        # TODO: confirm with Sepand
        # STATION = STATION[STATION['stationID'].isin(*validStationIDs)]

        # Form STATION_ID to know which stations are used in S,G,W
        STATION_ID = pd.Series(np.unique(STATION['stationID']))

        store.close()
        # return (timeIndex, freqIndex, S, STATION, STATION_ID)
        return (timeIndex, freqIndex, STATION_ID)

    data = [
        process(timeIndex, freqIndex, args)
        for (timeIndex, freqIndex) in itertools.product(args['timeRange'], args['freqRange'])
        ]

    store = openHDF5(args)
    for (timeIndex, freqIndex, STATION_ID) in data:
    # for (timeIndex, freqIndex, S, STATION, STATION_ID) in data:
        # store[dataKey('S', timeIndex, freqIndex)] = S
        # store[dataKey('STATION_UVW', timeIndex, freqIndex)] = STATION
        store[dataKey('STATION_ID', timeIndex, freqIndex)] = STATION_ID
    store.close()


def create_W(args):
    """
    Write the Ws to the HDF5 file.

    :param args: output of parseArgs()
    """
    printProgress()

    def computeBeamShape(stationID, station, outputLength):
        antennaCount = len(station)
        W = np.exp(-1j * 2 * np.pi * station['w'].values) / antennaCount

        if antennaCount != outputLength:
            padLength = outputLength - antennaCount
            W = np.hstack((W, np.zeros(padLength)))

        beamshapeCoeffs = pd.DataFrame(
            data=np.hstack((np.array([[antennaCount]]), W[np.newaxis, :])),
            index=[stationID],
            columns=np.arange(-1, outputLength)
        )
        beamshapeCoeffs.index.name = 'stationID'

        return beamshapeCoeffs

    def process(timeIndex, freqIndex, args):
        store = openHDF5(args)
        STATION = store[dataKey('STATION_UVW', timeIndex, freqIndex)]
        store.close()

        maxAntennaCount = max([len(station['antennaID']) for (_, station) in STATION.groupby(by='stationID')])
        W = pd.concat(
            [computeBeamShape(stationID, station, maxAntennaCount) for (stationID, station) in
             STATION.groupby(by='stationID')],
            ignore_index=False
        )

        return (timeIndex, freqIndex, W)

    data = [process(timeIndex, freqIndex, args) for (timeIndex, freqIndex) in
            itertools.product(args['timeRange'], args['freqRange'])]

    store = openHDF5(args)
    for (timeIndex, freqIndex, W) in data:
        store[dataKey('W', timeIndex, freqIndex)] = W
    store.close()


def create_G_process(timeIndex, freqIndex, args):
    """
    Inner-loop of create_G().
    This function should be placed in create_G(), but due to limitations of joblib, it has been placed here.

    :param timeIndex: time index
    :param freqIndex: frequency index
    :param args: output of parseArgs()
    :return: tuple (timeIndex, freqIndex, G)
    """

    store = openHDF5(args)
    W = store[dataKey('W', timeIndex, freqIndex)]
    STATION_UVW = store[dataKey('STATION_UVW', timeIndex, freqIndex)]
    STATION_ID = store[dataKey('STATION_ID', timeIndex, freqIndex)]
    store.close()

    stationCount = len(STATION_ID)
    Gmat = np.zeros((stationCount, stationCount), dtype=complex)

    for (l, k) in itertools.combinations_with_replacement(range(stationCount), 2):
        STATION_1 = STATION_UVW[STATION_UVW['stationID'] == STATION_ID[l]]
        STATION_2 = STATION_UVW[STATION_UVW['stationID'] == STATION_ID[k]]

        baseline_U = STATION_1['u'].values[:, np.newaxis] - STATION_2['u'].values
        baseline_V = STATION_1['v'].values[:, np.newaxis] - STATION_2['v'].values
        baseline_W = STATION_1['w'].values[:, np.newaxis] - STATION_2['w'].values

        baseline_norm = 2 * np.pi * np.sqrt(baseline_U ** 2 + baseline_V ** 2 + baseline_W ** 2)
        baseline_sinc = 4 * np.pi * np.sinc(baseline_norm / np.pi)

        antennaCount_1 = W.loc[STATION_ID[l], -1].real.astype(int)
        antennaCount_2 = W.loc[STATION_ID[k], -1].real.astype(int)
        W_1 = W.loc[STATION_ID[l], 0:(antennaCount_1 - 1)].values
        W_2 = W.loc[STATION_ID[k], 0:(antennaCount_2 - 1)].values

        Gmat[l, k] = (W_1.conj()).dot(baseline_sinc).dot(W_2)

    Gmat += np.triu(Gmat, 1).conj().transpose()
    G = pd.DataFrame(
        data=Gmat,
        index=STATION_ID,
        columns=STATION_ID
    )
    return (timeIndex, freqIndex, G)


def create_G(args):
    """
    Write the Gram matrices to the HDF5 file.
    The computations are done in parallel, hence it may be necessary to tune the --freqIndex option to limit the data
    output and eventual crashes.

    :param args: output of parseArgs()
    """
    printProgress()

    data = joblib.Parallel(n_jobs=args['coreCount'], verbose=args['verbosity'])(
        joblib.delayed(create_G_process)(timeIndex, freqIndex, args)
        for (timeIndex, freqIndex) in itertools.product(args['timeRange'], args['freqRange'])
    )

    store = openHDF5(args)
    for (timeIndex, freqIndex, G) in data:
        store[dataKey('G', timeIndex, freqIndex)] = G
    store.close()


if __name__ == '__main__':
    args = parseArgs()
    prepareFileSystem(args)

    # (time,freq)-independent steps
    create_POINT_DIR(args)
    create_TIME_MAP(args)
    create_FREQ_MAP(args)
    create_STATION_INFO(args)

    # (time,freq)-dependent steps
    create_STATION_UVW(args)
    create_S(args)

    filter_STATION_INFO_and_S(args)
