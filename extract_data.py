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
import setup  # to set a few directories
import argparse
import os
import subprocess
import sys
import numpy as np

if sys.version_info[0] > 2:
    sys.exit('Sorry casacore only runs on Python 2.')
else:
    from casacore import tables as casa_tables

from visi2ms import run_wsclean, convert_clean_outputs


def parseArgs():
    """
    parse various input arguments
    :return:
    """
    parser = argparse.ArgumentParser(
        description='''Extract relevant baselines and visibilities from ms file.''',
        epilog='''
        Example usage:
        # Since in the BOOTES ms file station 12, 13, 16 and 17 are not working, 
        # we use 4 more remote stations
        # for BOOTES field (single band)
        python2 extract_data.py \
            --basefile_name 'BOOTES24_SB180-189.2ch8s_SIM' \
            --catalog_file 'skycatalog.npz' \
            --num_channel 1 \
            --time_sampling_step 50 \
            --freq_channel_min 0 \
            --freq_channel_step 1 \
            --number_of_stations 28 \
            --FoV 5 \
            --modi_data_col 1 \
            --mgain 0.1 \
            --trim_data
            
        # for BOOTES field (multi-band)
        python2 extract_data.py \
            --basefile_name 'BOOTES24_SB180-189.2ch8s_SIM' \
            --catalog_file 'skycatalog.npz' \
            --num_channel 8 \
            --time_sampling_step 400 \
            --freq_channel_min 0 \
            --freq_channel_step 1 \
            --number_of_stations 28 \
            --FoV 5 \
            --modi_data_col 1 \
            --mgain 0.1 \
            --trim_data
            
        # for Toothbrush cluster (single-band)
        python2 extract_data.py \
            --basefile_name 'RX42_SB100-109.2ch10s' \
            --catalog_file 'TGSSADR1_7sigma_catalog.npz' \
            --nvss_catalog_file 'NVSS_CATALOG.npz' \
            --num_channel 1 \
            --time_sampling_step 50 \
            --freq_channel_min 19 \
            --freq_channel_step 1 \
            --number_of_stations 36 \
            --FoV 5 \
            --run_cs \
            --cs_gain 0.1 \
            --trim_data 
        ''',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--basefile_name', type=str, required=False,
                        default='BOOTES24_SB180-189.2ch8s_SIM',
                        help='MS file name')
    parser.add_argument('--blueBild_imag_file', type=str, required=False,
                        default=None, help='BlueBild image file')
    parser.add_argument('--catalog_file', type=str, required=False,
                        default=None, help='Catalog *.npz file')
    parser.add_argument('--nvss_catalog_file', type=str, required=False,
                        default=None, help='NVSS catalog *.npz file')
    parser.add_argument('--num_channel', type=int, required=False,
                        default=1, help='Number of subbands')
    parser.add_argument('--time_sampling_step', type=int, required=False,
                        default=50, help='Stepsize between adjacent integration time')
    parser.add_argument('--time_sampling_end', type=float, required=False,
                        default=float('inf'), help='Last integration time to be extracted')
    parser.add_argument('--freq_channel_min', type=int, required=False,
                        default=0, help='Subband frequency starting index')
    parser.add_argument('--freq_channel_step', type=int, required=False,
                        default=1, help='Stepsize between adjacent subbands')
    parser.add_argument('--number_of_stations', type=int, required=False,
                        default=24, help='Number of stations')
    parser.add_argument('--FoV', type=float, required=False,
                        default=5, help='Field of view in degrees')
    parser.add_argument('--modi_data_col', type=bool, required=False,
                        default=False, help='Whether modify the data column or not')
    parser.add_argument('--mgain', type=float, required=False,
                        default=0.59, help='Gain of the major iteration in CLEAN')
    parser.add_argument('--run_cs', default=False, action='store_true',
                        help='if present, then run compressive sensning')
    parser.add_argument('--cs_gain', type=float, required=False,
                        default=0.1, help='Gain used by the CS algorithm')
    parser.add_argument('--trim_data', default=False, action='store_true',
                        help='If present, then the data is trimmed (due to failed stations)')

    args = vars(parser.parse_args())
    if args['blueBild_imag_file'] == 'None':
        args['blueBild_imag_file'] = None

    if args['catalog_file'] == 'None':
        args['catalog_file'] = None

    if args['nvss_catalog_file'] == 'None':
        args['nvss_catalog_file'] = None

    return args


if __name__ == '__main__':
    args = parseArgs()
    data_root_path = os.environ['DATA_ROOT_PATH']
    blueBild_root_path = os.environ['BLUEBILD_ROOT_PATH']
    basefile_name = args['basefile_name']
    if args['blueBild_imag_file'] is not None:
        blueBild_imag_file = blueBild_root_path + args['blueBild_imag_file']
    else:
        blueBild_imag_file = None
    if args['catalog_file'] is not None:
        catalog_file = blueBild_root_path + args['catalog_file']
    else:
        catalog_file = None
    if args['nvss_catalog_file'] is not None:
        nvss_catalog_file = blueBild_root_path + args['nvss_catalog_file']
    else:
        nvss_catalog_file = None

    ms_file_name = data_root_path + basefile_name + '.ms'
    ms_data_file = blueBild_root_path + basefile_name + '.hdf5'

    num_channel = args['num_channel']
    time_sampling_step = args['time_sampling_step']

    time_sampling_end = \
        casa_tables.taql('SELECT DISTINCT TIME_CENTROID FROM '
                         '{ms_file_name}'.format(ms_file_name=ms_file_name)).getcol('TIME_CENTROID').size
    time_sampling_end = int(min(time_sampling_end, args['time_sampling_end']))

    # frequency channels in [freq_channel_min, freq_channel_max) are selected
    freq_channel_min = args['freq_channel_min']
    freq_channel_max = freq_channel_min + num_channel
    freq_channel_step = args['freq_channel_step']
    number_of_stations = args['number_of_stations']  # 24 <- for BOOTES field
    FoV = args['FoV']  # field of view in degree
    fig_file_format = 'png'
    dpi = 900

    time_range = '0:{0}:{1}'.format(time_sampling_end, time_sampling_step)

    # ===== extract relevant data from the original ms file =====
    original_table = casa_tables.table(ms_file_name)
    # extract the subtable that corresponds to the selected integration time indices
    sub_table_file_name = '{basefile_name}_every{time_sampling_step}th.ms'.format(
        basefile_name=basefile_name,
        time_sampling_step=time_sampling_step
    )
    sub_table_full_name = data_root_path + sub_table_file_name

    antenna1_lst = \
        np.sort(casa_tables.taql('select distinct ANTENNA1 from {ms_file_name}'.format(
            ms_file_name=ms_file_name
        )).getcol('ANTENNA1'))
    antenna2_lst = \
        np.sort(casa_tables.taql('select distinct ANTENNA2 from {ms_file_name}'.format(
            ms_file_name=ms_file_name
        )).getcol('ANTENNA2'))
    assert antenna1_lst.size == antenna2_lst.size
    number_of_stations = min(number_of_stations, antenna1_lst.size)
    args['number_of_stations'] = number_of_stations

    antenna1_limit = antenna1_lst[number_of_stations - 1]
    antenna2_limit = antenna2_lst[number_of_stations - 1]

    taql_cmd_str = 'select from {ms_file_name} where TIME in ' \
                   '(select distinct TIME from {ms_file_name} limit {time_range})' \
                   'and ANTENNA1<={antenna1_limit} ' \
                   'and ANTENNA2<={antenna2_limit} ' \
                   'giving {sub_table_name}'.format(ms_file_name=ms_file_name,
                                                    time_range=time_range,
                                                    antenna1_limit=antenna1_limit,
                                                    antenna2_limit=antenna2_limit,
                                                    sub_table_name=sub_table_full_name
                                                    )
    casa_tables.taql(taql_cmd_str)

    if args['modi_data_col']:
        # put back source in the DATA column
        with casa_tables.table(sub_table_full_name, readonly=False) as sub_table:
            # update DATA column
            sub_table.putcol('DATA', sub_table.getcol('DATA') +
                             0.001 * sub_table.getcol('DATA_SIMULATED'))
    else:
        # use this for normal setup
        with casa_tables.table(sub_table_full_name, readonly=False) as sub_table:
            sub_table.putcol('DATA', sub_table.getcol('DATA'))  # no effect in fact

    '''
    ====== CALL WSCLEAN ======
    '''
    intermediate_size = 1500
    final_image_size = 1024
    mgain = args['mgain']
    output_prefix = 'highres'  # pre-fix of the wsclean output images
    max_iter = 40000  # maximum number of CLEAN iterations

    exitcode = run_wsclean(ms_file=sub_table_full_name,
                           channel_range=(freq_channel_min, freq_channel_max),
                           mgain=mgain, FoV=FoV, max_iter=max_iter, auto_threshold=3,
                           output_img_size=final_image_size,
                           intermediate_size=intermediate_size,
                           output_name_prefix=data_root_path + output_prefix,
                           quiet=True)
    if exitcode:
        raise RuntimeError('wsCLEAN could not run!')

    '''
    ===== CONVERT FITS IMAGE TO NUMPY ARRAY =====
    '''
    CLEAN_data_file = \
        convert_clean_outputs(clean_output_prefix=data_root_path + output_prefix,
                              result_image_prefix='./result/' + output_prefix,
                              result_data_prefix='./data/' + sub_table_file_name[:-3],
                              fig_file_format=fig_file_format, dpi=dpi)

    '''
    ====== CALL WSCLEAN (Compressed Sensing)======
    '''
    run_cs = args['run_cs']
    if run_cs:
        output_prefix_cs = output_prefix + '_cs'  # pre-fix of the wsclean output images
        exitcode = run_wsclean(ms_file=sub_table_full_name,
                               channel_range=(freq_channel_min, freq_channel_max),
                               mgain=mgain, FoV=FoV, max_iter=max_iter, auto_threshold=3,
                               output_img_size=final_image_size,
                               intermediate_size=intermediate_size,
                               output_name_prefix=data_root_path + output_prefix_cs,
                               quiet=True, run_cs=run_cs, cs_gain=args['cs_gain'])
        if exitcode:
            raise RuntimeError('wsCLEAN could not run!')

        '''CONVERT FITS IMAGE TO NUMPY ARRAY'''

        CS_data_file = \
            convert_clean_outputs(clean_output_prefix=data_root_path + output_prefix_cs,
                                  result_image_prefix='./result/' + output_prefix_cs,
                                  result_data_prefix='./data/' + sub_table_file_name[:-3] + '_cs',
                                  fig_file_format=fig_file_format, dpi=dpi)

    '''
    ===== EXTRACT BLUEBILD IMAGES =====
    '''
    bash_cmd = 'export PATH="/usr/bin:$PATH" && ' \
               'export PATH="$HOME/anaconda2/bin:$PATH" && ' \
               'python2 real_data.py ' \
               '--dataFile {data_file} ' \
               '--timeRange np.r_[:{time_sampling_end}:{time_sampling_step}] ' \
               '--freqRange np.r_[{freq_channel_min}:{freq_channel_max}:{freq_channel_step}] ' \
               '--stationCount {number_of_stations} ' \
               '--FoV {FoV} ' \
               '--imageWidth {final_image_size} ' \
               '--catalog {catalog_file} ' \
               '--nvss_catalog {nvss_catalog} ' \
               '--cleanData {CLEAN_data_file} ' \
               '{cs_data_file} '.format(
        data_file=ms_data_file,
        time_sampling_end=time_sampling_end,
        time_sampling_step=time_sampling_step,
        freq_channel_min=freq_channel_min,
        freq_channel_max=freq_channel_max,
        freq_channel_step=freq_channel_step,
        number_of_stations=number_of_stations,
        FoV=FoV,
        final_image_size=final_image_size,
        blueBild_imag_file=blueBild_imag_file,
        nvss_catalog=nvss_catalog_file,
        catalog_file=catalog_file,
        CLEAN_data_file=CLEAN_data_file,
        cs_data_file='--csData ' + CS_data_file if run_cs else ''
    )
    if args['trim_data']:
        bash_cmd += ' --trim_data '

    if subprocess.call(bash_cmd, shell=True):
        raise RuntimeError('Failed to save converted data file.')
