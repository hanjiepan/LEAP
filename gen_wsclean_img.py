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
import os
import subprocess
import sys
import getopt

if sys.version_info[0] > 2:
    sys.exit('Sorry casacore only runs on Python 2.')
else:
    from casacore import tables as casa_tables

from visi2ms import run_wsclean, convert_clean_outputs

if __name__ == '__main__':
    sti_index_bg = 0
    sti_index_end = ''
    sti_index_step = 1
    station_count = 24
    res_img_sz = 1024
    res_img_name = 'highres'
    # parse arguments
    argv = sys.argv[1:]

    try:
        opts, args = getopt.getopt(argv, "hs:e:j:c:o:n:",
                                   ["start=", "end=", "stepsize=",
                                    "stationcount=",
                                    "outsize=", "outname="])
    except getopt.GetoptError:
        print('gen_wsclean_img.py -s <sti_start_index> '
              '-e <sti_end_index> -j <sti_step_size> '
              '-c <number_of_stations> -o <output_img_size> '
              '-n <output_img_name>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('gen_wsclean_img.py -s <sti_start_index> '
                  '-e <sti_end_index> -j <sti_step_size> '
                  '-c <number_of_stations> -o <output_img_size> '
                  '-n <output_img_name>')
            sys.exit()
        elif opt in ('-s', '--start'):
            sti_index_bg = int(arg)
        elif opt in ('-e', '--end'):
            sti_index_end = int(arg)
        elif opt in ('-j', '--stepsize'):
            sti_index_step = int(arg)
        elif opt in ('-c', '--stationcount'):
            station_count = int(arg)
        elif opt in ('-o', '--outsize'):
            res_img_sz = int(arg)
        elif opt in ('-n', '--outname'):
            res_img_name = arg

    data_root_path = os.environ['DATA_ROOT_PATH']
    blueBild_root_path = os.environ['BLUEBILD_ROOT_PATH']
    basefile_name = 'BOOTES24_SB180-189.2ch8s_SIM'
    ms_file_name = data_root_path + basefile_name + '.ms'
    # frequency channels in [freq_channel_min, freq_channel_max) are selected
    freq_channel_min = 0
    freq_channel_max = 1
    FoV = 5  # field of view in degree
    fig_file_format = 'png'
    dpi = 600

    time_range = '{sti_index_bg}:{sti_index_end}:{sti_index_step}'.format(
        sti_index_bg=sti_index_bg,
        sti_index_end=sti_index_end,
        sti_index_step=sti_index_step
    )

    print('STI indices: {}'.format(time_range))
    print('Number of stations: {}'.format(station_count))
    print('Output image size: {}'.format(res_img_sz))

    # ===== extract relevant data from the original ms file =====
    original_table = casa_tables.table(ms_file_name)
    # extract the subtable that corresponds to the selected integration time indices
    sub_table_file_name = '{basefile_name}_every{time_sampling_step}th.ms'.format(
        basefile_name=basefile_name,
        time_sampling_step=sti_index_step
    )
    sub_table_full_name = data_root_path + sub_table_file_name

    taql_cmd_str = 'select from {ms_file_name} where TIME in ' \
                   '(select distinct TIME from {ms_file_name} limit {time_range})' \
                   'and ANTENNA1<{number_of_stations} ' \
                   'and ANTENNA2<{number_of_stations} ' \
                   'giving {sub_table_name}'.format(ms_file_name=ms_file_name,
                                                    time_range=time_range,
                                                    number_of_stations=station_count,
                                                    sub_table_name=sub_table_full_name
                                                    )
    casa_tables.taql(taql_cmd_str)

    with casa_tables.table(sub_table_full_name, readonly=False) as sub_table:
        # update DATA column
        sub_table.putcol('DATA', sub_table.getcol('DATA') +
                         0.001 * sub_table.getcol('DATA_SIMULATED'))

    '''
    ====== CALL WSCLEAN ======
    '''
    intermediate_size = int(res_img_sz * 1.5)
    final_image_size = res_img_sz
    mgain = 0.59
    output_prefix = res_img_name  # pre-fix of the wsclean output images
    max_iter = 40000  # maximum number of CLEAN iterations

    exitcode = run_wsclean(ms_file=sub_table_full_name,
                           channel_range=(freq_channel_min, freq_channel_max),
                           mgain=mgain, FoV=FoV, max_iter=max_iter, auto_threshold=3,
                           output_img_size=final_image_size,
                           intermediate_size=intermediate_size,
                           output_name_prefix=data_root_path + output_prefix)
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
