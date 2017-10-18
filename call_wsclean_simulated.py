"""
call_wsclean_simulated.py: run CLEAN with simulated visibilities
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
from __future__ import division
import numpy as np
import argparse
from visi2ms import run_wsclean, update_visi_msfile, convert_clean_outputs


def parse_args():
    """
    parse input arguments
    :return:
    """
    parser = argparse.ArgumentParser(
        description='''
        Apply wsclean on an updated maasurement set based on simulated visibilities.
        ''',
        epilog='''
        Example usage:
        python2 call_wsclean_simulated.py \
                      --visi_file_name visibility.npz \
                      --msfile_in my_file.ms \
                      --num_station 24 \
                      --num_sti 63 \
                      --intermediate_size 1024 \
                      --output_img_size 512 \
                      --FoV 5 \
                      --output_name_prefix highres \
                      --datacolumn DATA \
                      --freq_channel_min 0 \
                      --freq_channel_max 1 \
                      --max_iter 40000 \
                      --mgain 0.59 \
                      --auto_threshold 3 \
                      --threshold 0.2 \
                      --run_cs 1 \
                      --cs_gain 0.2 \
                      --imag_format png \
                      --dpi 600
        ''',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('--visi_file_name', type=str, required=True,
                        help='Name of the saved numpy array file for the visibilities.')
    parser.add_argument('--msfile_in', type=str, required=True,
                        help='The original ms file, the visibilities of which will be updated.')
    parser.add_argument('--num_station', type=int, required=True,
                        help='Number of stations.')
    parser.add_argument('--num_sti', type=int, required=True,
                        help='Number of integration time.')
    parser.add_argument('--intermediate_size', type=int, required=False, default=1024,
                        help='Intermediated image size used in the wsclean algorithm. Default 1024.')
    parser.add_argument('--output_img_size', type=int, required=False, default=512,
                        help='Final output image size from the wsclean algorithm.')
    parser.add_argument('--FoV', type=float, required=False, default=5,
                        help='Width (and height) of the field of view in degrees.')
    # parser.add_argument('--pixel_size', type=float, required=False, default=1,
    #                     help='Pixel size (in arc second) of the wsclean output images.')
    parser.add_argument('--output_name_prefix', type=str, required=False, default='highres',
                        help='Prefix names used for the wsclean output images.')
    parser.add_argument('--datacolumn', type=str, required=False, default='DATA',
                        help='Name of the data column in the measurement set table.')
    parser.add_argument('--freq_channel_min', type=int, required=True,
                        help='''
                        An integer that specifies the lowest frequency
                        channel used in the input data.''')
    parser.add_argument('--freq_channel_max', type=int, required=True,
                        help='''
                        An integer that specifies the highest frequency
                        channel used in the input data.''')
    parser.add_argument('--max_iter', type=int, required=False, default=10000,
                        help='Maximum number of iterations allowd in wsclean.')
    parser.add_argument('--mgain', type=float, required=False, default=0.59,
                        help='Gain parameter at each of wsclean iteration.')
    parser.add_argument('--auto_threshold', type=float, required=False, default=3,
                        help='Auto-threshold level w.r.t. background noise level in wsclean.')
    parser.add_argument('--threshold', required=False, default=None,
                        help='Threshold level (manual specification).')
    parser.add_argument('--run_cs', type=bool, required=False, default=False,
                        help='Whether to run compressed sensing algorithm or not.')
    parser.add_argument('--cs_gain', type=float, required=False, default=0.2,
                        help='Step size for compressed sensing algorithm.')
    parser.add_argument('--imag_format', type=str, required=False, default='png',
                        help='Exported CLEAN image format.')
    parser.add_argument('--dpi', type=float, required=False, default=600,
                        help='dpi for the exported image.')

    args = vars(parser.parse_args())

    return args


if __name__ == '__main__':
    args = parse_args()
    freq_channel_min = args['freq_channel_min']
    freq_channel_max = args['freq_channel_max']
    num_station = args['num_station']
    num_sti = args['num_sti']
    run_cs = args['run_cs']

    visi = np.load(args['visi_file_name'])['visi']

    num_subband = freq_channel_max - freq_channel_min

    # update ms file with the given visibilities
    reference_ms_file = args['msfile_in']
    modified_ms_file = reference_ms_file[:-3] + '_modi.ms'

    mask_mtx = (1 - np.eye(num_station, dtype=int)).astype(bool)
    antenna2_idx, antenna1_idx = np.meshgrid(np.arange(num_station),
                                             np.arange(num_station))
    antenna1_idx = np.extract(mask_mtx, antenna1_idx)
    antenna2_idx = np.extract(mask_mtx, antenna2_idx)

    print('Updating MS table ...')
    update_visi_msfile(reference_ms_file=reference_ms_file,
                       modified_ms_file=modified_ms_file,
                       visi=visi,
                       antenna1_idx=antenna1_idx,
                       antenna2_idx=antenna2_idx,
                       num_station=num_station)

    # apply wsclean to the updated ms file (CLEAN)
    CLEAN_output_name_prefix = args['output_name_prefix']
    exitcode = run_wsclean(modified_ms_file,
                           channel_range=(freq_channel_min, freq_channel_max),
                           mgain=args['mgain'],
                           FoV=args['FoV'],
                           max_iter=args['max_iter'],
                           auto_threshold=args['auto_threshold'],
                           threshold=args['threshold'],
                           output_img_size=args['output_img_size'],
                           intermediate_size=args['intermediate_size'],
                           output_name_prefix=CLEAN_output_name_prefix,
                           quiet=True, run_cs=False)

    # convert wsclean output FITS images to numpy array
    print('Converting FITS images ...')
    CLEAN_data_file_name = convert_clean_outputs(
        clean_output_prefix=CLEAN_output_name_prefix,
        result_image_prefix='./result/' + CLEAN_output_name_prefix.split('/')[-1],
        result_data_prefix='./data/' + modified_ms_file.split('/')[-1][:-3],
        fig_file_format='png', dpi=600)

    # apply wsclean to the updated ms file (compressed sensing)
    if run_cs:
        CLEAN_output_name_prefix = args['output_name_prefix'] + '_cs'
        exitcode = run_wsclean(modified_ms_file,
                               channel_range=(freq_channel_min, freq_channel_max),
                               mgain=args['mgain'],
                               FoV=args['FoV'],
                               max_iter=args['max_iter'],
                               auto_threshold=args['auto_threshold'],
                               output_img_size=args['output_img_size'],
                               threshold=args['threshold'],
                               intermediate_size=args['intermediate_size'],
                               output_name_prefix=CLEAN_output_name_prefix,
                               quiet=True, run_cs=args['run_cs'],
                               cs_gain=args['cs_gain'])

        # convert wsclean output FITS images to numpy array
        CLEAN_data_file_name = convert_clean_outputs(
            clean_output_prefix=CLEAN_output_name_prefix,
            result_image_prefix='./result/' + CLEAN_output_name_prefix.split('/')[-1],
            result_data_prefix='./data/' + modified_ms_file.split('/')[-1][:-3] + '_cs',
            fig_file_format='png', dpi=300)
