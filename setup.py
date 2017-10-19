"""
setup.py: setup various data folder that may be different from computer to computer
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
import os

# can be: home_mac, lcav_mac, ibm_mac, mac_virtualbox,
# windows_virtualbox, zc2, iccluster, srv1, lcavsrv1
machine_type = 'lcavsrv1'

if machine_type == 'home_mac':

    os.environ['DATA_ROOT_PATH'] = '/Users/hjpan/Google\ Drive/RadioAstData/'
    os.environ['PROCESSED_DATA_ROOT_PATH'] = '/Users/hjpan/Google\ Drive/RadioAstData/'
    os.environ['COMPUTE_BACK_END'] = 'cpu'
    os.environ['THEANO_FLAGS'] = 'device=cpu'
    print('data_root_path is set to          : {0}'.format(os.environ['DATA_ROOT_PATH']))
    print('processed_data_root_path is set to: {0}'.format(os.environ['PROCESSED_DATA_ROOT_PATH']))

elif machine_type == 'lcav_mac':

    os.environ['DATA_ROOT_PATH'] = '/Users/pan/Google\ Drive/RadioAstData/'
    os.environ['PROCESSED_DATA_ROOT_PATH'] = '/Users/pan/Google\ Drive/RadioAstData/'
    os.environ['COMPUTE_BACK_END'] = 'cpu'
    os.environ['THEANO_FLAGS'] = 'device=cpu'
    print('data_root_path is set to          : {0}'.format(os.environ['DATA_ROOT_PATH']))
    print('processed_data_root_path is set to: {0}'.format(os.environ['PROCESSED_DATA_ROOT_PATH']))

elif machine_type == 'mac_virtualbox':

    os.environ['DATA_ROOT_PATH'] = '/home/hpa/Documents/Data/'
    os.environ['PROCESSED_DATA_ROOT_PATH'] = '/media/sf_RadioAstData/'
    os.environ['COMPUTE_BACK_END'] = 'cpu'
    os.environ['THEANO_FLAGS'] = 'device=cpu'
    print('data_root_path is set to          : {0}'.format(os.environ['DATA_ROOT_PATH']))
    print('processed_data_root_path is set to: {0}'.format(os.environ['PROCESSED_DATA_ROOT_PATH']))

elif machine_type == 'windows_virtualbox':

    os.environ['DATA_ROOT_PATH'] = '/home/pan/Documents/Data/'
    os.environ['PROCESSED_DATA_ROOT_PATH'] = '/home/pan/Documents/Data/'
    os.environ['COMPUTE_BACK_END'] = 'cpu'
    os.environ['THEANO_FLAGS'] = 'device=cpu'
    print('data_root_path is set to          : {0}'.format(os.environ['DATA_ROOT_PATH']))
    print('processed_data_root_path is set to: {0}'.format(os.environ['PROCESSED_DATA_ROOT_PATH']))

elif machine_type == 'zc2':

    os.environ['DATA_ROOT_PATH'] = '/home/ubuntu/data/FRI/processed_data/'
    os.environ['PROCESSED_DATA_ROOT_PATH'] = \
        '/home/ubuntu/data/FRI/background_bootes_images\ \(Sepand\)/'
    os.environ['COMPUTE_BACK_END'] = 'cpu'
    os.environ['THEANO_FLAGS'] = 'device=cpu'
    print('data_root_path is set to          : {0}'.format(os.environ['DATA_ROOT_PATH']))
    print('processed_data_root_path is set to: {0}'.format(os.environ['PROCESSED_DATA_ROOT_PATH']))

elif machine_type == 'iccluster':

    os.environ['DATA_ROOT_PATH'] = '/scratch/RadioAstData/'
    os.environ['PROCESSED_DATA_ROOT_PATH'] = '/scratch/RadioAstData/'
    os.environ['COMPUTE_BACK_END'] = 'gpu'
    os.environ['THEANO_FLAGS'] = 'device=gpu'
    print('data_root_path is set to          : {0}'.format(os.environ['DATA_ROOT_PATH']))
    print('processed_data_root_path is set to: {0}'.format(os.environ['PROCESSED_DATA_ROOT_PATH']))

elif machine_type == 'lcavsrv1':

    os.environ['DATA_ROOT_PATH'] = '/scratch/RadioAstData/'
    os.environ['PROCESSED_DATA_ROOT_PATH'] = '/scratch/RadioAstData/'
    os.environ['COMPUTE_BACK_END'] = 'cpu'
    os.environ['THEANO_FLAGS'] = 'device=cpu'
    print('data_root_path is set to          : {0}'.format(os.environ['DATA_ROOT_PATH']))
    print('processed_data_root_path is set to: {0}'.format(os.environ['PROCESSED_DATA_ROOT_PATH']))

elif machine_type == 'srv1':
    os.environ['COMPUTE_BACK_END'] = 'gpu'
    os.environ['THEANO_FLAGS'] = 'device=gpu'

else:

    os.environ['COMPUTE_BACK_END'] = 'cpu'
    os.environ['THEANO_FLAGS'] = 'device=cpu'
    print('DATA_ROOT_PATH and PROCESSED_DATA_ROOT_PATH not set!\n'
          'Change directories in extract_data.py and '
          'visual_example_fri_vs_clean.py manually.')
