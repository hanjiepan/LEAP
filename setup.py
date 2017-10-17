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

------------------------------------------------------------
setup various data folder that may be different from computer to computer
"""
import os

# can be: home_mac, lcav_mac, ibm_mac, mac_virtualbox,
# windows_virtualbox, zc2, iccluster, srv1, lcavsrv1
machine_type = 'lcavsrv1'

if machine_type == 'home_mac':

    os.environ['DATA_ROOT_PATH'] = '/Users/hjpan/Google\ Drive/RadioAstData/'
    os.environ['BLUEBILD_ROOT_PATH'] = '/Users/hjpan/Google\ Drive/RadioAstData/'
    os.environ['COMPUTE_BACK_END'] = 'cpu'
    os.environ['THEANO_FLAGS'] = 'device=cpu'
    print('data_root_path is set to    : {0}'.format(os.environ['DATA_ROOT_PATH']))
    print('blueBild_root_path is set to: {0}'.format(os.environ['BLUEBILD_ROOT_PATH']))

elif machine_type == 'lcav_mac':

    os.environ['DATA_ROOT_PATH'] = '/Users/pan/Google\ Drive/RadioAstData/'
    os.environ['BLUEBILD_ROOT_PATH'] = '/Users/pan/Google\ Drive/RadioAstData/'
    os.environ['COMPUTE_BACK_END'] = 'cpu'
    os.environ['THEANO_FLAGS'] = 'device=cpu'
    print('data_root_path is set to    : {0}'.format(os.environ['DATA_ROOT_PATH']))
    print('blueBild_root_path is set to: {0}'.format(os.environ['BLUEBILD_ROOT_PATH']))

elif machine_type == 'ibm_mac':

    os.environ['DATA_ROOT_PATH'] = '/Users/hpa/Google\ Drive/RadioAstData/'
    os.environ['BLUEBILD_ROOT_PATH'] = '/Users/hpa/Google\ Drive/RadioAstData/'
    os.environ['COMPUTE_BACK_END'] = 'cpu'
    os.environ['THEANO_FLAGS'] = 'device=cpu'
    print('data_root_path is set to    : {0}'.format(os.environ['DATA_ROOT_PATH']))
    print('blueBild_root_path is set to: {0}'.format(os.environ['BLUEBILD_ROOT_PATH']))

elif machine_type == 'mac_virtualbox':

    os.environ['DATA_ROOT_PATH'] = '/home/hpa/Documents/Data/'
    os.environ['BLUEBILD_ROOT_PATH'] = '/media/sf_RadioAstData/'
    os.environ['COMPUTE_BACK_END'] = 'cpu'
    os.environ['THEANO_FLAGS'] = 'device=cpu'
    print('data_root_path is set to    : {0}'.format(os.environ['DATA_ROOT_PATH']))
    print('blueBild_root_path is set to: {0}'.format(os.environ['BLUEBILD_ROOT_PATH']))

elif machine_type == 'windows_virtualbox':

    os.environ['DATA_ROOT_PATH'] = '/home/pan/Documents/Data/'
    os.environ['BLUEBILD_ROOT_PATH'] = '/home/pan/Documents/Data/'
    os.environ['COMPUTE_BACK_END'] = 'cpu'
    os.environ['THEANO_FLAGS'] = 'device=cpu'
    print('data_root_path is set to    : {0}'.format(os.environ['DATA_ROOT_PATH']))
    print('blueBild_root_path is set to: {0}'.format(os.environ['BLUEBILD_ROOT_PATH']))

elif machine_type == 'zc2':

    os.environ['DATA_ROOT_PATH'] = '/home/ubuntu/data/FRI/processed_data/'
    os.environ['BLUEBILD_ROOT_PATH'] = \
        '/home/ubuntu/data/FRI/background_bootes_images\ \(Sepand\)/'
    os.environ['COMPUTE_BACK_END'] = 'cpu'
    os.environ['THEANO_FLAGS'] = 'device=cpu'
    print('data_root_path is set to    : {0}'.format(os.environ['DATA_ROOT_PATH']))
    print('blueBild_root_path is set to: {0}'.format(os.environ['BLUEBILD_ROOT_PATH']))

elif machine_type == 'gpu3':

    os.environ['DATA_ROOT_PATH'] = '/home/hpa/data/FRI/processed_data/'
    os.environ['BLUEBILD_ROOT_PATH'] = \
        '/home/hpa/data/FRI/background_bootes_images\ \(Sepand\)/'
    os.environ['COMPUTE_BACK_END'] = 'gpu'
    os.environ['THEANO_FLAGS'] = 'device=gpu'
    print('data_root_path is set to    : {0}'.format(os.environ['DATA_ROOT_PATH']))
    print('blueBild_root_path is set to: {0}'.format(os.environ['BLUEBILD_ROOT_PATH']))

elif machine_type == 'iccluster':

    os.environ['DATA_ROOT_PATH'] = '/scratch/RadioAstData/'
    os.environ['BLUEBILD_ROOT_PATH'] = '/scratch/RadioAstData/'
    os.environ['COMPUTE_BACK_END'] = 'gpu'
    os.environ['THEANO_FLAGS'] = 'device=gpu'
    print('data_root_path is set to    : {0}'.format(os.environ['DATA_ROOT_PATH']))
    print('blueBild_root_path is set to: {0}'.format(os.environ['BLUEBILD_ROOT_PATH']))

elif machine_type == 'lcavsrv1':

    os.environ['DATA_ROOT_PATH'] = '/scratch/RadioAstData/'
    os.environ['BLUEBILD_ROOT_PATH'] = '/scratch/RadioAstData/'
    os.environ['COMPUTE_BACK_END'] = 'cpu'
    os.environ['THEANO_FLAGS'] = 'device=cpu'
    print('data_root_path is set to    : {0}'.format(os.environ['DATA_ROOT_PATH']))
    print('blueBild_root_path is set to: {0}'.format(os.environ['BLUEBILD_ROOT_PATH']))

elif machine_type == 'srv1':
    os.environ['COMPUTE_BACK_END'] = 'gpu'
    os.environ['THEANO_FLAGS'] = 'device=gpu'

else:

    os.environ['COMPUTE_BACK_END'] = 'cpu'
    os.environ['THEANO_FLAGS'] = 'device=cpu'
    print('DATA_ROOT_PATH and BLUEBILD_ROOT_PATH not set!\n'
          'Change directories in extract_data.py and '
          'visual_example_fri_vs_clean.py manually.')
