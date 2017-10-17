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
generate the phase transition plot with real LOFAR layout.
run extract_data.py first.
"""
from __future__ import division
import setup  # to set a few directories
import numpy as np
import scipy.constants
import os
import sys
import sympy
import datetime
import subprocess
from astropy import units
from astropy.coordinates import SkyCoord
import scipy.io
from functools import partial
from scipy import linalg
from alg_fri_planar_beamforming import planar_recon_2d_dirac_joint_beamforming
from build_linear_mapping_beamforming import planar_beamforming_func, \
    compile_theano_func_build_amp_mtx, compile_theano_func_build_G_mtx
from plotter import planar_plot_diracs_J2000, truncate_colormap
import matplotlib.pyplot as plt
from utils import planar_gen_visibility_beamforming, \
    planar_compute_all_baselines, planar_distance, detect_peaks

if __name__ == '__main__':
    print('Warning: should run extract_data.py first.')
    np.set_printoptions(precision=3, formatter={'float': '{: 0.3f}'.format})
    script_purpose = 'plotting'  # can be either 'testing', 'production' or 'plotting'
    # depends on the purpose, we choose a different set of parameters
    parameter_set = {}
    if script_purpose == 'testing':
        parameter_set = {
            'snr_experiment': -5,
            'dynamic_range': 0.2,  # ratio the the two Dirac amplitudes
            'coverage_rate': 0.5,
            'G_iter': 5,
            'mgain': 0.2,
            'load_data': False,
            'dpi': 300,
            'marker_scale': 0.3,
            'cmap': 'magma_r',
            'more_data': False
        }
    elif script_purpose == 'production':
        parameter_set = {
            'snr_experiment': -10,
            'dynamic_range': 0.2,
            'coverage_rate': 0.5,
            'G_iter': 20,
            'mgain': 0.1,
            'load_data': True,
            'data_file_name':
                './data/ast_src_resolve/src_param_20170331-155144.npz',
            'load_precomputed_result': False,
            'dpi': 600,
            'marker_scale': 0.3,
            'cmap': 'magma_r',
            'more_data': False
        }
    elif script_purpose == 'plotting':
        parameter_set = {
            'dynamic_range': 0.2,
            'mgain': 0.1,
            'load_data': True,
            'data_file_name':
                './data/ast_src_resolve/src_param_20170331-155144.npz',
            'load_precomputed_result': True,
            'precomputed_result_name':
                './result/ast_src_resolve/dynamic_range_plot_data_precomputed.npz',
            'dpi': 600,
            'marker_scale': 0.3,
            'cmap': truncate_colormap(plt.get_cmap('magma_r'), 0, 0.75),
            'more_data': True
        }
    else:
        RuntimeError('Unknown script purpose: {}'.format(script_purpose))

    backend = os.environ['COMPUTE_BACK_END']  # either 'cpu' or 'gpu'
    save_fig = True  # save figure or not
    fig_format = r'png'  # file type used to save the figure, e.g., pdf, png, etc.
    fig_dir = r'./result/ast_src_resolve/'  # directory to save figure
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    result_dir = './result/ast_src_resolve/'  # directory to save the result
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    param_dir = './data/ast_src_resolve/'  # directory to save source parametrisation
    if not os.path.exists(param_dir):
        os.makedirs(param_dir)

    # compile theano functions if backend == 'gpu'
    if backend == 'gpu':
        theano_build_G_func = compile_theano_func_build_G_mtx()
        theano_build_amp_func = compile_theano_func_build_amp_mtx()
    else:
        theano_build_G_func = None
        theano_build_amp_func = None

    # separation in [radian]
    separation = np.radians(1.5)  # 1.5 degree apart

    # SNR [dB] in the visibilities
    if script_purpose == 'plotting':
        snr_experiment = np.load(parameter_set['precomputed_result_name'])['snr_experiment']
    else:
        snr_experiment = parameter_set['snr_experiment']

    # various experiment settings
    light_speed = scipy.constants.speed_of_light  # speed of light

    # load LOFAR layout
    num_station = 24
    time_sampling_step = 50
    time_sampling_end = time_sampling_step * 62 + 1  # 63 STI; open interval so + 1
    num_sti = (time_sampling_end - 1) // time_sampling_step + 1
    data_file_name = \
        './data/BOOTES24_SB180-189.2ch8s_SIM_{num_sti}STI_146MHz_{num_station}Station_1Subband.npz'.format(
            num_sti=num_sti,
            num_station=num_station
        )
    # extract data
    bash_cmd = 'export PATH="/usr/bin:$PATH" && ' \
               'export PATH="$HOME/anaconda2/bin:$PATH" && ' \
               'python2 extract_data.py ' \
               '--basefile_name "BOOTES24_SB180-189.2ch8s_SIM" ' \
               '--catalog_file "skycatalog.npz" ' \
               '--num_channel 1 ' \
               '--time_sampling_step {time_sampling_step} ' \
               '--time_sampling_end {time_sampling_end} ' \
               '--freq_channel_min 0 ' \
               '--freq_channel_step 1 ' \
               '--number_of_stations {num_station} ' \
               '--FoV 5'.format(
        time_sampling_step=time_sampling_step,
        time_sampling_end=time_sampling_end,
        num_station=num_station
    )
    if subprocess.call(bash_cmd, shell=True):
        raise RuntimeError('Could not extract data!')

    data_root_path = os.environ['DATA_ROOT_PATH']
    basefile_name = 'BOOTES24_SB180-189.2ch8s_SIM'
    ms_file_name = data_root_path + basefile_name + '.ms'
    sub_table_file_name = '{basefile_name}_every{time_sampling_step}th.ms'.format(
        basefile_name=basefile_name,
        time_sampling_step=time_sampling_step
    )
    sub_table_full_name = data_root_path + sub_table_file_name

    lofar_data = np.load(data_file_name)

    freq_subbands_hz = lofar_data['freq_subbands_hz']

    '''the array coordinate is arranged as a 4D matrix, where
            dimension 0: antenna index within one station
            dimension 1: station index
            dimension 2: STI index
            dimension 3: (of size 3) corresponds to x, y, and z coordinates'''
    array_coordinate = lofar_data['array_coordinate']

    FoV_degree = lofar_data['FoV']  # field of view

    # number of antennas, stations, short time intervals (STI), xyz
    assert array_coordinate.shape[-1] == 3
    num_antenna, num_station, num_sti = array_coordinate.shape[:-1]
    num_subband = np.asarray(freq_subbands_hz).size

    # convert to usable data
    r_antenna_x = array_coordinate[:, :, :num_sti, 0]
    r_antenna_y = array_coordinate[:, :, :num_sti, 1]
    r_antenna_z = array_coordinate[:, :, :num_sti, 2]

    # number of point sources
    K = 2
    K_est = K  # estimated number of point sources
    plane_norm_vec = (0, 0, 1)
    # if we put one source exactly at the center, then CLEAN runs into trouble

    # reconstruct point sources
    max_ini = 15  # maximum number of random initializations
    tau_x = tau_y = float(np.radians(min(max(0.5, np.degrees(separation) * 3), FoV_degree)))

    norm_factor = np.reshape(light_speed / (2 * np.pi * freq_subbands_hz),
                             (1, 1, 1, -1), order='F')
    # normalised antenna coordinates
    p_x_normalised = np.reshape(
        r_antenna_x, (-1, num_station, num_sti, num_subband), order='F') / norm_factor
    p_y_normalised = np.reshape(
        r_antenna_y, (-1, num_station, num_sti, num_subband), order='F') / norm_factor
    p_z_normalised = np.reshape(
        r_antenna_z, (-1, num_station, num_sti, num_subband), order='F') / norm_factor

    if parameter_set['load_data']:
        dirac_param = np.load(parameter_set['data_file_name'])
        x_ks = dirac_param['x_ks']
        y_ks = dirac_param['y_ks']
        alpha_ks = dirac_param['alpha_ks']
    else:
        # generate Dirac parameters
        alpha_ks = np.tile(np.array([1, 1 * parameter_set['dynamic_range']])[:, np.newaxis],
                           (1, num_subband))
        x1 = y1 = 0  # first source location

        rnd_angle = np.random.rand() * np.pi
        x2 = x1 + separation * np.cos(rnd_angle)
        y2 = y1 + separation * np.sin(rnd_angle)

        x_ks = np.array([x1, x2])
        y_ks = np.array([y1, y2])
        # if we put one source exactly at the center, then CLEAN runs into trouble
        x_ks -= np.cos(rnd_angle) * separation / 5.
        y_ks -= np.sin(rnd_angle) * separation / 5.

    print('first source locations (x, y): ({0:.2f}, {1:.2f})[arcmin]'.format(
        np.degrees(x_ks[0]) * 60, np.degrees(y_ks[0]) * 60))
    print('second source locations (x, y): ({0:.2f}, {1:.2f})[arcmin]'.format(
        np.degrees(x_ks[1]) * 60, np.degrees(y_ks[1]) * 60))

    # generate noiseless visibilities based on the antenna layout and subband frequency
    partial_beamforming_func = partial(planar_beamforming_func,
                                       strategy='matched',
                                       x0=0, y0=0)
    '''the visibility measurements are arranged as a 3D matrix, where
            dimension 0: cross-correlation index
            dimension 1: STI index
            dimension 2: subband index'''
    visi_noiseless, visi_noisy = \
        planar_gen_visibility_beamforming(
            alpha_ks, x_ks, y_ks,
            p_x_normalised, p_y_normalised,
            partial_beamforming_func,
            num_station, num_subband, num_sti,
            snr_data=snr_experiment
        )
    num_visibility = visi_noiseless.size

    # save data in order to run CLEAN on the same data
    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dirac_data_file_name = param_dir + 'src_param_' + time_stamp + '.npz'
    print('Saving Dirac parameters and visibilities in {0}'.format(dirac_data_file_name))
    np.savez(dirac_data_file_name,
             x_ks=x_ks, y_ks=y_ks, alpha_ks=alpha_ks,
             visi=visi_noisy, visi_noiseless=visi_noiseless)

    if not parameter_set['load_precomputed_result']:
        '''define the period of the (periodic)-sinc interpolation:
            the coverage_rate percentile smallest frequencies are contained in one period.
            if coverage_rate = 1, then all frequencies are completly contained.
            '''
        coverage_rate = parameter_set['coverage_rate']

        # compute all the baselines
        all_baselines_x, all_baselines_y = \
            planar_compute_all_baselines(p_x_normalised, p_y_normalised, num_antenna,
                                         num_station, num_subband, num_sti)

        # determine periodic sinc interpolation parameters
        kth_idx = int(all_baselines_x.size * coverage_rate) - 1
        M = int(np.ceil(np.partition(np.abs(all_baselines_x).flatten(),
                                     kth_idx)[kth_idx] / np.pi))
        N = int(np.ceil(np.partition(np.abs(all_baselines_y).flatten(),
                                     kth_idx)[kth_idx] / np.pi))
        M_tau_x = np.ceil(M * tau_x / 2) * 2 + 1  # M * tau_x is an odd number
        N_tau_y = np.ceil(N * tau_y / 2) * 2 + 1  # N * tau_y is an odd number
        tau_inter_x = sympy.Rational(M_tau_x, M)  # interpolation step size: 2 pi / tau_inter
        tau_inter_y = sympy.Rational(N_tau_y, N)

        print(('M = {0:.0f}, N = {1:.0f},\n'
               'tau_x = {2:.2e}, tau_y = {3:.2e},\n'
               'tau_inter_x = {4:.2e}, tau_inter_y = {5:.2e},\n'
               'M * tau_inter_x = {6:.0f}, '
               'N * tau_inter_y = {7:.0f}').format(M, N, tau_x, tau_y,
                                                   float(tau_inter_x.evalf()),
                                                   float(tau_inter_y.evalf()),
                                                   float((M * tau_inter_x).evalf()),
                                                   float((N * tau_inter_y).evalf())))

        xk_recon, yk_recon, alpha_k_recon = \
            planar_recon_2d_dirac_joint_beamforming(
                visi_noisy, r_antenna_x, r_antenna_y,
                2 * np.pi * freq_subbands_hz, light_speed, K=K_est,
                tau_x=tau_x, tau_y=tau_y, M=M, N=N, tau_inter_x=tau_inter_x,
                tau_inter_y=tau_inter_y, max_ini=max_ini, num_rotation=1,
                G_iter=parameter_set['G_iter'],
                plane_norm_vec=plane_norm_vec, verbose=True,
                backend=backend, theano_build_G_func=theano_build_G_func,
                theano_build_amp_func=theano_build_amp_func
            )
    else:
        precomputed_result = np.load(parameter_set['precomputed_result_name'])
        xk_recon = precomputed_result['xk_recon']
        yk_recon = precomputed_result['yk_recon']
        alpha_k_recon = precomputed_result['alpha_k_recon']

    # compute partial reconstruction error
    dist_recon, idx_sort = planar_distance(x_ks, y_ks, xk_recon, yk_recon)

    xk_recon_sorted = xk_recon[idx_sort[:, 1]]
    yk_recon_sorted = yk_recon[idx_sort[:, 1]]
    alpha_k_recon_sorted = alpha_k_recon[idx_sort[:, 1]]

    x_ks_sorted = x_ks[idx_sort[:, 0]]
    y_ks_sorted = y_ks[idx_sort[:, 0]]
    alpha_ks_sorted = alpha_ks[idx_sort[:, 0], :]

    # in degree, minute, and second representation
    dist_recon_dms = SkyCoord(
        ra=0, dec=dist_recon, unit=units.radian
    ).to_string('dms').split(' ')[1]

    # print reconstruction results
    print('Reconstructed locations (in degrees) and amplitudes:')
    print('Ground truth horizontal locations   : {0}\n'.format(np.degrees(x_ks_sorted)))
    print('Reconstructed horizontal locations   : {0}\n'.format(np.degrees(xk_recon_sorted)))

    print('Ground truth vertical locations     : {0}\n'.format(np.degrees(y_ks_sorted)))
    print('Reconstructed vertical locations     : {0}\n'.format(np.degrees(yk_recon_sorted)))

    print('Ground truth amplitudes : \n{0}\n'.format(np.real(alpha_ks_sorted)))
    print('Reconstructed amplitudes : \n{0}\n'.format(np.real(alpha_k_recon_sorted)))

    print('Reconstructed locations error: {0}'.format(dist_recon_dms))

    # save plotting data
    np.savez(result_dir + 'dynamic_range_plot_data.npz',
             x_ks=x_ks, y_ks=y_ks, alpha_ks=alpha_ks,
             xk_recon=xk_recon, yk_recon=yk_recon, alpha_k_recon=alpha_k_recon,
             dist_recon=dist_recon, separation=separation, snr_experiment=snr_experiment)

    # run wsclean with the same visibilities
    if sys.version_info[0] > 2:
        sys.exit('Sorry casacore only runs on Python 2.')
    else:
        from casacore import tables as casa_tables

        antenna1_lst = \
            np.sort(casa_tables.taql('select distinct ANTENNA1 from {ms_file_name}'.format(
                ms_file_name=ms_file_name
            )).getcol('ANTENNA1'))
        antenna2_lst = \
            np.sort(casa_tables.taql('select distinct ANTENNA2 from {ms_file_name}'.format(
                ms_file_name=ms_file_name
            )).getcol('ANTENNA2'))
        assert antenna1_lst.size == antenna2_lst.size
        num_station = min(num_station, antenna1_lst.size)

        antenna1_limit = antenna1_lst[num_station - 1]
        antenna2_limit = antenna2_lst[num_station - 1]

        taql_cmd_str = 'select from {ms_file_name} where TIME in ' \
                       '(select distinct TIME from {ms_file_name} limit {time_range})' \
                       'and ANTENNA1<={antenna1_limit} ' \
                       'and ANTENNA2<={antenna2_limit} ' \
                       'giving {sub_table_name}'.format(
            ms_file_name=ms_file_name,
            time_range='0:{0}:{1}'.format(time_sampling_end,
                                          time_sampling_step),
            antenna1_limit=antenna1_limit,
            antenna2_limit=antenna2_limit,
            sub_table_name=sub_table_full_name
        )
        casa_tables.taql(taql_cmd_str)

        # get pointing direction
        taql_cmd_str = 'select REFERENCE_DIR from {ms_file_name}::FIELD'.format(
            ms_file_name=ms_file_name
        )
        direction = casa_tables.taql(taql_cmd_str).getcol('REFERENCE_DIR').squeeze()
        sky_ra, sky_dec = direction

    bash_cmd = 'export PATH="/usr/bin:$PATH" && ' \
               'export PATH="$HOME/anaconda2/bin:$PATH" && ' \
               'python2 call_wsclean_simulated.py ' \
               '--visi_file_name {visi_file_name} ' \
               '--msfile_in {msfile_in} ' \
               '--num_station {num_station} ' \
               '--num_sti {num_sti} ' \
               '--intermediate_size {intermediate_size} ' \
               '--output_img_size {output_img_size} ' \
               '--FoV {FoV} ' \
               '--output_name_prefix {output_name_prefix} ' \
               '--freq_channel_min {freq_channel_min} ' \
               '--freq_channel_max {freq_channel_max} ' \
               '--max_iter {max_iter} ' \
               '--mgain {mgain} ' \
               '--auto_threshold {auto_threshold} ' \
               '--imag_format {imag_format} ' \
               '--dpi {dpi}'.format(
        visi_file_name=dirac_data_file_name,
        msfile_in=sub_table_full_name,
        num_station=num_station,
        num_sti=num_sti,
        intermediate_size=1024,
        output_img_size=606,
        FoV=min(max(1, np.degrees(separation) * 7), FoV_degree),
        output_name_prefix=data_root_path + 'highres',
        freq_channel_min=0,
        freq_channel_max=0 + num_subband,
        max_iter=40000,
        mgain=parameter_set['mgain'],
        auto_threshold=3,
        imag_format='png',
        dpi=parameter_set['dpi']
    )
    if subprocess.call(bash_cmd, shell=True):
        raise RuntimeError('wsCLEAN could not run!')

    # load CLEAN results
    clean_data = np.load('./data/' + sub_table_file_name[:-3] + '_modi-CLEAN_data.npz')
    # or the compressive sensing results (uncomment to use CS result instead of CLEAN result)
    # clean_data = np.load('./data/' + sub_table_file_name[:-3] + '_modi_cs-CLEAN_data.npz')
    img_clean = clean_data['img_clean']
    img_dirty = clean_data['img_dirty']
    src_model_clean = clean_data['src_model']
    x_plt_CLEAN = clean_data['x_plt_CLEAN_rad']
    y_plt_CLEAN = clean_data['y_plt_CLEAN_rad']

    file_name = fig_dir + \
                'visual_comparison_2src_dynamic_range{dynamic_range}'.format(
                    dynamic_range=parameter_set['dynamic_range']
                )
    # back ground image: dirty image
    planar_plot_diracs_J2000(
        x_plt_grid=x_plt_CLEAN, y_plt_grid=y_plt_CLEAN,
        RA_focus_rad=sky_ra, DEC_focus_rad=sky_dec,
        background_img=img_dirty,
        cmap=parameter_set['cmap'],
        marker_scale=parameter_set['marker_scale'],
        marker_alpha=0.7, save_fig=save_fig,
        file_name=file_name + '_bg_img_dirty',
        label_ref_sol='ground truth', label_recon='reconstruction',
        file_format=fig_format, dpi=parameter_set['dpi'], close_fig=False,
        title_str='dirty image'
    )
    # back ground image: CLEAN image
    planar_plot_diracs_J2000(
        x_plt_grid=x_plt_CLEAN, y_plt_grid=y_plt_CLEAN,
        RA_focus_rad=sky_ra, DEC_focus_rad=sky_dec,
        background_img=img_clean,
        cmap=parameter_set['cmap'],
        marker_scale=parameter_set['marker_scale'],
        marker_alpha=0.7, save_fig=save_fig,
        file_name=file_name + '_bg_img_clean',
        label_ref_sol='ground truth', label_recon='reconstruction',
        file_format=fig_format, dpi=parameter_set['dpi'], close_fig=False,
        title_str='CLEAN image'
    )
    # back ground image: CLEAN image
    planar_plot_diracs_J2000(
        x_plt_grid=x_plt_CLEAN, y_plt_grid=y_plt_CLEAN,
        RA_focus_rad=sky_ra, DEC_focus_rad=sky_dec,
        x_ref=x_ks, y_ref=y_ks, amplitude_ref=alpha_ks,
        background_img=img_clean,
        cmap=parameter_set['cmap'],
        marker_scale=parameter_set['marker_scale'],
        marker_alpha=0.7, save_fig=save_fig,
        file_name=file_name + '_bg_img_clean_with_gt',
        label_ref_sol='ground truth', label_recon='reconstruction',
        file_format=fig_format, dpi=parameter_set['dpi'], close_fig=False,
        title_str='CLEAN image'
    )
    planar_plot_diracs_J2000(
        x_plt_grid=x_plt_CLEAN, y_plt_grid=y_plt_CLEAN,
        RA_focus_rad=sky_ra, DEC_focus_rad=sky_dec,
        x_ref=x_ks, y_ref=y_ks, amplitude_ref=alpha_ks,
        x_recon=xk_recon, y_recon=yk_recon, amplitude_recon=alpha_k_recon,
        background_img=img_clean,
        cmap=parameter_set['cmap'],
        marker_scale=parameter_set['marker_scale'],
        marker_alpha=0.7, save_fig=save_fig,
        file_name=file_name + '_bg_img_clean_with_gt_recon',
        label_ref_sol='ground truth', label_recon='reconstruction',
        file_format=fig_format, dpi=parameter_set['dpi'], close_fig=False
    )

    # now use more stations and more integration time for CLEAN
    if parameter_set['more_data']:
        # load LOFAR layout
        num_station = 24  # <= one station is missing if use 57 -> not all correlations available
        time_sampling_step = 1
        time_sampling_end = 3150
        num_sti = (time_sampling_end - 1) // time_sampling_step + 1
        data_file_name = \
            './data/BOOTES24_SB180-189.2ch8s_SIM_{num_sti}STI_146MHz_{num_station}Station_1Subband.npz'.format(
                num_sti=num_sti,
                num_station=num_station
            )
        # extract data
        bash_cmd = 'export PATH="/usr/bin:$PATH" && ' \
                   'export PATH="$HOME/anaconda2/bin:$PATH" && ' \
                   'python2 extract_data.py ' \
                   '--basefile_name "BOOTES24_SB180-189.2ch8s_SIM" ' \
                   '--catalog_file "skycatalog.npz" ' \
                   '--num_channel 1 ' \
                   '--time_sampling_step {time_sampling_step} ' \
                   '--time_sampling_end {time_sampling_end} ' \
                   '--freq_channel_min 0 ' \
                   '--freq_channel_step 1 ' \
                   '--number_of_stations {num_station} ' \
                   '--FoV 5'.format(
            time_sampling_step=time_sampling_step,
            time_sampling_end=time_sampling_end,
            num_station=num_station
        )
        if subprocess.call(bash_cmd, shell=True):
            raise RuntimeError('Could not extract data!')

        data_root_path = os.environ['DATA_ROOT_PATH']
        basefile_name = 'BOOTES24_SB180-189.2ch8s_SIM'
        sub_table_file_name = '{basefile_name}_every{time_sampling_step}th.ms'.format(
            basefile_name=basefile_name,
            time_sampling_step=time_sampling_step
        )
        sub_table_full_name = data_root_path + sub_table_file_name

        lofar_data = np.load(data_file_name)

        freq_subbands_hz = lofar_data['freq_subbands_hz']

        '''the array coordinate is arranged as a 4D matrix, where
                dimension 0: antenna index within one station
                dimension 1: station index
                dimension 2: STI index
                dimension 3: (of size 3) corresponds to x, y, and z coordinates'''
        array_coordinate = lofar_data['array_coordinate']

        FoV_degree = lofar_data['FoV']  # field of view

        # number of antennas, stations, short time intervals (STI), xyz
        assert array_coordinate.shape[-1] == 3
        num_antenna, num_station, num_sti = array_coordinate.shape[:-1]
        num_subband = np.asarray(freq_subbands_hz).size

        # convert to usable data
        r_antenna_x = array_coordinate[:, :, :num_sti, 0]
        r_antenna_y = array_coordinate[:, :, :num_sti, 1]
        r_antenna_z = array_coordinate[:, :, :num_sti, 2]

        norm_factor = np.reshape(light_speed / (2 * np.pi * freq_subbands_hz),
                                 (1, 1, 1, -1), order='F')
        # normalised antenna coordinates
        p_x_normalised = np.reshape(
            r_antenna_x, (-1, num_station, num_sti, num_subband), order='F') / norm_factor
        p_y_normalised = np.reshape(
            r_antenna_y, (-1, num_station, num_sti, num_subband), order='F') / norm_factor
        p_z_normalised = np.reshape(
            r_antenna_z, (-1, num_station, num_sti, num_subband), order='F') / norm_factor

        # generate noiseless visibilities based on the antenna layout and subband frequency
        partial_beamforming_func = partial(planar_beamforming_func,
                                           strategy='matched',
                                           x0=0, y0=0)
        '''the visibility measurements are arranged as a 3D matrix, where
                dimension 0: cross-correlation index
                dimension 1: STI index
                dimension 2: subband index'''
        visi_noiseless, visi_noisy = \
            planar_gen_visibility_beamforming(
                alpha_ks, x_ks, y_ks,
                p_x_normalised, p_y_normalised,
                partial_beamforming_func,
                num_station, num_subband, num_sti,
                snr_data=snr_experiment
            )
        num_visibility = visi_noiseless.size

        # save data in order to run CLEAN on the same data
        time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        dirac_data_file_name = param_dir + 'src_param_' + time_stamp + '.npz'
        print('Saving Dirac parameters and visibilities in {0}'.format(dirac_data_file_name))
        np.savez(dirac_data_file_name,
                 x_ks=x_ks, y_ks=y_ks, alpha_ks=alpha_ks,
                 visi=visi_noisy, visi_noiseless=visi_noiseless)

        # run wsclean with the same visibilities
        if sys.version_info[0] > 2:
            sys.exit('Sorry casacore only runs on Python 2.')
        else:
            from casacore import tables as casa_tables

            antenna1_lst = \
                np.sort(casa_tables.taql('select distinct ANTENNA1 from {ms_file_name}'.format(
                    ms_file_name=ms_file_name
                )).getcol('ANTENNA1'))
            antenna2_lst = \
                np.sort(casa_tables.taql('select distinct ANTENNA2 from {ms_file_name}'.format(
                    ms_file_name=ms_file_name
                )).getcol('ANTENNA2'))
            assert antenna1_lst.size == antenna2_lst.size
            num_station = min(num_station, antenna1_lst.size)

            antenna1_limit = antenna1_lst[num_station - 1]
            antenna2_limit = antenna2_lst[num_station - 1]

            taql_cmd_str = 'select from {ms_file_name} where TIME in ' \
                           '(select distinct TIME from {ms_file_name} limit {time_range})' \
                           'and ANTENNA1<={antenna1_limit} ' \
                           'and ANTENNA2<={antenna2_limit} ' \
                           'giving {sub_table_name}'.format(
                ms_file_name=ms_file_name,
                time_range='0:{0}:{1}'.format(time_sampling_end,
                                              time_sampling_step),
                antenna1_limit=antenna1_limit,
                antenna2_limit=antenna2_limit,
                sub_table_name=sub_table_full_name
            )
            casa_tables.taql(taql_cmd_str)

            # get pointing direction
            taql_cmd_str = 'select REFERENCE_DIR from {ms_file_name}::FIELD'.format(
                ms_file_name=ms_file_name
            )
            direction = casa_tables.taql(taql_cmd_str).getcol('REFERENCE_DIR').squeeze()
            sky_ra, sky_dec = direction

        bash_cmd = 'export PATH="/usr/bin:$PATH" && ' \
                   'export PATH="$HOME/anaconda2/bin:$PATH" && ' \
                   'python2 call_wsclean_simulated.py ' \
                   '--visi_file_name {visi_file_name} ' \
                   '--msfile_in {msfile_in} ' \
                   '--num_station {num_station} ' \
                   '--num_sti {num_sti} ' \
                   '--intermediate_size {intermediate_size} ' \
                   '--output_img_size {output_img_size} ' \
                   '--FoV {FoV} ' \
                   '--output_name_prefix {output_name_prefix} ' \
                   '--freq_channel_min {freq_channel_min} ' \
                   '--freq_channel_max {freq_channel_max} ' \
                   '--max_iter {max_iter} ' \
                   '--mgain {mgain} ' \
                   '--auto_threshold {auto_threshold} ' \
                   '--imag_format {imag_format} ' \
                   '--dpi {dpi}'.format(
            visi_file_name=dirac_data_file_name,
            msfile_in=sub_table_full_name,
            num_station=num_station,
            num_sti=num_sti,
            intermediate_size=1024,
            output_img_size=606,
            FoV=min(max(1, np.degrees(separation) * 7), FoV_degree),
            output_name_prefix=data_root_path + 'highres',
            freq_channel_min=0,
            freq_channel_max=0 + num_subband,
            max_iter=40000,
            mgain=parameter_set['mgain'],
            auto_threshold=3,
            imag_format='png',
            dpi=parameter_set['dpi']
        )
        if subprocess.call(bash_cmd, shell=True):
            raise RuntimeError('wsCLEAN could not run!')

        # load CLEAN results
        clean_data = np.load('./data/' + sub_table_file_name[:-3] + '_modi-CLEAN_data.npz')
        img_clean = clean_data['img_clean']
        img_dirty = clean_data['img_dirty']
        src_model_clean = clean_data['src_model']
        x_plt_CLEAN = clean_data['x_plt_CLEAN_rad']
        y_plt_CLEAN = clean_data['y_plt_CLEAN_rad']

        file_name = fig_dir + \
                    'visual_comparison_2src_dynamic_range{dynamic_range}'.format(
                        dynamic_range=parameter_set['dynamic_range']
                    )
        # back ground image: dirty image
        planar_plot_diracs_J2000(
            x_plt_grid=x_plt_CLEAN, y_plt_grid=y_plt_CLEAN,
            RA_focus_rad=sky_ra, DEC_focus_rad=sky_dec,
            background_img=img_dirty,
            cmap=parameter_set['cmap'],
            marker_scale=parameter_set['marker_scale'] * 0.5,
            marker_alpha=0.5, save_fig=save_fig,
            file_name=file_name + '_bg_img_dirty_more_data',
            label_ref_sol='ground truth', label_recon='reconstruction',
            file_format=fig_format, dpi=parameter_set['dpi'], close_fig=False,
            title_str='dirty image ({num_station} stations, '
                      '{num_sti} integration times)'.format(
                num_station=num_station,
                num_sti=num_sti
            )
        )
        planar_plot_diracs_J2000(
            x_plt_grid=x_plt_CLEAN, y_plt_grid=y_plt_CLEAN,
            RA_focus_rad=sky_ra, DEC_focus_rad=sky_dec,
            x_ref=x_ks, y_ref=y_ks, amplitude_ref=alpha_ks,
            background_img=img_dirty,
            cmap=parameter_set['cmap'],
            marker_scale=parameter_set['marker_scale'] * 0.5,
            marker_alpha=0.5, save_fig=save_fig,
            file_name=file_name + '_bg_img_dirty_more_data_with_gt',
            label_ref_sol='ground truth', label_recon='reconstruction',
            file_format=fig_format, dpi=parameter_set['dpi'], close_fig=False,
            title_str='dirty image ({num_station} stations, '
                      '{num_sti} integration times)'.format(
                num_station=num_station,
                num_sti=num_sti
            )
        )
        # back ground image: CLEAN image
        planar_plot_diracs_J2000(
            x_plt_grid=x_plt_CLEAN, y_plt_grid=y_plt_CLEAN,
            RA_focus_rad=sky_ra, DEC_focus_rad=sky_dec,
            background_img=img_clean,
            cmap=parameter_set['cmap'],
            marker_scale=parameter_set['marker_scale'] * 0.5,
            marker_alpha=0.5, save_fig=save_fig,
            file_name=file_name + '_bg_img_clean_more_data',
            label_ref_sol='ground truth', label_recon='reconstruction',
            file_format=fig_format, dpi=parameter_set['dpi'], close_fig=False,
            title_str='CLEAN image ({num_station} stations, '
                      '{num_sti} integration times)'.format(
                num_station=num_station,
                num_sti=num_sti
            )
        )
        planar_plot_diracs_J2000(
            x_plt_grid=x_plt_CLEAN, y_plt_grid=y_plt_CLEAN,
            RA_focus_rad=sky_ra, DEC_focus_rad=sky_dec,
            x_ref=x_ks, y_ref=y_ks, amplitude_ref=alpha_ks,
            background_img=img_clean,
            cmap=parameter_set['cmap'],
            marker_scale=parameter_set['marker_scale'] * 0.5,
            marker_alpha=0.5, save_fig=save_fig,
            file_name=file_name + '_bg_img_clean_more_data_with_gt',
            label_ref_sol='ground truth', label_recon='reconstruction',
            file_format=fig_format, dpi=parameter_set['dpi'], close_fig=False,
            title_str='CLEAN image ({num_station} stations, '
                      '{num_sti} integration times)'.format(
                num_station=num_station,
                num_sti=num_sti
            )
        )

    plt.show()

    # reset numpy print option
    np.set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan',
                        precision=8, suppress=False, threshold=1000, formatter=None)
