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
Illustration of the model order selection process.
Also we try to address the false detection issue.
"""
from __future__ import division
import setup  # to set a few directories
import numpy as np
import scipy.constants
import os
import sympy
import datetime
import subprocess
from functools import partial
from scipy import linalg
from alg_fri_planar_beamforming import planar_recon_2d_dirac_joint_beamforming
from build_linear_mapping_beamforming import planar_beamforming_func, \
    compile_theano_func_build_amp_mtx, compile_theano_func_build_G_mtx
from plotter import planar_plot_diracs_J2000
from utils import planar_gen_visibility_beamforming, \
    planar_compute_all_baselines, planar_distance
import matplotlib

if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt


if __name__ == '__main__':
    np.set_printoptions(precision=3, formatter={'float': '{: 0.3f}'.format})
    script_purpose = 'plotting'  # can be either 'testing', 'production', or 'plotting'
    parameter_set = {}
    if script_purpose == 'testing':
        parameter_set = {
            'snr_experiment': 0,
            'coverage_rate': 1,
            'G_iter': 2,
            'mgain': 0.1,
            'load_data': False,
            'load_precomputed_result': False,
            'intermidiate_file_name':
                './result/intermediate_result_model_order.npz',
            'dpi': 300,
            'marker_scale': 0.6,
            'cmap': 'magma_r',
            'more_data': False
        }
    elif script_purpose == 'production':
        parameter_set = {
            'snr_experiment': 0,
            'coverage_rate': 1,
            'G_iter': 20,
            'mgain': 0.1,
            'load_data': True,
            'data_file_name':
                './data/ast_src_resolve/src_param_20170701-163710.npz',
            'load_precomputed_result': False,
            'intermidiate_file_name':
                './result/intermediate_result_model_order.npz',
            'dpi': 600,
            'marker_scale': 0.6,
            'cmap': 'magma_r',
            'more_data': False
        }
    elif script_purpose == 'plotting':
        parameter_set = {
            'load_data': True,
            'load_precomputed_result': True,
            'data_file_name':
                './result/ast_src_resolve/model_order_plot_data_precomputed.npz',
            'mgain': 0.1,
            'precomputed_result_name':
                './result/ast_src_resolve/model_order_plot_data_precomputed.npz',
            'dpi': 600,
            'marker_scale': 0.6,
            'cmap': 'magma_r',
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
    if script_purpose != 'plotting':
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
    sky_ra = lofar_data['RA_rad'].squeeze()
    sky_dec = lofar_data['DEC_rad'].squeeze()

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
    K = 3
    K_est_lst = np.arange(start=1, stop=6, step=1)
    plane_norm_vec = (0, 0, 1)

    # generate Dirac parameters
    if parameter_set['load_data']:
        dirac_param = np.load(parameter_set['data_file_name'])
        x_ks = dirac_param['x_ks']
        y_ks = dirac_param['y_ks']
        alpha_ks = dirac_param['alpha_ks']
    else:
        # generate Diracs that are located within a field of view 30 arcmin
        fov_dirac = np.radians(30 / 60)
        x_ks = fov_dirac * np.random.rand(K) - 0.5 * fov_dirac
        y_ks = fov_dirac * np.random.rand(K) - 0.5 * fov_dirac
        alpha_ks = np.abs(np.random.randn(K, 1))

    print('Source locations (x, y) [arcmin]: ({0})'.format(
        np.degrees(np.column_stack((x_ks, y_ks))) * 60)
    )
    print('Source intensities: {0}'.format(alpha_ks))

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
        visi_noiseless = dirac_param['visi_noiseless']
        visi_noisy = dirac_param['visi']
        img_clean = dirac_param['img_clean']
        x_plt = dirac_param['x_plt']
        y_plt = dirac_param['y_plt']
    else:
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
        # save data in order to run CLEAN on the same data
        time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        dirac_data_file_name = param_dir + 'src_param_' + time_stamp + '.npz'
        print('Saving Dirac parameters and visibilities in {0}'.format(dirac_data_file_name))
        dirac_data_dict = {
            'x_ks': x_ks,
            'y_ks': y_ks,
            'alpha_ks': alpha_ks,
            'visi': visi_noisy,
            'visi_noiseless': visi_noiseless
        }
        np.savez(dirac_data_file_name, **dirac_data_dict)

        # generate CLEAN image from the same measurements
        max_width_deg = np.degrees(np.max(np.concatenate((np.abs(x_ks), np.abs(y_ks))))).squeeze()
        FoV_clean = min(max(1, max_width_deg * 7), FoV_degree)
        clean_img_sz = int(max(FoV_clean // (0.5 * max_width_deg), 1024))
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
            intermediate_size=int(1.5 * clean_img_sz),
            output_img_size=clean_img_sz,
            FoV=FoV_clean,
            output_name_prefix=data_root_path + 'highres',
            freq_channel_min=0,
            freq_channel_max=0 + num_subband,
            max_iter=40000,
            mgain=parameter_set['mgain'],
            auto_threshold=3,
            imag_format='png',
            dpi=600
        )
        if subprocess.call(bash_cmd, shell=True):
            raise RuntimeError('wsCLEAN could not run!')

        # load CLEAN results
        clean_data = np.load('./data/' + sub_table_file_name[:-3] + '_modi-CLEAN_data.npz')
        img_clean = clean_data['img_clean']
        x_plt = clean_data['x_plt_CLEAN_rad']
        y_plt = clean_data['y_plt_CLEAN_rad']
        dirac_data_dict['img_clean'] = img_clean
        dirac_data_dict['x_plt'] = x_plt
        dirac_data_dict['y_plt'] = y_plt
        # save the CLEAN image to the Dirac data file
        np.savez(dirac_data_file_name, **dirac_data_dict)

    # compute the noise energy (to be compared with the fitting error)
    # ell-2 norm
    noise_level = linalg.norm(visi_noiseless.flatten() - visi_noisy.flatten())

    # reconstruct point sources
    max_ini = 15  # maximum number of random initializations
    tau_x = tau_y = np.radians(
        min(max(0.5, np.degrees(np.max(np.concatenate((
                        np.abs(x_ks), np.abs(y_ks)
                    )))).squeeze() * 3),
            FoV_degree)).squeeze()

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
        M_tau_x = int(np.ceil(M * tau_x / 2)) * 2 + 1  # M * tau_x is an odd number
        N_tau_y = int(np.ceil(N * tau_y / 2)) * 2 + 1  # N * tau_y is an odd number
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

        result_summary = np.empty(len(K_est_lst), dtype=object)
        for loop_count, K_est in enumerate(K_est_lst):
            xk_recon, yk_recon, alpha_k_recon, fitting_error = \
                planar_recon_2d_dirac_joint_beamforming(
                    visi_noisy, r_antenna_x, r_antenna_y,
                    2 * np.pi * freq_subbands_hz, light_speed, K=K_est,
                    tau_x=tau_x, tau_y=tau_y, M=M, N=N, tau_inter_x=tau_inter_x,
                    tau_inter_y=tau_inter_y, max_ini=max_ini, num_rotation=1,
                    G_iter=parameter_set['G_iter'],
                    plane_norm_vec=plane_norm_vec, verbose=True,
                    backend=backend, theano_build_G_func=theano_build_G_func,
                    theano_build_amp_func=theano_build_amp_func,
                    return_error=True)
            # compute reconstructed source distance with the ground truth
            dist_recon, idx_sort = planar_distance(x_ks, y_ks, xk_recon, yk_recon)
            # store result in a dictionary
            result_loop = {
                'K_est': K_est,
                'xk_recon': xk_recon,
                'yk_recon': yk_recon,
                'alpha_k_recon': alpha_k_recon,
                'dist_recon': dist_recon,
                'fitting_error': fitting_error
            }
            result_summary[loop_count] = result_loop

        # save results
        np.savez(parameter_set['intermidiate_file_name'],
                 result_summary=result_summary,
                 x_ks=x_ks, y_ks=y_ks, alpha_ks=alpha_ks,
                 visi=visi_noisy, visi_noiseless=visi_noiseless,
                 snr_experiment=snr_experiment, noise_level=noise_level,
                 img_clean=img_clean, x_plt=x_plt, y_plt=y_plt)
    else:
        precomputed_result = np.load(parameter_set['precomputed_result_name'], encoding='latin1')
        result_summary = precomputed_result['result_summary']

    # plot the fitting errors with different K_est
    file_name = fig_dir + 'model_order_sel'
    fitting_error_all = []
    for result_loop in result_summary:
        xk_recon = result_loop['xk_recon']
        yk_recon = result_loop['yk_recon']
        alpha_k_recon = result_loop['alpha_k_recon']
        K_est = result_loop['K_est']
        fitting_error_all.append(result_loop['fitting_error'])
        planar_plot_diracs_J2000(
            x_plt_grid=x_plt, y_plt_grid=y_plt,
            RA_focus_rad=sky_ra, DEC_focus_rad=sky_dec,
            x_ref=x_ks, y_ref=y_ks, amplitude_ref=alpha_ks,
            x_recon=xk_recon, y_recon=yk_recon, amplitude_recon=alpha_k_recon,
            cmap=parameter_set['cmap'],
            background_img=img_clean,
            marker_scale=parameter_set['marker_scale'], save_fig=save_fig,
            file_name=file_name + '_Kest_{}'.format(K_est),
            label_ref_sol='ground truth', label_recon='reconstruction',
            file_format=fig_format, dpi=parameter_set['dpi'],
            close_fig=True, title_str=r'$K_{{\rm est}}={0}$'.format(K_est))

    fitting_error_all = np.array(fitting_error_all)
    # plot the objective function values against different K_est
    fig = plt.figure(figsize=(4, 2.5), dpi=90)
    ax = plt.axes([0.19, 0.185, 0.72, 0.72])
    ax.plot(K_est_lst, fitting_error_all, linewidth=1,
            color=[0, 0.447, 0.741],
            label='fitting error')
    plt.xlabel('estimated number of sources')

    plt.ylabel('fitting error')
    ax.set_title('evolution of fitting errors', position=(0.5, 1.01), fontsize=11)
    plt.grid(linestyle=':')
    plt.xlim([plt.gca().get_xlim()[0], fitting_error_all.size])
    # we want to DECREASE the model order and see the effect on the fitting error
    plt.gca().invert_xaxis()
    ax.plot(np.linspace(plt.gca().get_xlim()[0], plt.gca().get_xlim()[1], 100),
            noise_level * np.ones(100),
            color=[0.850, 0.325, 0.098],
            linewidth=1, linestyle='-.', alpha=0.7,
            label='noise level')

    plt.legend(fontsize=9, columnspacing=0.1, labelspacing=0.1,
               framealpha=0.8, frameon=True)
    if save_fig:
        file_name = fig_dir + 'model_sel_obj_vals.pdf'
        plt.savefig(file_name, format='pdf', dpi=900, transparent=True)

    plt.show()

    # reset numpy print option
    np.set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan',
                        precision=8, suppress=False, threshold=1000, formatter=None)
