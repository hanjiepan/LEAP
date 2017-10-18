"""
experiment_src_separation.py: generate the phase transition plot with real LOFAR layout
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
import setup
import numpy as np
import scipy.constants
import os
import sys
import subprocess
import getopt
import sympy
import datetime
import scipy.io
from functools import partial
from scipy import linalg
from alg_fri_planar_beamforming import planar_recon_2d_dirac_joint_beamforming
from build_linear_mapping_beamforming import planar_beamforming_func, compile_theano_func_build_G_mtx, \
    compile_theano_func_build_amp_mtx
from plotter import plot_phase_transition_2dirac
import matplotlib.pyplot as plt
from utils import planar_gen_visibility_beamforming, planar_compute_all_baselines, planar_distance

if __name__ == '__main__':
    backend = os.environ['COMPUTE_BACK_END']  # either 'cpu' or 'gpu'
    # can be either 'testing', 'production', or 'plotting'
    script_purpose = 'plotting'
    # depends on the purpose, we choose a different set of parameters
    parameter_set = {}
    if script_purpose == 'testing':
        parameter_set = {
            'coverage_rate': 0.75,
            'G_iter': 1,
            'max_noise_realisation': 1,
            'load_intermediate': False,
            'load_plot_data': False,
            'cmap': None,
            'run_data_extraction': False
        }
    elif script_purpose == 'production':
        parameter_set = {
            'coverage_rate': 1,
            'G_iter': 10,
            'max_noise_realisation': 100,
            'load_intermediate': False,
            'load_plot_data': False,
            'cmap': 'magma',
            'run_data_extraction': False
        }
    elif script_purpose == 'plotting':
        parameter_set = {
            'load_plot_data': True,
            'cmap': 'magma',
            'run_data_extraction': False
        }
    else:
        RuntimeError('Unknown script purpose: {}'.format(script_purpose))

    if sys.version_info[0] > 2:
        # data extraction relies on Python2 only code
        parameter_set['run_data_extraction'] = False

    # parse arguments
    argv = sys.argv[1:]
    sep_index_bg = 0
    sep_index_end = None
    snr_index_bg = 0
    snr_index_end = None

    try:
        opts, args = getopt.getopt(argv, "hs:e:b:t:",
                                   ["start=", "end=", "begin=", "terminal="])
    except getopt.GetoptError:
        print('experiment_src_separation.py -s <separation_start_index> '
              '-e <separation_end_index> -b <snr_begin_index> -t <snr_terminal_index>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('experiment_src_separation.py -s <start_index> -e <end_index>')
            sys.exit()
        elif opt in ('-s', '--start'):
            sep_index_bg = int(arg)
        elif opt in ('-e', '--end'):
            sep_index_end = int(arg)
        elif opt in ('-b', '--begin'):
            snr_index_bg = int(arg)
        elif opt in ('-t', '--terminal'):
            snr_index_end = int(arg)

    print('Separation index from {0} to {1}'.format(sep_index_bg, sep_index_end))

    save_fig = True  # save figure or not
    fig_format = r'pdf'  # file type used to save the figure, e.g., pdf, png, etc.
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

    # a list that contains the separation distance between two sources
    min_sep = 10 / 3600.  # 10 arc second
    max_sep = 10 / 60.  # 10 arc minute
    # separation in [radian]
    separation_lst = np.radians(np.logspace(np.log10(min_sep), np.log10(max_sep),
                                            base=10, num=10))[sep_index_bg: sep_index_end]
    print('Separation list: {0}'.format(separation_lst))

    # a list of SNRs
    snr_seq = -10 + 5 * np.arange(6 + 1)  # -10dB to 20dB

    if snr_index_end is None:
        snr_index_end = snr_seq.size

    print('SNR list: {0}'.format(snr_seq[snr_index_bg:snr_index_end]))

    np.set_printoptions(precision=3, formatter={'float': '{: 0.3f}'.format})

    if not parameter_set['load_plot_data']:
        # maximum number of noise (and signal) realisations
        max_noise_realisation = parameter_set['max_noise_realisation']

        # various experiment settings
        light_speed = scipy.constants.speed_of_light  # speed of light

        # load LOFAR layout
        num_station = 24
        time_sampling_step = 50
        time_sampling_end = 50 * 62 + 1  # 63 STI; open interval so + 1
        num_sti = (time_sampling_end - 1) // time_sampling_step + 1
        data_file_name = \
            './data/BOOTES24_SB180-189.2ch8s_SIM_{num_sti}STI_146MHz_{num_station}Station_1Subband.npz'.format(
                num_sti=num_sti,
                num_station=num_station
            )
        if parameter_set['run_data_extraction']:
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
                       '--FoV 5 ' \
                       '--modi_data_col 1 ' \
                       '--mgain 0.1 '.format(
                time_sampling_step=time_sampling_step,
                time_sampling_end=time_sampling_end,
                num_station=num_station
            )
            if subprocess.call(bash_cmd, shell=True):
                raise RuntimeError('Could not extract data!')

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

        # reconstruct point sources
        max_ini = 15  # maximum number of random initializations

        '''define the period of the (periodic)-sinc interpolation:
        the coverage_rate percentile smallest frequencies are contained in one period.
        if coverage_rate = 1, then all frequencies are completly contained.
        '''
        coverage_rate = parameter_set['coverage_rate']

        norm_factor = np.reshape(light_speed / (2 * np.pi * freq_subbands_hz),
                                 (1, 1, 1, -1), order='F')
        # normalised antenna coordinates
        p_x_normalised = np.reshape(
            r_antenna_x, (-1, num_station, num_sti, num_subband), order='F') / norm_factor
        p_y_normalised = np.reshape(
            r_antenna_y, (-1, num_station, num_sti, num_subband), order='F') / norm_factor
        p_z_normalised = np.reshape(
            r_antenna_z, (-1, num_station, num_sti, num_subband), order='F') / norm_factor

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

        # generate Dirac parameters
        alpha_ks = np.ones((K, num_subband), dtype=float)  # unit norm
        x1 = y1 = 0  # first source location

        avg_recon_dist = np.zeros((separation_lst.size, snr_seq.size), dtype=float)
        avg_recon_success_rate = np.zeros((separation_lst.size, snr_seq.size), dtype=float)

        for sep_ind, separation in enumerate(separation_lst):
            # choose the field of view based on the separation (speed consideration)
            tau_x = tau_y = float(np.radians(min(max(0.5, np.degrees(separation) * 3), FoV_degree)))
            M_tau_x = np.ceil(M * tau_x / 2) * 2 + 1  # M * tau_x is an odd number
            N_tau_y = np.ceil(N * tau_y / 2) * 2 + 1  # N * tau_y is an odd number
            tau_inter_x = sympy.Rational(M_tau_x, M)  # interpolation step size: 2 pi / tau_inter
            tau_inter_y = sympy.Rational(N_tau_y, N)

            for snr_ind, snr_loop in enumerate(snr_seq):

                if snr_index_bg <= snr_ind < snr_index_end:
                    intermediate_file_name = result_dir + \
                                             'phase_transition_plot_data_sepind{sep_ind}' \
                                             '_snr{snr_ind}_intermediate.npz'.format(
                                                 sep_ind=sep_ind, snr_ind=snr_ind)
                    if parameter_set['load_intermediate'] and os.path.isfile(intermediate_file_name):
                        intermediate_result = np.load(intermediate_file_name)
                        noise_range0 = intermediate_result['noise_realisation'] + 1
                        avg_recon_success_rate = intermediate_result['avg_recon_success_rate']
                        avg_recon_dist = intermediate_result['avg_recon_dist']
                    else:
                        noise_range0 = 0

                    for noise_realisation in range(noise_range0, max_noise_realisation):

                        rnd_angle = (np.random.rand() + noise_realisation) * np.pi / max_noise_realisation
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

                        time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                        np.savez(param_dir + 'src_param_' + time_stamp,
                                 x_ks=x_ks, y_ks=y_ks, alpha_ks=alpha_ks)

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
                                snr_data=snr_loop
                            )

                        # TODO: set verbose=False
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

                        # compute partial reconstruction error
                        dist_recon_loop, idx_sort = planar_distance(x_ks, y_ks, xk_recon, yk_recon)
                        avg_recon_dist[sep_ind, snr_ind] += dist_recon_loop

                        # deal with the specific case when only 1 Dirac is reconstructed
                        if not hasattr(xk_recon, '__iter__'):
                            xk_recon = np.array([xk_recon])
                            yk_recon = np.array([yk_recon])

                        if len(idx_sort.shape) == 1:
                            xk_recon_sorted = np.array([xk_recon])
                            yk_recon_sorted = np.array([yk_recon])
                            x_ks_sorted = np.array(x_ks[idx_sort[0]])
                            y_ks_sorted = np.array(y_ks[idx_sort[0]])
                        else:
                            xk_recon_sorted = xk_recon[idx_sort[:, 1]]
                            yk_recon_sorted = yk_recon[idx_sort[:, 1]]
                            x_ks_sorted = x_ks[idx_sort[:, 0]]
                            y_ks_sorted = y_ks[idx_sort[:, 0]]

                        for k_loop in range(min(K_est, xk_recon.size)):
                            if planar_distance(x_ks_sorted[k_loop], y_ks_sorted[k_loop],
                                               xk_recon_sorted[k_loop], yk_recon_sorted[k_loop])[0] < \
                                            0.5 * separation:
                                avg_recon_success_rate[sep_ind, snr_ind] += 1

                        # save result for every noise realization
                        np.savez(intermediate_file_name,
                                 noise_realisation=noise_realisation,
                                 avg_recon_success_rate=avg_recon_success_rate,
                                 avg_recon_dist=avg_recon_dist
                                 )

                # save after each separation cases. In case the simulation is interrupted.
                np.savez(result_dir +
                         'phase_transition_plot_data_sepind'
                         '{sep_index_bg}_to_{sep_index_end}_'
                         'snr{snr_index_bg}_to_{snr_index_end}.npz'.format(
                             sep_index_bg=sep_index_bg,
                             sep_index_end=sep_index_end - 1 if sep_index_end is not None else 'end',
                             snr_index_bg=snr_index_bg,
                             snr_index_end=snr_index_end
                         ),
                         avg_recon_dist=avg_recon_dist / max_noise_realisation,
                         avg_recon_success_rate=avg_recon_success_rate / (K_est * max_noise_realisation),
                         separation_lst=separation_lst,
                         snr_seq=snr_seq)

        avg_recon_dist /= max_noise_realisation
        avg_recon_success_rate /= K_est * max_noise_realisation

        # save plotting data
        np.savez(result_dir +
                 'phase_transition_plot_data_sepind'
                 '{sep_index_bg}_to_{sep_index_end}_'
                 'snr{snr_index_bg}_to_{snr_index_end}.npz'.format(
                     sep_index_bg=sep_index_bg,
                     sep_index_end=sep_index_end - 1 if sep_index_end is not None else 'end',
                     snr_index_bg=snr_index_bg,
                     snr_index_end=snr_index_end
                 ),
                 avg_recon_dist=avg_recon_dist,
                 avg_recon_success_rate=avg_recon_success_rate,
                 separation_lst=separation_lst,
                 snr_seq=snr_seq)

    else:
        # load plotting data
        plot_data_file_name = result_dir + \
                              'phase_transition_plot_data_sepind' \
                              '{sep_index_bg}_to_{sep_index_end}_' \
                              'snr{snr_index_bg}_to_{snr_index_end}.npz'.format(
                                  sep_index_bg=sep_index_bg,
                                  sep_index_end=sep_index_end - 1
                                  if sep_index_end is not None else 'end',
                                  snr_index_bg=snr_index_bg,
                                  snr_index_end=snr_index_end
                              )
        plot_data = np.load(plot_data_file_name)
        avg_recon_success_rate = plot_data['avg_recon_success_rate']
        separation_lst = plot_data['separation_lst']
        snr_seq = plot_data['snr_seq']

    file_name = fig_dir + 'phase_transition_2src.' + fig_format
    plot_phase_transition_2dirac(avg_recon_success_rate,
                                 separation_lst, snr_seq,
                                 save_fig, fig_format, file_name,
                                 fig_title='reconstruction success rate (LEAP)',
                                 dpi=300, cmap=parameter_set['cmap'],
                                 color_bar_min=0, close_fig=False,
                                 plt_line=False)
    plt.show()

    # reset numpy print option
    np.set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan',
                        precision=8, suppress=False, threshold=1000, formatter=None)
