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
generate the phase transition plot with real LOFAR layout (CLEAN)
"""
from __future__ import division
import setup
import numpy as np
import scipy.constants
import os
import sys
import getopt
import datetime
import subprocess
import scipy.io
from functools import partial
from scipy import linalg
from build_linear_mapping_beamforming import planar_beamforming_func
from plotter import plot_phase_transition_2dirac
import matplotlib.pyplot as plt
from utils import planar_gen_visibility_beamforming, planar_distance, \
    detect_peaks, sph2cart

if __name__ == '__main__':
    backend = os.environ['COMPUTE_BACK_END']  # either 'cpu' or 'gpu'
    # can be either 'testing', 'production', or 'plotting'
    script_purpose = 'plotting'
    # depends on the purpose, we choose a different set of parameters
    parameter_set = {}
    if script_purpose == 'testing':
        parameter_set = {
            'mgain': 0.02,
            'max_noise_realisation': 1,
            'load_intermediate': False,
            'load_plot_data': False,
            'cmap': None
        }
    elif script_purpose == 'production':
        parameter_set = {
            'mgain': 0.03,
            'max_noise_realisation': 100,
            'load_intermediate': False,
            'load_plot_data': False,
            'cmap': 'magma'
        }
    elif script_purpose == 'plotting':
        parameter_set = {
            'load_plot_data': True,
            'cmap': 'magma'
        }
    else:
        RuntimeError('Unknown script purpose: {}'.format(script_purpose))

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

        data_root_path = os.environ['DATA_ROOT_PATH']
        basefile_name = 'BOOTES24_SB180-189.2ch8s_SIM'
        ms_file_name = data_root_path + basefile_name + '.ms'
        sub_table_file_name = '{basefile_name}_every{time_sampling_step}th.ms'.format(
            basefile_name=basefile_name,
            time_sampling_step=time_sampling_step
        )
        sub_table_full_name = data_root_path + sub_table_file_name
        # extract subtable from ms file
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

        lofar_data = np.load(data_file_name)

        freq_subbands_hz = lofar_data['freq_subbands_hz']

        '''the array coordinate is arranged as a 4D matrix, where
                dimension 0: antenna index within one station
                dimension 1: station index
                dimension 2: STI index
                dimension 3: (of size 3) corresponds to x, y, and z coordinates'''
        array_coordinate = lofar_data['array_coordinate']

        sky_ra = lofar_data['RA_rad']
        sky_dec = lofar_data['DEC_rad']

        FoV_degree = lofar_data['FoV']  # field of view

        mtx_J2000_to_uvw = np.array([
            [-np.sin(sky_ra), np.cos(sky_ra), 0],
            [-np.cos(sky_ra) * np.sin(sky_dec),
             -np.sin(sky_ra) * np.sin(sky_dec),
             np.cos(sky_dec)],
            sph2cart(1, 0.5 * np.pi - sky_dec, sky_ra)
        ])

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

        norm_factor = np.reshape(light_speed / (2 * np.pi * freq_subbands_hz),
                                 (1, 1, 1, -1), order='F')
        # normalised antenna coordinates
        p_x_normalised = np.reshape(
            r_antenna_x, (-1, num_station, num_sti, num_subband), order='F') / norm_factor
        p_y_normalised = np.reshape(
            r_antenna_y, (-1, num_station, num_sti, num_subband), order='F') / norm_factor
        p_z_normalised = np.reshape(
            r_antenna_z, (-1, num_station, num_sti, num_subband), order='F') / norm_factor

        # generate Dirac parameters
        alpha_ks = np.ones((K, num_subband), dtype=float)  # unit norm
        x1 = y1 = 0  # first source location

        avg_recon_dist = np.zeros((separation_lst.size, snr_seq.size), dtype=float)
        avg_recon_success_rate = np.zeros((separation_lst.size, snr_seq.size), dtype=float)

        for sep_ind, separation in enumerate(separation_lst):

            for snr_ind, snr_loop in enumerate(snr_seq):

                if snr_index_bg <= snr_ind < snr_index_end:
                    intermediate_file_name = result_dir + \
                                             'clean_phase_transition_plot_data_sepind{sep_ind}' \
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

                        time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                        dirac_data_file_name = param_dir + 'src_param_' + time_stamp + '.npz'
                        print('Saving Dirac parameters and visibilities in {0}'.format(dirac_data_file_name))
                        np.savez(dirac_data_file_name,
                                 x_ks=x_ks, y_ks=y_ks, alpha_ks=alpha_ks,
                                 visi=visi_noisy, visi_noiseless=visi_noiseless)

                        FoV_loop = min(max(1, np.degrees(separation) * 7), FoV_degree)
                        FoV_loop_radian = float(np.radians(FoV_loop))
                        clean_img_sz = int(max(FoV_loop // (0.5 * np.degrees(separation)), 1024))

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
                            FoV=FoV_loop,
                            output_name_prefix=data_root_path + 'highres',
                            freq_channel_min=0,
                            freq_channel_max=0 + num_subband,
                            max_iter=40000,
                            mgain=parameter_set['mgain'],
                            auto_threshold=3,
                            imag_format='png',
                            dpi=300
                        )
                        if subprocess.call(bash_cmd, shell=True):
                            raise RuntimeError('wsCLEAN could not run!')

                        # load CLEAN results
                        clean_data = np.load('./data/' + sub_table_file_name[:-3] + '_modi-CLEAN_data.npz')
                        img_clean = clean_data['img_clean']
                        src_model_clean = clean_data['src_model']
                        RA_J2000_rad_grid = clean_data['x_plt_CLEAN_rad']  # in radian
                        DEC_J2000_rad_grid = clean_data['y_plt_CLEAN_rad']  # in radian
                        peak_locs = detect_peaks(
                            src_model_clean * (src_model_clean > 0.8 * src_model_clean.max())
                        )[2]
                        # coordinate conversion: SIN (from CLEAN recon) -> J2000 -> UVW
                        clean_RA_recon_J2000 = RA_J2000_rad_grid[peak_locs[0, :], peak_locs[1, :]]
                        clean_DEC_recon_J2000 = DEC_J2000_rad_grid[peak_locs[0, :], peak_locs[1, :]]
                        clean_xk_recon_J2000 = np.cos(clean_DEC_recon_J2000) * np.cos(clean_RA_recon_J2000)
                        clean_yk_recon_J2000 = np.cos(clean_DEC_recon_J2000) * np.sin(clean_RA_recon_J2000)
                        clean_zk_recon_J2000 = np.sin(clean_DEC_recon_J2000)
                        clean_recon_uvw = np.dot(mtx_J2000_to_uvw,
                                                 np.row_stack((
                                                     clean_xk_recon_J2000.flatten('F'),
                                                     clean_yk_recon_J2000.flatten('F'),
                                                     clean_zk_recon_J2000.flatten('F')
                                                 ))
                                                 )
                        clean_xk_recon = clean_recon_uvw[0, :]
                        clean_yk_recon = clean_recon_uvw[1, :]

                        clean_amp_recon = img_clean[peak_locs[0, :], peak_locs[1, :]]

                        # deal with the specific case when only 1 Dirac is reconstructed
                        if not hasattr(clean_xk_recon, '__iter__'):
                            clean_xk_recon = np.array([clean_xk_recon])
                            clean_yk_recon = np.array([clean_yk_recon])
                        # choose only K_est of them in case of over-estimation
                        if clean_xk_recon.size > K_est:
                            amp_sort_idx = np.argsort(clean_amp_recon)[::-1]
                            clean_xk_recon = clean_xk_recon[amp_sort_idx[:K_est]]
                            clean_yk_recon = clean_yk_recon[amp_sort_idx[:K_est]]

                        # compute partial reconstruction error
                        dist_recon_loop, idx_sort = planar_distance(x_ks, y_ks, clean_xk_recon, clean_yk_recon)
                        avg_recon_dist[sep_ind, snr_ind] += dist_recon_loop

                        if len(idx_sort.shape) == 1:
                            clean_xk_recon_sorted = np.array([clean_xk_recon])
                            clean_yk_recon_sorted = np.array([clean_yk_recon])
                            x_ks_sorted = np.array([x_ks[idx_sort[0]]])
                            y_ks_sorted = np.array([y_ks[idx_sort[0]]])
                        else:
                            clean_xk_recon_sorted = clean_xk_recon[idx_sort[:, 1]]
                            clean_yk_recon_sorted = clean_yk_recon[idx_sort[:, 1]]
                            x_ks_sorted = x_ks[idx_sort[:, 0]]
                            y_ks_sorted = y_ks[idx_sort[:, 0]]

                        for k_loop in range(min(K_est, clean_xk_recon_sorted.size)):
                            if planar_distance(x_ks_sorted[k_loop], y_ks_sorted[k_loop],
                                               clean_xk_recon_sorted[k_loop], clean_yk_recon_sorted[k_loop])[0] < \
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
                         'clean_phase_transition_plot_data_sepind'
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
                 'clean_phase_transition_plot_data_sepind'
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
                              'clean_phase_transition_plot_data_sepind' \
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

    file_name = fig_dir + 'clean_phase_transition_2src.' + fig_format
    plot_phase_transition_2dirac(avg_recon_success_rate,
                                 separation_lst, snr_seq,
                                 save_fig, fig_format, file_name,
                                 fig_title='reconstruction success rate (CLEAN)',
                                 dpi=300, cmap=parameter_set['cmap'],
                                 color_bar_min=0, close_fig=False,
                                 plt_line=False)
    plt.show()

    # reset numpy print option
    np.set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan',
                        precision=8, suppress=False, threshold=1000, formatter=None)
