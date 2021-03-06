"""
toothbrush_cluster_narrow_fov_experiment.py: experiments with toothbrush cluster
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
import sympy
import scipy.io
from functools import partial
from astropy import units
from astropy.coordinates import SkyCoord
from alg_fri_planar_beamforming import planar_recon_2d_dirac_joint_beamforming, \
    planar_select_reliable_recon
from build_linear_mapping_beamforming import planar_beamforming_func, \
    compile_theano_func_build_amp_mtx, compile_theano_func_build_G_mtx
from plotter import planar_plot_diracs_J2000, planar_plot_diracs_zoom_J2000
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import planar_compute_all_baselines, partition_stages, planar_distance, UVW2J2000

if __name__ == '__main__':
    np.set_printoptions(precision=3, formatter={'float': '{: 0.3f}'.format})
    script_purpose = 'plotting'  # can be either 'testing', 'production', or 'plotting'
    # depends on the purpose, we choose a different set of parameters
    parameter_set = {}
    if script_purpose == 'testing':
        parameter_set = {
            'data_file_name':
                './data/RX42_SB100-109.2ch10s_63STI_132MHz_36Station_1Subband_FoV5.npz',
            # './data/RX42_SB100-109.2ch10s_63STI_132MHz_36Station_1Subband_FoV3.npz',
            'load_intermediate_file': False,  # whether or not load the intermediate results
            'intermidiate_file_name':
                './result/intermediate_result_toothbrush_cluster.npz',
            'coverage_rate': 0.5,
            'G_iter': 2,
            'marker_scale': 0.2,
            'dpi': 300,
            'cmap': 'magma_r'  # 'Spectral_r'
        }
    elif script_purpose == 'production':
        parameter_set = {
            'data_file_name':
                './data/RX42_SB100-109.2ch10s_63STI_132MHz_36Station_1Subband_FoV5.npz',
            'load_intermediate_file': False,  # whether or not load the intermediate results
            'intermidiate_file_name':
                './result/intermediate_result_toothbrush_cluster.npz',
            'coverage_rate': 0.7,
            'G_iter': 20,
            'marker_scale': 0.2,
            'dpi': 300,
            'cmap': 'magma_r'  # 'Spectral_r'
        }
    elif script_purpose == 'plotting':
        parameter_set = {
            'data_file_name':
                './data/RX42_SB100-109.2ch10s_63STI_132MHz_36Station_1Subband_FoV5.npz',
            'load_intermediate_file': True,  # whether or not load the intermediate results
            'intermidiate_file_name':
                './result/intermediate_result_toothbrush_cluster.npz',
            'coverage_rate': 0.7,
            'G_iter': 20,
            'marker_scale': 0.2,
            'dpi': 300,
            'cmap': 'magma_r'  # 'Spectral_r'
        }
        assert parameter_set['load_intermediate_file']  # has to load precomputed result
    else:
        RuntimeError('Unknown script purpose: {}'.format(script_purpose))

    backend = os.environ['COMPUTE_BACK_END']  # either 'cpu' or 'gpu'
    save_fig = True
    fig_dir = './result/'
    if save_fig and not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # compile theano functions if backend == 'gpu'
    if backend == 'gpu':
        theano_build_G_func = compile_theano_func_build_G_mtx()
        theano_build_amp_func = compile_theano_func_build_amp_mtx()
    else:
        theano_build_G_func = None
        theano_build_amp_func = None

    intermidiate_file_name = parameter_set['intermidiate_file_name']
    load_intermediate_file = parameter_set['load_intermediate_file']

    # experimental parameters
    light_speed = scipy.constants.speed_of_light  # speed of light

    # load data file
    data_file_name = parameter_set['data_file_name']

    # maximum number of sources in the catalog to be extracted. Sources with the largest amplitudes are taken.
    max_catalog_src_num = float('inf')  # if inf, then take all catalog sources

    # list of background images used
    bg_img_lst = ['dirty', 'CLEAN', 'CS']

    if data_file_name[-3:] == 'npz':
        lofar_data = np.load(data_file_name)

        freq_subbands_hz = lofar_data['freq_subbands_hz']

        array_coordinate = lofar_data['array_coordinate']

        '''the visibility measurements are arranged as a 3D matrix, where
        dimension 0: cross-correlation index
        dimension 1: STI index
        dimension 2: subband index'''
        visi_noisy = lofar_data['visi_noisy']

        background_img = {}
        if 'img_dirty' in lofar_data:
            background_img['dirty'] = lofar_data['img_dirty']
        else:
            background_img['dirty'] = None

        if 'img_clean' in lofar_data:
            background_img['CLEAN'] = lofar_data['img_clean']
        else:
            background_img['CLEAN'] = None

        if 'img_cs' in lofar_data:
            background_img['CS'] = lofar_data['img_cs']
        else:
            background_img['CS'] = None

        if 'img_lsq' in lofar_data:
            background_img['blueBuild'] = lofar_data['img_lsq']
        else:
            background_img['blueBuild'] = None

        x_plt = lofar_data['x_plt']
        y_plt = lofar_data['y_plt']

        if 'xlabels' in lofar_data:
            xticklabels = lofar_data['xlabels'].tolist()
        else:
            xticklabels = None
        if 'ylabels' in lofar_data:
            yticklabels = lofar_data['ylabels'].tolist()
        else:
            yticklabels = None

        sky_ra = lofar_data['RA_rad']
        sky_dec = lofar_data['DEC_rad']

        FoV_degree = lofar_data['FoV']  # field of view

        if lofar_data['skycatalog_intensities'] is not None and \
                lofar_data['skycatalog_U'] is not None and \
                lofar_data['skycatalog_V'] is not None and \
                lofar_data['nvss_skycatalog_intensities'] is not None and \
                lofar_data['nvss_skycatalog_U'] is not None and \
                lofar_data['nvss_skycatalog_V'] is not None:
            catalog_available = True
            skycatalog_intensities = lofar_data['skycatalog_intensities']
            extract_len = min(skycatalog_intensities.size, max_catalog_src_num)
            skycatalog_intensities = skycatalog_intensities[:extract_len]
            skycatalog_U = lofar_data['skycatalog_U'][:extract_len]
            skycatalog_V = lofar_data['skycatalog_V'][:extract_len]

            NVSS_skycatalog_intensities = lofar_data['nvss_skycatalog_intensities']
            NVSS_skycatalog_U = lofar_data['nvss_skycatalog_U']
            NVSS_skycatalog_V = lofar_data['nvss_skycatalog_V']
        else:
            catalog_available = False
            skycatalog_intensities = None
            skycatalog_U = None
            skycatalog_V = None

            NVSS_skycatalog_intensities = None
            NVSS_skycatalog_U = None
            NVSS_skycatalog_V = None

    else:
        raise NameError('Unrecognized data file.')

    '''the array coordinate is arranged as a 4D matrix, where
        dimension 0: antenna index within one station
        dimension 1: station index
        dimension 2: STI index
        dimension 3: (of size 3) corresponds to x, y, and z coordinates'''
    # number of antennas, stations, short time intervals (STI), xyz
    assert array_coordinate.shape[-1] == 3
    num_antenna, num_station, num_sti = array_coordinate.shape[:-1]
    num_subband = np.asarray(freq_subbands_hz).size

    # convert to usable data
    r_antenna_x = array_coordinate[:, :, :num_sti, 0]
    r_antenna_y = array_coordinate[:, :, :num_sti, 1]
    r_antenna_z = array_coordinate[:, :, :num_sti, 2]

    # number of point sources
    K_est = 100  # estimated number of point sources
    plane_norm_vec = (0, 0, 1)

    # reconstruct point sources
    max_ini = 15  # maximum number of random initializations
    tau_x = tau_y = float(np.radians(FoV_degree))
    '''define the period of the (periodic)-sinc interpolation:
    the coverage_rate percentile smallest frequencies are contained in one period.
    if coverage_rate = 1, then all frequencies are completely contained.
    '''
    coverage_rate = parameter_set['coverage_rate']

    norm_factor = np.reshape(light_speed / (2 * np.pi * freq_subbands_hz),
                             (1, 1, 1, -1), order='F')
    # normalised antenna coordinates
    p_y_normalised = np.reshape(
        r_antenna_y, (-1, num_station, num_sti, 1), order='F') / norm_factor
    p_x_normalised = np.reshape(
        r_antenna_x, (-1, num_station, num_sti, 1), order='F') / norm_factor
    p_z_normalised = np.reshape(
        r_antenna_z, (-1, num_station, num_sti, 1), order='F') / norm_factor

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

    # for the first stage
    K_est_stage0 = 100
    removal_blk_len0 = 60

    removal_blk_len = 6
    stage_blk_len = 8
    K_est_stage_lst, removal_blk_len_lst = \
        partition_stages(K_est - (K_est_stage0 - removal_blk_len0),
                         stage_blk_len, removal_blk_len)

    K_est_stage_lst.insert(0, K_est_stage0)
    removal_blk_len_lst.insert(0, removal_blk_len0)
    removal_blk_len_lst.append(0)

    print(K_est_stage_lst, len(K_est_stage_lst))
    print(removal_blk_len_lst, len(removal_blk_len_lst))

    max_stage = len(K_est_stage_lst)

    partial_beamforming_func = partial(planar_beamforming_func,
                                       strategy='matched',
                                       x0=plane_norm_vec[0],
                                       y0=plane_norm_vec[1])

    stage0 = 0  # default starting index for different stages
    # load intermediate results if available
    if load_intermediate_file and os.path.isfile(intermidiate_file_name):
        intermidiate_result = np.load(intermidiate_file_name)
        stage0 = intermidiate_result['stages'].tolist() + 1
        xk_recon = intermidiate_result['xk_recon']
        yk_recon = intermidiate_result['yk_recon']

    if script_purpose != 'plotting':
        for stages in range(stage0, max_stage):
            # estimated number of Diracs for the PARTIAL reconstruction
            K_est_stage = K_est_stage_lst[stages]

            if stages == 0:
                file_name = (fig_dir + 'planar_K_{0}_numSta_{1}_locations_stage{2}'
                             ).format(repr(K_est), repr(num_station), stages)
                for bg_img in bg_img_lst:
                    planar_plot_diracs_J2000(
                        x_plt, y_plt,
                        RA_focus_rad=sky_ra, DEC_focus_rad=sky_dec,
                        background_img=background_img[bg_img],
                        cmap=parameter_set['cmap'],
                        marker_scale=parameter_set['marker_scale'],
                        save_fig=save_fig,
                        file_name=file_name + '_bg_img_' + bg_img.lower(),
                        file_format='png', dpi=parameter_set['dpi'], close_fig=True,
                        title_str=bg_img + ' image')

                xk_recon, yk_recon, alpha_k_recon = \
                    planar_recon_2d_dirac_joint_beamforming(
                        visi_noisy, r_antenna_x, r_antenna_y,
                        2 * np.pi * freq_subbands_hz, light_speed, K=K_est_stage,
                        tau_x=tau_x, tau_y=tau_y, M=M, N=N, tau_inter_x=tau_inter_x,
                        tau_inter_y=tau_inter_y, max_ini=max_ini, num_rotation=1,
                        G_iter=parameter_set['G_iter'],
                        plane_norm_vec=plane_norm_vec, verbose=True,
                        backend=backend, theano_build_G_func=theano_build_G_func,
                        theano_build_amp_func=theano_build_amp_func
                    )
            else:
                xk_recon, yk_recon, alpha_k_recon = planar_select_reliable_recon(
                    visi_noisy, r_antenna_x, r_antenna_y, xk_recon, yk_recon,
                    2 * np.pi * freq_subbands_hz, light_speed,
                    partial_beamforming_func, num_station, num_subband, num_sti,
                    removal_blk_len_lst[stages - 1],
                    theano_func=theano_build_amp_func,
                    backend=backend
                )

                if catalog_available:
                    # compute partial reconstruction error
                    dist_recon_stage, idx_sort = \
                        planar_distance(skycatalog_U, skycatalog_V, xk_recon, yk_recon)
                    # in degree, minute, and second representation
                    dist_recon_stage_dms = SkyCoord(
                        ra=0, dec=dist_recon_stage, unit=units.radian
                    ).to_string('dms').split(' ')[1]
                    print('Partial recon error: {0}'.format(dist_recon_stage_dms))

                file_name = (fig_dir + 'planar_K_{0}_numSta_{1}_locations_stage{2}'
                             ).format(repr(K_est), repr(num_station), stages)
                for bg_img in bg_img_lst:
                    planar_plot_diracs_J2000(
                        x_plt, y_plt,
                        RA_focus_rad=sky_ra, DEC_focus_rad=sky_dec,
                        x_ref=skycatalog_U[idx_sort[:, 0]] if catalog_available else None,
                        y_ref=skycatalog_V[idx_sort[:, 0]] if catalog_available else None,
                        amplitude_ref=skycatalog_intensities[idx_sort[:, 0]] if catalog_available else None,
                        x_recon=xk_recon,
                        y_recon=yk_recon,
                        amplitude_recon=np.mean(alpha_k_recon, axis=1),
                        background_img=background_img[bg_img],
                        cmap=parameter_set['cmap'],
                        marker_scale=parameter_set['marker_scale'],
                        save_fig=save_fig,
                        file_name=file_name + '_bg_img_' + bg_img.lower(),
                        label_ref_sol='catalog', label_recon='reconstruction',
                        legend_loc=2, file_format='png',
                        dpi=parameter_set['dpi'], close_fig=True
                    )

                if K_est_stage > 0:
                    xk_recon, yk_recon, alpha_k_recon = \
                        planar_recon_2d_dirac_joint_beamforming(
                            visi_noisy, r_antenna_x, r_antenna_y,
                            2 * np.pi * freq_subbands_hz, light_speed, K=K_est_stage,
                            tau_x=tau_x, tau_y=tau_y, M=M, N=N, tau_inter_x=tau_inter_x,
                            tau_inter_y=tau_inter_y, max_ini=max_ini, num_rotation=1,
                            G_iter=parameter_set['G_iter'],
                            plane_norm_vec=plane_norm_vec, verbose=True,
                            backend=backend, theano_build_G_func=theano_build_G_func,
                            theano_build_amp_func=theano_build_amp_func,
                            x_ref=xk_recon, y_ref=yk_recon)

            # save intermediate result in case the simulation got interrupted
            np.savez(intermidiate_file_name,
                     xk_recon=xk_recon, yk_recon=yk_recon,
                     alpha_k_recon=alpha_k_recon, stages=stages)
    else:
        # the initial images without catalog or reconstructions
        file_name = (fig_dir + 'planar_K_{0}_numSta_{1}_locations_stage{2}'
                     ).format(repr(K_est), repr(num_station), 0)
        for bg_img in bg_img_lst:
            planar_plot_diracs_J2000(
                x_plt, y_plt,
                RA_focus_rad=sky_ra, DEC_focus_rad=sky_dec,
                background_img=background_img[bg_img],
                cmap=parameter_set['cmap'],
                save_fig=save_fig,
                file_name=file_name + '_bg_img_' + bg_img.lower() + '_extSrc',
                file_format='png', dpi=parameter_set['dpi'], close_fig=True,
                title_str=bg_img + ' image')
            # zoom-in plot
            planar_plot_diracs_zoom_J2000(
                x_plt, y_plt, zoom_box=[0.45, 0.46, 0.1, 0.1],
                RA_focus_rad=sky_ra, DEC_focus_rad=sky_dec,
                background_img=background_img[bg_img],
                cmap=parameter_set['cmap'],
                save_fig=save_fig,
                file_name=file_name + '_bg_img_' + bg_img.lower() + '_extSrc_zoom',
                file_format='png', dpi=parameter_set['dpi'], close_fig=True,
                title_str=bg_img + ' image (zoom-in)')

        xk_recon, yk_recon, alpha_k_recon = planar_select_reliable_recon(
            visi_noisy, r_antenna_x, r_antenna_y, xk_recon, yk_recon,
            2 * np.pi * freq_subbands_hz, light_speed,
            partial_beamforming_func, num_station, num_subband, num_sti,
            removal_blk_len_lst[stage0 - 1],
            theano_func=theano_build_amp_func,
            backend=backend
        )

    if catalog_available:
        # compute partial reconstruction error
        dist_recon, idx_sort = \
            planar_distance(skycatalog_U, skycatalog_V, xk_recon, yk_recon)
        # in degree, minute, and second representation
        dist_recon_dms = SkyCoord(
            ra=0, dec=dist_recon, unit=units.radian
        ).to_string('dms').split(' ')[1]

        # print reconstruction results
        print('Reconstructed locations (in degrees) and amplitudes:')
        if xk_recon.size == 1:
            print('Reconstructed  horizontal locations  : {0}\n'.format(np.degrees(xk_recon)))
            print('Reconstructed vertical locations     : {0}\n'.format(np.degrees(yk_recon)))
        else:
            # convert to hmsdms format
            RA_catalog_hms, DEC_catalog_dms = \
                UVW2J2000(sky_ra, sky_dec,
                          u_rad=skycatalog_U[idx_sort[:, 0]],
                          v_rad=skycatalog_V[idx_sort[:, 0]],
                          convert_dms=True)[-2:]

            RA_recon_hms, DEC_recon_dms = \
                UVW2J2000(sky_ra, sky_dec,
                          u_rad=xk_recon[idx_sort[:, 1]],
                          v_rad=yk_recon[idx_sort[:, 1]],
                          convert_dms=True)[-2:]

            print('Catalog RA       : {0}\n'.format(RA_catalog_hms))
            print('Reconstructed RA : {0}\n'.format(RA_recon_hms))

            print('Catalog DEC      : {0}\n'.format(DEC_catalog_dms))
            print('Reconstructed DEC: {0}\n'.format(DEC_recon_dms))

        print('Reconstructed locations error: {0}'.format(dist_recon_dms))
    else:
        # print reconstruction results
        print('Reconstructed locations (in degrees) and amplitudes:')
        if xk_recon.size == 1:
            print('Reconstructed  horizontal locations  : {0}\n'.format(np.degrees(xk_recon)))
            print('Reconstructed vertical locations     : {0}\n'.format(np.degrees(yk_recon)))
        else:
            print('Reconstructed horizontal locations   : {0}\n'.format(np.degrees(xk_recon)))
            print('Reconstructed vertical locations     : {0}\n'.format(np.degrees(yk_recon)))

    # reset numpy print option
    np.set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan',
                        precision=8, suppress=False, threshold=1000, formatter=None)

    # plot results
    # save data needed for plotting
    np.savez(
        './result/plot_data_beamforming_toothbrush.npz',
        K_est=K_est, num_station=num_station,
        x_plt=x_plt, y_plt=y_plt,
        skycatalog_U=skycatalog_U, skycatalog_V=skycatalog_V,
        skycatalog_intensities=skycatalog_intensities,
        NVSS_skycatalog_U=NVSS_skycatalog_U,
        NVSS_skycatalog_V=NVSS_skycatalog_V,
        NVSS_skycatalog_intensities=NVSS_skycatalog_intensities,
        xk_recon=xk_recon, yk_recon=yk_recon,
        alpha_k_recon=alpha_k_recon,
        background_img=np.array(background_img),
        save_fig=save_fig
    )

    # load data
    plot_data = np.load('./result/plot_data_beamforming_toothbrush.npz')
    K_est = plot_data['K_est'].tolist()
    num_station = plot_data['num_station'].tolist()
    x_plt = plot_data['x_plt']
    y_plt = plot_data['y_plt']

    xk_recon = plot_data['xk_recon']
    yk_recon = plot_data['yk_recon']
    alpha_k_recon = plot_data['alpha_k_recon']

    background_img = plot_data['background_img'].tolist()

    save_fig = plot_data['save_fig'].tolist()

    skycatalog_U = plot_data['skycatalog_U']
    skycatalog_V = plot_data['skycatalog_V']
    skycatalog_intensities = plot_data['skycatalog_intensities']
    NVSS_U = plot_data['NVSS_skycatalog_U']
    NVSS_V = plot_data['NVSS_skycatalog_V']
    NVSS_intensities = plot_data['NVSS_skycatalog_intensities']

    file_name = (fig_dir + 'planar_K_{0}_numSta_{1}_locations').format(repr(K_est), repr(num_station))
    for bg_img in bg_img_lst:
        planar_plot_diracs_J2000(
            x_plt, y_plt,
            RA_focus_rad=sky_ra, DEC_focus_rad=sky_dec,
            x_ref=skycatalog_U[idx_sort[:, 0]] if catalog_available else None,
            y_ref=skycatalog_V[idx_sort[:, 0]] if catalog_available else None,
            amplitude_ref=skycatalog_intensities[idx_sort[:, 0]]
            if catalog_available else None,
            x_recon=xk_recon,
            y_recon=yk_recon,
            amplitude_recon=np.mean(alpha_k_recon, axis=1),
            background_img=background_img[bg_img],
            cmap=parameter_set['cmap'],
            marker_scale=parameter_set['marker_scale'],
            save_fig=save_fig,
            file_name=file_name + '_bg_img_' + bg_img.lower() + '_extSrc',
            label_ref_sol='catalog', label_recon='reconstruction', legend_loc=2,
            file_format='png', dpi=600, close_fig=False
        )
        # zoom in plot of the center of the field of view around the extended source
        # back ground image, catalog data TGSS
        planar_plot_diracs_zoom_J2000(
            x_plt, y_plt, zoom_box=[0.45, 0.46, 0.1, 0.1],
            RA_focus_rad=sky_ra, DEC_focus_rad=sky_dec,
            x_ref=skycatalog_U[idx_sort[:, 0]] if catalog_available else None,
            y_ref=skycatalog_V[idx_sort[:, 0]] if catalog_available else None,
            amplitude_ref=skycatalog_intensities[idx_sort[:, 0]]
            if catalog_available else None,
            x_recon=xk_recon,
            y_recon=yk_recon,
            amplitude_recon=np.mean(alpha_k_recon, axis=1),
            max_amp_ref=np.max(skycatalog_intensities[idx_sort[:, 0]]) * 0.8
            if catalog_available else None,
            background_img=background_img[bg_img],
            cmap=parameter_set['cmap'],
            marker_scale=parameter_set['marker_scale'] * 5,
            legend_marker_scale=0.5,
            save_fig=save_fig,
            file_name=file_name + '_bg_img_' + bg_img.lower() + '_extSrc_zoom',
            title_str='LEAP reconstruction (zoom-in)',
            label_ref_sol='catalog (TGSS)',
            label_recon='reconstruction', legend_loc=2,
            file_format='png', dpi=600, close_fig=False
        )

    # zoom in plot of the center of the field of view around the extended source
    # back ground image, catalog data NVSS
    if catalog_available:
        # compute partial reconstruction error
        nvss_dist_recon, nvss_idx_sort = \
            planar_distance(NVSS_U, NVSS_V, xk_recon, yk_recon)
        # in degree, minute, and second representation
        nvss_dist_recon_dms = SkyCoord(
            ra=0, dec=nvss_dist_recon, unit=units.radian
        ).to_string('dms').split(' ')[1]
        print('Reconstructed locations error (NVSS catalog): {0}'.format(nvss_dist_recon_dms))

    for bg_img in bg_img_lst:
        planar_plot_diracs_zoom_J2000(
            x_plt, y_plt, zoom_box=[0.45, 0.46, 0.1, 0.1],
            RA_focus_rad=sky_ra, DEC_focus_rad=sky_dec,
            x_ref=NVSS_U[nvss_idx_sort[:, 0]] if catalog_available else None,
            y_ref=NVSS_V[nvss_idx_sort[:, 0]] if catalog_available else None,
            amplitude_ref=NVSS_intensities[nvss_idx_sort[:, 0]]
            if catalog_available else None,
            x_recon=xk_recon,
            y_recon=yk_recon,
            amplitude_recon=np.mean(alpha_k_recon, axis=1),
            max_amp_ref=np.max(NVSS_intensities[nvss_idx_sort[:, 0]]) * 0.7
            if catalog_available else None,
            background_img=background_img[bg_img],
            cmap=parameter_set['cmap'],
            marker_scale=parameter_set['marker_scale'] * 5,
            legend_marker_scale=0.5,
            save_fig=save_fig,
            file_name=file_name + '_bg_img_' + bg_img.lower() + '_extSrc_zoom_nvss',
            title_str='LEAP reconstruction (zoom-in)',
            label_ref_sol='catalog (NVSS)',
            label_recon='reconstruction', legend_loc=2,
            file_format='png', dpi=parameter_set['dpi'], close_fig=False
        )

    plt.show()
