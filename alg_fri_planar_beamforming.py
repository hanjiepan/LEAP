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
import time
import numexpr as ne
from skimage.util.shape import view_as_blocks
import numpy as np
import scipy as sp
from scipy import linalg
import scipy.optimize
from functools import partial
from joblib import Parallel, delayed
from poly_commn_roots import find_roots
from build_linear_mapping_beamforming import planar_beamforming_func,\
    planar_mtx_freq2visibility_beamforming,\
    planar_mtx_fri2visibility_beamforming, planar_update_G_ri_beamforming,\
    planar_build_mtx_amp_ri_beamforming, planar_update_G_beamforming,\
    planar_build_mtx_amp_beamforming, planar_build_inrange_sel_mtx
from utils import hermitian_expansion, R_mtx_joint, R_mtx_joint_ri,\
    convmtx2_valid, cpx_mtx2real, output_shrink, R_mtx_joint_ri_half,\
    T_mtx_joint_ri_half, hermitian_expan_mtx


def planar_recon_2d_dirac_joint_beamforming(a, p_x, p_y, omega_bands, light_speed,
                                            K, tau_x, tau_y,
                                            M, N, tau_inter_x, tau_inter_y,
                                            noise_level=0, max_ini=20,
                                            stop_cri='max_iter', num_rotation=1,
                                            G_iter=1, plane_norm_vec=None,
                                            verbose=False, backend='cpu',
                                            theano_build_G_func=None,
                                            theano_build_amp_func=None,
                                            **kwargs):
    """
    INTERFACE for joint estimation of the Dirac deltas x and y locations in the UVW
    coordinates with beam-forming. Here, we have considered the problem in the tangential plane.
    :param a: the measured (complex-valued) visibilities in a 3D matrix form, where
              1) dimension 0: cross-correlation index
              2) dimension 1: different short time intervals (STI-s)
              3) dimension 2: different sub-bands
    :param p_x: a 4D matrix that contains antennas' x-coordinates, where
              1) dimension 0: antenna coordinate within each station
              2) dimension 1: station index
              3) dimension 2: short time interval (STI) index
              4) dimension 3: sub-band index
    :param p_y: a 4D matrix that contains antennas' y-coordinates
    :param omega_bands: mid-band (ANGULAR) frequencies [radian/sec]
    :param light_speed: speed of light
    :param K: number of point sources
    :param tau_x:
    :param tau_y:
    :param M: [M,N] the equivalence of "bandwidth" in time domain (because of duality)
    :param N: [M,N] the equivalence of "bandwidth" in time domain (because of duality)
    :param tau_inter_x: the Fourier domain interpolation step-size is 2 pi / tau_inter
    :param tau_inter_y: the Fourier domain interpolation step-size is 2 pi / tau_inter
    :param noise_level: noise level in the measured visibilities
    :param max_ini: maximum number of random initialisation used
    :param stop_cri: either 'mse' or 'max_iter'
    :param num_rotation: number of random rotation applied. if num_rotation == 1, then
            no rotation is applied.
    :param G_iter: number of iterations used to update the linear mapping from the FRI
            sequence to the visibility measurements based on the reconstructed Dirac
            locations. If G_iter == 1, then no update.
    :param plane_norm_vec: norm vector of the imaging plane. This vector is given by the
            center of the field-of-view (FoV) of the radio-telescope. If the argument is not
            None, then reconstructed sources that are not in the same imaging hemisphere will
            be reflected back to the imaging hemisphere.
    :param use_lu: whether to use LU decomposition to improved efficiency or not
    :param verbose: whether output intermediate results for debugging or not
    :param backend: either 'cpu' or 'gpu'
    :param kwargs: include
            x_ref: reference Dirac horizontal locations in the UVW coordinate, e.g., previous reconstruction
            y_ref: reference Dirac vertical locations in the UVW coordinate, e.g., previous reconstruction
    :return:
    """
    if 'x_ref' in kwargs and 'y_ref' in kwargs:
        ref_sol_available = True
        x_ref = kwargs['x_ref']
        y_ref = kwargs['y_ref']
        xy_ref = np.vstack((np.array(x_ref).flatten('F'), np.array(y_ref).flatten('F')))
        K_ref = x_ref.size
    else:
        ref_sol_available = False
        K_ref = 0

    if 'store_obj_val' in kwargs:
        store_obj_val = kwargs['store_obj_val']
    else:
        store_obj_val = False

    if store_obj_val:
        obj_val_all = []

    if 'return_error' in kwargs:
        return_error = kwargs['return_error']
    else:
        return_error = False

    assert len(a.shape) == 3
    # whether update the linear mapping or not
    update_G = (G_iter != 1)

    num_bands = a.shape[2]  # number of bands considered
    num_sti = a.shape[1]  # number of short time intervals
    num_station = p_x.shape[1]  # number of stations

    assert a.shape[0] == num_station * (num_station - 1)
    assert a.shape[2] == np.array(omega_bands).size

    '''verify input parameters'''
    # interpolation points cannot be too far apart
    assert tau_inter_x >= tau_x and tau_inter_y >= tau_y

    # M * tau is an odd number
    assert M * tau_inter_x % 2 == 1 and N * tau_inter_y % 2 == 1

    # G is a tall matrix
    assert M * tau_inter_x * N * tau_inter_y <= num_station * (num_station - 1) * num_sti * num_bands

    # minimum number of annihilation equations compared with number of unknowns
    # -> no longer necessary (the calculation here did not take into account of
    # the non-vector like annihilating filters)
    # assert (M * tau_inter_x - K) * N * tau_inter_y >= K + 1 and \
    #        (N * tau_inter_y - K) * M * tau_inter_x >= K + 1

    tau_inter_x = float(tau_inter_x)
    tau_inter_y = float(tau_inter_y)

    m_limit = int(np.floor(M * tau_inter_x // 2))
    n_limit = int(np.floor(N * tau_inter_y // 2))

    if plane_norm_vec is not None:
        plane_norm_vec0, plane_norm_vec1, plane_norm_vec2 = plane_norm_vec
    else:
        plane_norm_vec0, plane_norm_vec1, plane_norm_vec2 = 0, 0, 1

    norm_factor = np.reshape(light_speed / omega_bands, (1, 1, 1, -1), order='F')
    # normalised antenna coordinates
    p_x_normalised = np.reshape(p_x, (-1, num_station, num_sti, 1),
                                order='F') / norm_factor
    p_y_normalised = np.reshape(p_y, (-1, num_station, num_sti, 1),
                                order='F') / norm_factor

    # reshape a_ri to a 2D matrix such that:
    # 1) dimension0: concatenated 'a' for different STI-s but the SAME sub-band frequency
    # 2) dimension1: different subband frequencies
    a = np.reshape(a, (-1, num_bands), order='F')
    a_lst = [a[:, band_count] for band_count in range(num_bands)]  # the list representation

    min_error_all = float('inf')

    for rand_rotate in np.linspace(0, np.pi, num=num_rotation, endpoint=False):
        # apply a random rotation
        if num_rotation == 1:  # <= i.e., no rotation
            rotate_angle = 0
        else:
            rotate_angle = rand_rotate + np.pi / num_rotation * (np.random.rand() - 0.5)

        # build rotation matrix
        rotate_mtx = np.array([[np.cos(rotate_angle), -np.sin(rotate_angle)],
                               [np.sin(rotate_angle), np.cos(rotate_angle)]])

        # rotate antenna steering vector
        p_rotated = np.dot(rotate_mtx,
                           np.vstack((
                               p_x_normalised.flatten('F'),
                               p_y_normalised.flatten('F')
                           ))
                           )
        p_x_rotated = np.reshape(p_rotated[0, :], p_x_normalised.shape, order='F')
        p_y_rotated = np.reshape(p_rotated[1, :], p_y_normalised.shape, order='F')

        x0_rotated = np.cos(rotate_angle) * plane_norm_vec0 - \
                     np.sin(rotate_angle) * plane_norm_vec1
        y0_rotated = np.sin(rotate_angle) * plane_norm_vec0 + \
                     np.cos(rotate_angle) * plane_norm_vec1

        partial_beamforming_func = partial(planar_beamforming_func,
                                           strategy='matched',
                                           x0=x0_rotated, y0=y0_rotated)

        tic = time.time()
        # linear transformation matrix that maps uniform samples of sinusoids to visibilities
        mtx_freq2visibility = planar_mtx_freq2visibility_beamforming(
            p_x_rotated, p_y_rotated, M, N, tau_inter_x, tau_inter_y,
            partial_beamforming_func, num_station, num_sti, num_bands,
            backend=backend, theano_func=theano_build_G_func
        )

        G_lst = planar_mtx_fri2visibility_beamforming(mtx_freq2visibility, real_value=False)
        del mtx_freq2visibility
        toc = time.time()
        print('time takes to build G: {0:.2f}sec'.format(toc - tic))

        for count_G in range(G_iter):

            K_alg = min(determine_max_coef_sz(2 * n_limit + 1, 2 * m_limit + 1), 400)

            if count_G == 0:
                '''
                Extract the measurements and the lines in G such that the baselines are
                included within on period of the periodic sinc interpolation.
                Here we use the centroid of a station as a crude approximation to determine
                whether to include a given visibility measurement in the first estimation or not.
                '''
                extract_indicator_lst = \
                    planar_build_inrange_sel_mtx(p_x_rotated, p_y_rotated,
                                                 num_station, num_bands, num_sti,
                                                 M * np.pi, N * np.pi)
                G_lst_count0 = [
                    G_lst[band_count][extract_indicator_lst[band_count], :]
                    for band_count in range(num_bands)
                ]
                a_count0_lst = [
                    a_lst[band_count][extract_indicator_lst[band_count]]
                    for band_count in range(num_bands)
                ]

                # make sure the subset of measurements are still enough for the algorithm,
                # i.e. G is a tall matrix
                assert M * tau_inter_x * N * tau_inter_y <= np.concatenate(a_count0_lst).size

                if ref_sol_available:
                    xy_ref_rotated = np.dot(rotate_mtx, xy_ref)
                    x_ref_rotated = xy_ref_rotated[0, :]
                    y_ref_rotated = xy_ref_rotated[1, :]

                    G_amp_ref_lst = [
                        planar_build_mtx_amp_beamforming(
                            p_x_rotated[:, :, :, band_count],
                            p_y_rotated[:, :, :, band_count],
                            x_ref_rotated, y_ref_rotated,
                            partial_beamforming_func,
                            theano_func=theano_build_amp_func,
                            backend=backend
                        )
                        for band_count in range(num_bands)
                    ]

                    G_amp_ref_lst_count0 = [
                        G_amp_ref_lst[band_count][extract_indicator_lst[band_count], :]
                        for band_count in range(num_bands)
                    ]

                    # c_row_opt, c_col_opt, min_error, b_opt_lst, ini = \
                    #     planar_dirac_recon_alg_joint(G_lst_count0, a_count0_lst, K_alg, M, N,
                    #                                  tau_inter_x, tau_inter_y,
                    #                                  noise_level, max_ini, stop_cri,
                    #                                  G_amp_ref_lst=G_amp_ref_lst_count0)
                    c_row_opt, c_col_opt = \
                        planar_dirac_recon_alg_joint_slow(G_lst_count0, a_count0_lst, K_alg, M, N,
                                                          tau_inter_x, tau_inter_y,
                                                          noise_level, max_ini=max_ini, stop_cri=stop_cri,
                                                          G_amp_ref_lst=G_amp_ref_lst_count0)[:2]

                    xk_recon_rotated, yk_recon_rotated = planar_extract_innovation(
                        a_count0_lst, K, c_row_opt, c_col_opt, p_x_rotated, p_y_rotated,
                        tau_inter_x, tau_inter_y, partial_beamforming_func,
                        theano_func=theano_build_amp_func,
                        backend=backend,
                        G_amp_ref_lst=G_amp_ref_lst_count0,
                        x_ref_rotated=x_ref_rotated,
                        y_ref_rotated=y_ref_rotated,
                        extract_indicator_lst=extract_indicator_lst
                    )
                else:
                    # c_row_opt, c_col_opt, min_error, b_opt_lst, ini = \
                    #     planar_dirac_recon_alg_joint(G_lst_count0, a_count0_lst, K_alg, M, N,
                    #                                  tau_inter_x, tau_inter_y,
                    #                                  noise_level, max_ini, stop_cri)
                    c_row_opt, c_col_opt = \
                        planar_dirac_recon_alg_joint_slow(G_lst_count0, a_count0_lst, K_alg, M, N,
                                                          tau_inter_x, tau_inter_y,
                                                          noise_level, max_ini=max_ini,
                                                          stop_cri=stop_cri)[:2]

                    xk_recon_rotated, yk_recon_rotated = planar_extract_innovation(
                        a_count0_lst, K, c_row_opt, c_col_opt,
                        p_x_rotated, p_y_rotated,
                        tau_inter_x, tau_inter_y,
                        partial_beamforming_func,
                        theano_func=theano_build_amp_func,
                        backend=backend,
                        extract_indicator_lst=extract_indicator_lst
                    )

                G_lst = planar_update_G_beamforming(
                    xk_recon_rotated[K_ref:], yk_recon_rotated[K_ref:],
                    M, N, tau_inter_x, tau_inter_y,
                    p_x_rotated, p_y_rotated, G_lst,
                    beam_weights_func=partial_beamforming_func,
                    num_station=num_station, num_sti=num_sti, num_bands=num_bands,
                    theano_func=theano_build_amp_func, backend=backend
                )

            if ref_sol_available:
                # c_row_opt, c_col_opt, min_error, b_opt_lst, ini = \
                #     planar_dirac_recon_alg_joint(G_lst, a_lst, K, M, N,
                #                                  tau_inter_x, tau_inter_y,
                #                                  noise_level, max_ini, stop_cri,
                #                                  G_amp_ref_lst=G_amp_ref_lst)
                c_row_opt, c_col_opt, min_error, b_opt_lst, ini = \
                    planar_dirac_recon_alg_joint_slow(G_lst, a_lst, K, M, N,
                                                      tau_inter_x, tau_inter_y,
                                                      noise_level, max_ini, stop_cri,
                                                      G_amp_ref_lst=G_amp_ref_lst)
                xk_recon_rotated, yk_recon_rotated = planar_extract_innovation(
                    a_lst, K, c_row_opt, c_col_opt, p_x_rotated, p_y_rotated,
                    tau_inter_x, tau_inter_y, partial_beamforming_func,
                    theano_func=theano_build_amp_func,
                    backend=backend,
                    G_amp_ref_lst=G_amp_ref_lst,
                    x_ref_rotated=x_ref_rotated,
                    y_ref_rotated=y_ref_rotated
                )
            else:
                # c_row_opt, c_col_opt, min_error, b_opt_lst, ini = \
                #     planar_dirac_recon_alg_joint(G_lst, a_lst, K, M, N,
                #                                  tau_inter_x, tau_inter_y,
                #                                  noise_level, max_ini, stop_cri
                #                                  )
                c_row_opt, c_col_opt, min_error, b_opt_lst, ini = \
                    planar_dirac_recon_alg_joint_slow(G_lst, a_lst, K, M, N,
                                                      tau_inter_x, tau_inter_y,
                                                      noise_level, max_ini, stop_cri
                                                      )
                xk_recon_rotated, yk_recon_rotated = planar_extract_innovation(
                    a_lst, K, c_row_opt, c_col_opt, p_x_rotated, p_y_rotated,
                    tau_inter_x, tau_inter_y, partial_beamforming_func,
                    theano_func=theano_build_amp_func,
                    backend=backend
                )

            # rotate back
            xy_rotate_back = np.dot(rotate_mtx.T,
                                    np.vstack((
                                        xk_recon_rotated.flatten('F'),
                                        yk_recon_rotated.flatten('F')
                                    ))
                                    )
            xk_recon = xy_rotate_back[0, :]
            yk_recon = xy_rotate_back[1, :]

            # use the correctly identified colatitude and azimuth to reconstruct
            # the correct amplitudes
            error_loop, alphak_recon = \
                planar_compute_fitting_error_amp_beamforming(
                    a_lst, p_x_normalised, p_y_normalised, xk_recon, yk_recon,
                    beam_weights_func=partial_beamforming_func,
                    num_bands=num_bands,
                    theano_func=theano_build_amp_func,
                    backend=backend
                )[:2]

            if store_obj_val:
                obj_val_all.append(error_loop)

            if verbose:
                print('objective function value: {0:.3e}'.format(error_loop))

            if error_loop < min_error_all:
                min_error_all = error_loop
                xk_opt = xk_recon
                yk_opt = yk_recon
                alphak_opt = np.reshape(np.concatenate(alphak_recon),
                                        (-1, num_bands), order='F')

            xy_opt_rotated = np.dot(rotate_mtx,
                                    np.vstack((xk_opt.flatten('F'),
                                               yk_opt.flatten('F')))
                                    )
            x_opt_rotated = xy_opt_rotated[0, :]
            y_opt_rotated = xy_opt_rotated[1, :]

            if ref_sol_available:
                G_amp_ref_lst = [
                    planar_build_mtx_amp_beamforming(
                        p_x_rotated[:, :, :, band_count],
                        p_y_rotated[:, :, :, band_count],
                        x_opt_rotated[:K_ref], y_opt_rotated[:K_ref],
                        beam_weights_func=partial_beamforming_func,
                        theano_func=theano_build_amp_func,
                        backend=backend
                    )
                    for band_count in range(num_bands)
                ]

            if update_G and xk_opt.size > K_ref:
                G_lst = planar_update_G_beamforming(
                    x_opt_rotated[K_ref:], y_opt_rotated[K_ref:],
                    M, N, tau_inter_x, tau_inter_y,
                    p_x_rotated, p_y_rotated, G_lst,
                    beam_weights_func=partial_beamforming_func,
                    num_station=num_station, num_sti=num_sti, num_bands=num_bands,
                    theano_func=theano_build_amp_func, backend=backend
                )

        if verbose:
            print('======================================')

    if return_error:
        return xk_opt, yk_opt, np.reshape(alphak_opt, (-1, num_bands), order='F'), \
               min_error_all
    elif store_obj_val:
        return xk_opt, yk_opt, np.reshape(alphak_opt, (-1, num_bands), order='F'), \
               np.array(obj_val_all)
    else:
        return xk_opt, yk_opt, np.reshape(alphak_opt, (-1, num_bands), order='F')


def planar_recon_2d_dirac_joint_beamforming_ri(a, p_x, p_y, omega_bands, light_speed,
                                               K, tau_x, tau_y,
                                               M, N, tau_inter_x, tau_inter_y,
                                               noise_level=0, max_ini=20,
                                               stop_cri='max_iter', num_rotation=1,
                                               G_iter=1, plane_norm_vec=None,
                                               verbose=False, backend='cpu',
                                               theano_build_G_func=None,
                                               theano_build_amp_func=None,
                                               **kwargs):
    """
    INTERFACE for joint estimation of the Dirac deltas x and y locations in the UVW
    coordinates with beam-forming. Here, we have considered the problem in the tangential plane.
    :param a: the measured (complex-valued) visibilities in a 3D matrix form, where
              1) dimension 0: cross-correlation index
              2) dimension 1: different short time intervals (STI-s)
              3) dimension 2: different sub-bands
    :param p_x: a 4D matrix that contains antennas' x-coordinates, where
              1) dimension 0: antenna coordinate within each station
              2) dimension 1: station index
              3) dimension 2: short time interval (STI) index
              4) dimension 3: sub-band index
    :param p_y: a 4D matrix that contains antennas' y-coordinates
    :param omega_bands: mid-band (ANGULAR) frequencies [radian/sec]
    :param light_speed: speed of light
    :param K: number of point sources
    :param tau_x:
    :param tau_y:
    :param M: [M,N] the equivalence of "bandwidth" in time domain (because of duality)
    :param N: [M,N] the equivalence of "bandwidth" in time domain (because of duality)
    :param tau_inter_x: the Fourier domain interpolation step-size is 2 pi / tau_inter
    :param tau_inter_y: the Fourier domain interpolation step-size is 2 pi / tau_inter
    :param noise_level: noise level in the measured visibilities
    :param max_ini: maximum number of random initialisation used
    :param stop_cri: either 'mse' or 'max_iter'
    :param num_rotation: number of random rotation applied. if num_rotation == 1, then
            no rotation is applied.
    :param G_iter: number of iterations used to update the linear mapping from the FRI
            sequence to the visibility measurements based on the reconstructed Dirac
            locations. If G_iter == 1, then no update.
    :param plane_norm_vec: norm vector of the imaging plane. This vector is given by the
            center of the field-of-view (FoV) of the radio-telescope. If the argument is not
            None, then reconstructed sources that are not in the same imaging hemisphere will
            be reflected back to the imaging hemisphere.
    :param use_lu: whether to use LU decomposition to improved efficiency or not
    :param verbose: whether output intermediate results for debugging or not
    :param backend: either 'cpu' or 'gpu'
    :param kwargs: include
            x_ref: reference Dirac horizontal locations in the UVW coordinate, e.g., previous reconstruction
            y_ref: reference Dirac vertical locations in the UVW coordinate, e.g., previous reconstruction
    :return:
    """
    if 'x_ref' in kwargs and 'y_ref' in kwargs:
        ref_sol_available = True
        x_ref = kwargs['x_ref']
        y_ref = kwargs['y_ref']
        xy_ref = np.vstack((np.array(x_ref).flatten('F'), np.array(y_ref).flatten('F')))
        K_ref = x_ref.size
    else:
        ref_sol_available = False
        K_ref = 0

    assert len(a.shape) == 3
    # whether update the linear mapping or not
    update_G = (G_iter != 1)

    num_bands = a.shape[2]  # number of bands considered
    num_sti = a.shape[1]  # number of short time intervals
    num_station = p_x.shape[1]  # number of stations

    assert a.shape[0] == num_station * (num_station - 1)
    assert a.shape[2] == np.array(omega_bands).size

    '''verify input parameters'''
    # interpolation points cannot be too far apart
    assert tau_inter_x >= tau_x and tau_inter_y >= tau_y

    # M * tau is an odd number
    assert M * tau_inter_x % 2 == 1 and N * tau_inter_y % 2 == 1

    # G is a tall matrix
    assert M * tau_inter_x * N * tau_inter_y <= num_station * (num_station - 1) * num_sti * num_bands

    tau_inter_x = float(tau_inter_x)
    tau_inter_y = float(tau_inter_y)

    # extension matrix, which maps the first real-valued representation in the first
    # half interval of a Hermitian symmetric matrix/vector to its full range values.
    m_limit = int(np.floor(M * tau_inter_x // 2))
    n_limit = int(np.floor(N * tau_inter_y // 2))
    m_len = 2 * m_limit + 1
    n_len = 2 * n_limit + 1
    expand_b_mtx_real, expand_b_mtx_imag = hermitian_expan_mtx(m_len * n_len)
    expand_b_mtx = linalg.block_diag(expand_b_mtx_real, expand_b_mtx_imag)

    if plane_norm_vec is not None:
        plane_norm_vec0, plane_norm_vec1, plane_norm_vec2 = plane_norm_vec
    else:
        plane_norm_vec0, plane_norm_vec1, plane_norm_vec2 = 0, 0, 1

    norm_factor = np.reshape(light_speed / omega_bands, (1, 1, 1, -1), order='F')
    # normalised antenna coordinates
    p_x_normalised = np.reshape(p_x, (-1, num_station, num_sti, 1),
                                order='F') / norm_factor
    p_y_normalised = np.reshape(p_y, (-1, num_station, num_sti, 1),
                                order='F') / norm_factor

    # reshape a_ri to a 2D matrix such that:
    # 1) dimension0: concatenated 'a' for different STI-s but the SAME sub-band frequency
    # 2) dimension1: different subband frequencies
    a = np.reshape(a, (-1, num_bands), order='F')
    a_cpx_lst = [a[:, band_count] for band_count in range(num_bands)]

    # # we use the real-valued representation of the measurements
    a_ri = np.row_stack((a.real, a.imag))
    a_ri_lst = [a_ri[:, band_count] for band_count in range(num_bands)]  # the list representation

    min_error_all = float('inf')

    for rand_rotate in np.linspace(0, np.pi, num=num_rotation, endpoint=False):
        # apply a random rotation
        if num_rotation == 0:  # <= i.e., no rotation
            rotate_angle = 0
        else:
            rotate_angle = rand_rotate + np.pi / num_rotation * np.random.rand()

        # build rotation matrix
        rotate_mtx = np.array([[np.cos(rotate_angle), -np.sin(rotate_angle)],
                               [np.sin(rotate_angle), np.cos(rotate_angle)]])

        # rotate antenna steering vector
        p_rotated = np.dot(rotate_mtx,
                           np.vstack((
                               p_x_normalised.flatten('F'),
                               p_y_normalised.flatten('F')
                           ))
                           )
        p_x_rotated = np.reshape(p_rotated[0, :], p_x_normalised.shape, order='F')
        p_y_rotated = np.reshape(p_rotated[1, :], p_y_normalised.shape, order='F')

        x0_rotated = np.cos(rotate_angle) * plane_norm_vec0 - \
                     np.sin(rotate_angle) * plane_norm_vec1
        y0_rotated = np.sin(rotate_angle) * plane_norm_vec0 + \
                     np.cos(rotate_angle) * plane_norm_vec1

        def partial_beamforming_func(baseline_x, baseline_y, x0=x0_rotated, y0=y0_rotated):
            return planar_beamforming_func(baseline_x, baseline_y,
                                           strategy='matched', x0=x0, y0=y0)

        tic = time.time()
        # linear transformation matrix that maps uniform samples of sinusoids to visibilities
        mtx_freq2visibility = planar_mtx_freq2visibility_beamforming(
            p_x_rotated, p_y_rotated, M, N, tau_inter_x, tau_inter_y,
            partial_beamforming_func, num_station, num_sti, num_bands,
            backend=backend, theano_func=theano_build_G_func
        )

        G_ri_lst = planar_mtx_fri2visibility_beamforming(
            mtx_freq2visibility, real_value=True, symb=True, expand_b_mtx=expand_b_mtx)
        toc = time.time()
        print('time takes to build G: {0:.2f}sec\n'.format(toc - tic))

        for count_G in range(G_iter):

            K_alg = min(determine_max_coef_sz(2 * n_limit + 1, 2 * m_limit + 1), 400)

            if count_G == 0:
                '''
                Extract the measurements and the lines in G such that the baselines are
                included within on period of the periodic sinc interpolation.
                Here we use the controid of a station as a crude approximation to determine
                whether to include a given visibility measurement in the first estimation or not.
                '''
                extract_indicator_lst = \
                    planar_build_inrange_sel_mtx(p_x_rotated, p_y_rotated,
                                                 num_station, num_bands, num_sti,
                                                 M * np.pi, N * np.pi)
                G_ri_lst_count0 = \
                    [
                        np.dot(
                            cpx_mtx2real(
                                mtx_freq2visibility[band_count][extract_indicator_lst[band_count], :]
                            ), expand_b_mtx)
                        for band_count in range(num_bands)
                    ]
                del mtx_freq2visibility
                a_cpx_count0_lst = [
                    a[extract_indicator_lst[band_count], band_count]
                    for band_count in range(num_bands)
                ]
                a_ri_count0_lst = [
                    np.concatenate((
                        a_cpx_count0_lst[band_count].squeeze().real,
                        a_cpx_count0_lst[band_count].squeeze().imag
                    ))
                    for band_count in range(num_bands)]

                # make sure the subset of measurements are still enough for the algorithm,
                # i.e. G is a tall matrix
                assert M * tau_inter_x * N * tau_inter_y <= np.concatenate(a_ri_count0_lst).size

                if ref_sol_available:
                    xy_ref_rotated = np.dot(rotate_mtx, xy_ref)
                    x_ref_rotated = xy_ref_rotated[0, :]
                    y_ref_rotated = xy_ref_rotated[1, :]

                    G_amp_ref_cpx_lst = [
                        planar_build_mtx_amp_beamforming(
                            p_x_rotated[:, :, :, band_count],
                            p_y_rotated[:, :, :, band_count],
                            x_ref_rotated, y_ref_rotated,
                            partial_beamforming_func,
                            theano_func=theano_build_amp_func,
                            backend=backend
                        )
                        for band_count in range(num_bands)
                    ]
                    G_amp_ref_ri_lst = [
                        np.vstack((
                            G_amp_ref_cpx_lst[band_count].real,
                            G_amp_ref_cpx_lst[band_count].imag
                        ))
                        for band_count in range(num_bands)
                    ]

                    G_amp_ref_cpx_lst_count0 = [
                        G_amp_ref_cpx_lst[band_count][extract_indicator_lst[band_count], :]
                        for band_count in range(num_bands)
                    ]
                    G_amp_ref_ri_lst_count0 = [
                        np.vstack((
                            G_amp_ref_cpx_lst_count0[band_count].real,
                            G_amp_ref_cpx_lst_count0[band_count].imag
                        ))
                        for band_count in range(num_bands)
                    ]

                    c_row_opt, c_col_opt, min_error, b_opt_lst, ini = \
                        planar_dirac_recon_alg_joint_ri(
                            G_ri_lst_count0, a_ri_count0_lst, K_alg, M, N,
                            tau_inter_x, tau_inter_y,
                            noise_level, max_ini, stop_cri,
                            G_amp_ref_lst=G_amp_ref_ri_lst_count0
                        )

                    xk_recon_rotated, yk_recon_rotated = planar_extract_innovation(
                        a_cpx_count0_lst, K, c_row_opt, c_col_opt,
                        p_x_rotated, p_y_rotated,
                        tau_inter_x, tau_inter_y,
                        partial_beamforming_func,
                        theano_func=theano_build_amp_func,
                        backend=backend,
                        ref_sol_available=True,
                        G_amp_ref_lst=G_amp_ref_cpx_lst_count0,
                        x_ref_rotated=x_ref_rotated,
                        y_ref_rotated=y_ref_rotated,
                        extract_indicator_lst=extract_indicator_lst
                    )
                    del G_amp_ref_cpx_lst_count0, G_amp_ref_ri_lst_count0

                else:
                    c_row_opt, c_col_opt = \
                        planar_dirac_recon_alg_joint_ri(
                            G_ri_lst_count0, a_ri_count0_lst, K_alg, M, N,
                            tau_inter_x, tau_inter_y,
                            noise_level, max_ini, stop_cri
                        )[:2]

                    xk_recon_rotated, yk_recon_rotated = planar_extract_innovation(
                        a_cpx_count0_lst, K, c_row_opt, c_col_opt,
                        p_x_rotated, p_y_rotated,
                        tau_inter_x, tau_inter_y,
                        partial_beamforming_func,
                        theano_func=theano_build_amp_func,
                        backend=backend,
                        extract_indicator_lst=extract_indicator_lst
                    )

                G_ri_lst = planar_update_G_ri_beamforming(
                    xk_recon_rotated[K_ref:], yk_recon_rotated[K_ref:],
                    M, N, tau_inter_x, tau_inter_y,
                    p_x_rotated, p_y_rotated, G_ri_lst,
                    beam_weights_func=partial_beamforming_func,
                    num_station=num_station, num_sti=num_sti, num_bands=num_bands
                )

            if ref_sol_available:
                c_row_opt, c_col_opt = \
                    planar_dirac_recon_alg_joint_ri(G_ri_lst, a_ri_lst, K, M, N,
                                                    tau_inter_x, tau_inter_y,
                                                    noise_level, max_ini, stop_cri,
                                                    G_amp_ref_lst=G_amp_ref_ri_lst)[:2]

                xk_recon_rotated, yk_recon_rotated = planar_extract_innovation(
                    a_cpx_lst, K, c_row_opt, c_col_opt,
                    p_x_rotated, p_y_rotated,
                    tau_inter_x, tau_inter_y,
                    partial_beamforming_func,
                    theano_func=theano_build_amp_func,
                    backend=backend,
                    G_amp_ref_lst=G_amp_ref_cpx_lst,
                    x_ref_rotated=x_ref_rotated,
                    y_ref_rotated=y_ref_rotated
                )
            else:
                c_row_opt, c_col_opt, min_error, b_opt_lst, ini = \
                    planar_dirac_recon_alg_joint_ri(G_ri_lst, a_ri_lst, K, M, N,
                                                    tau_inter_x, tau_inter_y,
                                                    noise_level, max_ini, stop_cri)

                xk_recon_rotated, yk_recon_rotated = planar_extract_innovation(
                    a_cpx_lst, K, c_row_opt, c_col_opt,
                    p_x_rotated, p_y_rotated,
                    tau_inter_x, tau_inter_y,
                    partial_beamforming_func,
                    theano_func=theano_build_amp_func,
                    backend=backend
                )

            # rotate back
            xy_rotate_back = np.dot(rotate_mtx.T,
                                    np.vstack((
                                        xk_recon_rotated.flatten('F'),
                                        yk_recon_rotated.flatten('F')
                                    ))
                                    )
            xk_recon = xy_rotate_back[0, :]
            yk_recon = xy_rotate_back[1, :]

            # use the correctly identified colatitude and azimuth to reconstruct
            # the correct amplitudes
            error_loop, alphak_recon = \
                planar_compute_fitting_error_amp_beamforming_ri(
                    a_ri_lst, p_x_normalised, p_y_normalised, xk_recon, yk_recon,
                    beam_weights_func=partial_beamforming_func,
                    num_station=num_station, num_bands=num_bands, num_sti=num_sti,
                    backend=backend
                )

            if verbose:
                print('objective function value: {0:.3e}'.format(error_loop))

            if error_loop < min_error_all:
                min_error_all = error_loop
                xk_opt = xk_recon
                yk_opt = yk_recon
                alphak_opt = np.reshape(np.concatenate(alphak_recon), (-1, num_bands), order='F')

            xy_opt_rotated = np.dot(rotate_mtx,
                                    np.vstack((xk_opt.flatten('F'),
                                               yk_opt.flatten('F')))
                                    )
            x_opt_rotated = xy_opt_rotated[0, :]
            y_opt_rotated = xy_opt_rotated[1, :]

            if ref_sol_available:
                G_amp_ref_ri_lst = [
                    planar_build_mtx_amp_ri_beamforming(
                        p_x_rotated[:, :, :, band_count],
                        p_y_rotated[:, :, :, band_count],
                        x_opt_rotated[:K_ref], y_opt_rotated[:K_ref],
                        beam_weights_func=partial_beamforming_func,
                        num_station=num_station, num_sti=num_sti
                    )
                    for band_count in range(num_bands)
                ]

            if update_G and xk_opt.size > K_ref:
                G_ri_lst = planar_update_G_ri_beamforming(
                    x_opt_rotated[K_ref:], y_opt_rotated[K_ref:],
                    M, N, tau_inter_x, tau_inter_y,
                    p_x_rotated, p_y_rotated, G_ri_lst,
                    beam_weights_func=partial_beamforming_func,
                    num_station=num_station, num_sti=num_sti, num_bands=num_bands
                )

        if verbose:
            print('======================================')

    return xk_opt, yk_opt, np.reshape(alphak_opt, (-1, num_bands), order='F')


def planar_dirac_recon_alg_joint(G0, a_lst, K, M, N, tau_inter_x, tau_inter_y,
                                 noise_level=0, max_ini=50, stop_cri='mse', **kwargs):
    """
    reconstruct 2D Dirac deltas with joint annihilation along both horizontal and vertical
    directions.
    :param G0: a list of the linear mapping from the FRI sequence to the visibilities for all subbands
    :param a: visibility measurements
    :param K: number of point sources
    :param M: [M,N] the equivalence of "bandwidth" in time domain (because of duality)
    :param N: [M,N] the equivalence of "bandwidth" in time domain (because of duality)
    :param tau_inter_x: the Fourier domain interpolation step-size is 2 pi / tau_inter
    :param tau_inter_y: the Fourier domain interpolation step-size is 2 pi / tau_inter
    :param noise_level: level of noise
    :param max_ini: maximum number of random initialisations
    :param stop_cri: stopping criteria, either 'mse' or 'max_iter'
    :param kwargs: include
                G_amp_ref_lst: a linear mapping from amplitudes to visibilities based on a
                        reference solution, e.g., previously reconstructed Dirac locations.
    :return:
    """
    compute_mse = (stop_cri == 'mse')
    update_G = False
    mtx_extract_b = None

    # assert len(a.shape) == 2
    num_bands = len(a_lst)  # number of subbands

    # determine the size of the 2D annihilating filters
    sz_coef_row0 = np.int(np.floor(np.sqrt(K + 1)))
    sz_coef_row1 = np.int(np.ceil((K + 1) / sz_coef_row0))
    assert sz_coef_row0 * sz_coef_row1 >= K + 1

    if sz_coef_row0 == sz_coef_row1:
        sz_coef_col0, sz_coef_col1 = sz_coef_row0 + 1, sz_coef_row1
    else:
        sz_coef_col1, sz_coef_col0 = sz_coef_row0, sz_coef_row1

    assert sz_coef_col0 >= sz_coef_row0 and sz_coef_row1 >= sz_coef_col1

    # FRI sequence size
    m_limit = np.int(np.floor(M * tau_inter_x // 2))
    n_limit = np.int(np.floor(N * tau_inter_y // 2))
    sz_fri1 = 2 * m_limit + 1
    sz_fri0 = 2 * n_limit + 1

    # number of coefficients for the annihilating filter
    num_coef_row = sz_coef_row0 * sz_coef_row1
    num_coef_col = sz_coef_col0 * sz_coef_col1

    # size of various matrices
    sz_S0 = num_coef_row + num_coef_col - 2 * (K + 1)

    # sz_Tb1 = num_coef_row + num_coef_col

    sz_R1 = sz_fri0 * sz_fri1

    sz_coef = num_coef_row + num_coef_col

    if 'G_amp_ref_lst' in kwargs:
        update_G = True
        G_amp_ref_lst = kwargs['G_amp_ref_lst']
        K_ref = G_amp_ref_lst[0].shape[1]
        sz_G1 = sz_R1 + K_ref
        mtx_extract_b = np.eye(sz_G1)[K_ref:, :]
        G_lst = [
            np.column_stack((
                G_amp_ref_lst[band_count], G0[band_count]
            ))
            for band_count in range(num_bands)
        ]
    else:
        G_lst = G0
        sz_G1 = sz_R1

    R_test = R_mtx_joint(np.random.randn(sz_coef_row0, sz_coef_row1),
                         np.random.randn(sz_coef_col0, sz_coef_col1),
                         sz_fri0, sz_fri1, mtx_extract_b)
    s_test = linalg.svd(R_test, compute_uv=False)
    sz_Tb0_effective = min(R_test.shape[0], R_test.shape[1]) - \
                       np.where(np.abs(s_test) < 1e-12)[0].size

    sz_R0_effective = sz_Tb0_effective

    # pre-compute a few matrices / vectors
    Gt_a_lst = []
    GtG_inv_lst = []
    Tbeta0_lst = []

    if 'beta' in kwargs:
        beta_lst = kwargs['beta']
        compute_beta = False
    else:
        beta_lst = []
        compute_beta = True

    for band_count in range(num_bands):
        G_loop = G_lst[band_count]
        a_loop = a_lst[band_count]

        GtG_inv_loop = linalg.solve(np.dot(G_loop.conj().T, G_loop),
                                    np.eye(sz_G1, dtype=float),
                                    check_finite=False)
        Gt_a_loop = np.dot(G_loop.conj().T, a_loop)

        if compute_beta:
            beta_loop = np.dot(GtG_inv_loop, Gt_a_loop)
            beta_lst.append(beta_loop)
        else:
            beta_loop = beta_lst[band_count]

        if update_G:
            beta_loop = np.dot(mtx_extract_b, beta_loop)

        Tbeta0_loop = linalg.block_diag(
            convmtx2_valid(np.reshape(beta_loop, (sz_fri0, sz_fri1), order='F'),
                           sz_coef_row0, sz_coef_row1),
            convmtx2_valid(np.reshape(beta_loop, (sz_fri0, sz_fri1), order='F'),
                           sz_coef_col0, sz_coef_col1)
        )

        # append to list
        Gt_a_lst.append(Gt_a_loop)
        GtG_inv_lst.append(GtG_inv_loop)
        Tbeta0_lst.append(Tbeta0_loop)

    # initialise loop
    max_iter = 30
    min_error = float('inf')

    rhs = np.concatenate((np.zeros(sz_coef + sz_S0), np.array([1, 1])))

    for ini in range(max_ini):
        # select a subset of size (K + 1) of these coefficients
        # here the indices corresponds to the part of coefficients that are ZERO
        S = planar_sel_coef_subset(num_coef_row, num_coef_col, K)
        S_H = S.conj().T

        # randomly initialised the annihilating filter coefficients
        c_row = np.random.randn(sz_coef_row0, sz_coef_row1) + \
                1j * np.random.randn(sz_coef_row0, sz_coef_row1)  # for row annihilation
        c_col = np.random.randn(sz_coef_col0, sz_coef_col1) + \
                1j * np.random.randn(sz_coef_col0, sz_coef_col1)  # for col annihilation
        c0 = linalg.block_diag(c_row.flatten('F')[:, np.newaxis],
                               c_col.flatten('F')[:, np.newaxis])
        c0_H = c0.conj().T

        R_loop = R_mtx_joint(c_row, c_col, sz_fri0, sz_fri1, mtx_extract_b)
        Q_H = linalg.qr(R_loop, mode='economic', check_finite=False,
                        pivoting=True)[0][:, :sz_R0_effective].conj().T

        R_loop = np.dot(Q_H, R_loop)

        # last row in mtx_loop
        mtx_loop_last_row = np.hstack((
            np.vstack((S, c0_H)),
            np.zeros((sz_S0 + 2, sz_S0 + 2), dtype=float)
        ))

        for inner in range(max_iter):
            if inner == 0:
                mtx_loop = np.vstack((
                    np.hstack((
                        planar_compute_mtx_obj_multiband(
                            GtG_inv_lst, Tbeta0_lst, R_loop, Q_H, num_bands, sz_coef
                        ),
                        S_H, c0
                    )),
                    mtx_loop_last_row
                ))
            else:
                mtx_loop[:sz_coef, :sz_coef] = mtx_loop_upper_left

            try:
                c = linalg.solve(mtx_loop, rhs, check_finite=False)[:sz_coef]
            except linalg.LinAlgError:
                break

            c_row = np.reshape(c[:num_coef_row], (sz_coef_row0, sz_coef_row1), order='F')
            c_col = np.reshape(c[num_coef_row:], (sz_coef_col0, sz_coef_col1), order='F')

            # build new R matrix
            R_loop = R_mtx_joint(c_row, c_col, sz_fri0, sz_fri1, mtx_extract_b)
            Q_H = linalg.qr(R_loop, mode='economic', check_finite=False,
                            pivoting=True)[0][:, :sz_R0_effective].conj().T

            R_loop = np.dot(Q_H, R_loop)

            # compute the fitting error
            error_loop, mtx_loop_upper_left = \
                planar_compute_obj_val(GtG_inv_lst, Tbeta0_lst, R_loop, Q_H, c, num_bands, sz_coef)

            if error_loop < min_error:
                min_error = error_loop
                # NOTE THAT THE FIRST SEGEMENT CORRESPONDS TO THE THE AMPLITUDE OF THE PREVIOUSLY
                # RECONSTRUCTED DIRAC WHEN UPDATE_G = TRUE
                R_opt = R_loop
                c_row_opt = c_row
                c_col_opt = c_col

            if compute_mse and min_error < noise_level:
                break

        if compute_mse and min_error < noise_level:
            break

    # only compute b explicitly at the end
    b_opt_lst = planar_compute_b(G_lst, GtG_inv_lst, beta_lst, R_opt, num_bands, a_lst)[2]

    return c_row_opt, c_col_opt, min_error, b_opt_lst, ini


def planar_dirac_recon_alg_joint_slow(G0, a_lst, K, M, N, tau_inter_x, tau_inter_y,
                                      noise_level=0, max_ini=50, stop_cri='mse', **kwargs):
    """
    reconstruct 2D Dirac deltas with joint annihilation along both horizontal and vertical
    directions.
    :param G0: a list of the linear mapping from the FRI sequence to the visibilities for all subbands
    :param a: visibility measurements
    :param K: number of point sources
    :param M: [M,N] the equivalence of "bandwidth" in time domain (because of duality)
    :param N: [M,N] the equivalence of "bandwidth" in time domain (because of duality)
    :param tau_inter_x: the Fourier domain interpolation step-size is 2 pi / tau_inter
    :param tau_inter_y: the Fourier domain interpolation step-size is 2 pi / tau_inter
    :param noise_level: level of noise
    :param max_ini: maximum number of random initialisations
    :param stop_cri: stopping criteria, either 'mse' or 'max_iter'
    :param kwargs: include
                G_amp_ref_lst: a linear mapping from amplitudes to visibilities based on a
                        reference solution, e.g., previously reconstructed Dirac locations.
    :return:
    """
    compute_mse = (stop_cri == 'mse')
    update_G = False
    mtx_extract_b = None

    # assert len(a.shape) == 2
    num_bands = len(a_lst)  # number of subbands

    # determine the size of the 2D annihilating filters
    sz_coef_row0 = sz_coef_row1 = sz_coef_col0 = sz_coef_col1 = \
        np.int(np.ceil(np.sqrt(K + 2)))

    # FRI sequence size
    m_limit = np.int(np.floor(M * tau_inter_x // 2))
    n_limit = np.int(np.floor(N * tau_inter_y // 2))
    sz_fri1 = 2 * m_limit + 1
    sz_fri0 = 2 * n_limit + 1

    # number of coefficients for the annihilating filter
    num_coef_row = sz_coef_row0 * sz_coef_row1
    num_coef_col = sz_coef_col0 * sz_coef_col1

    # size of various matrices
    sz_S0 = num_coef_row + num_coef_col - 2 * (K + 2)

    # sz_Tb1 = num_coef_row + num_coef_col

    sz_R1 = sz_fri0 * sz_fri1

    sz_coef = num_coef_row + num_coef_col

    if 'G_amp_ref_lst' in kwargs:
        update_G = True
        G_amp_ref_lst = kwargs['G_amp_ref_lst']
        K_ref = G_amp_ref_lst[0].shape[1]
        sz_G1 = sz_R1 + K_ref
        mtx_extract_b = np.eye(sz_G1)[K_ref:, :]
        G_lst = [
            np.column_stack((
                G_amp_ref_lst[band_count], G0[band_count]
            ))
            for band_count in range(num_bands)
        ]
    else:
        G_lst = G0

    try:
        R_test = R_mtx_joint(np.random.randn(sz_coef_row0, sz_coef_row1),
                             np.random.randn(sz_coef_col0, sz_coef_col1),
                             sz_fri0, sz_fri1, mtx_extract_b)
        s_test = linalg.svd(R_test, compute_uv=False)
    except ValueError:
        R_test = R_mtx_joint(np.random.randn(sz_coef_row0, sz_coef_row1),
                             np.random.randn(sz_coef_col0, sz_coef_col1),
                             sz_fri0, sz_fri1, mtx_extract_b)
        s_test = linalg.svd(R_test, compute_uv=False)
    sz_Tb0_effective = min(R_test.shape[0], R_test.shape[1]) - \
                       np.where(np.abs(s_test) < 1e-12)[0].size

    sz_R0_effective = sz_Tb0_effective

    # pre-compute a few matrices / vectors
    Gt_a_lst = []
    lu_GtG_lst = []
    Tbeta0_lst = []

    if 'beta' in kwargs:
        beta_lst = kwargs['beta']
        compute_beta = False
    else:
        beta_lst = []
        compute_beta = True

    for band_count in range(num_bands):
        G_loop = G_lst[band_count]
        a_loop = a_lst[band_count]

        lu_GtG_loop = linalg.lu_factor(np.dot(G_loop.conj().T, G_loop),
                                       check_finite=False)
        Gt_a_loop = np.dot(G_loop.conj().T, a_loop)

        if compute_beta:
            beta_loop = linalg.lu_solve(lu_GtG_loop, Gt_a_loop, check_finite=False)
            beta_lst.append(beta_loop)
        else:
            beta_loop = beta_lst[band_count]

        if update_G:
            beta_loop = np.dot(mtx_extract_b, beta_loop)

        Tbeta0_loop = linalg.block_diag(
            convmtx2_valid(np.reshape(beta_loop, (sz_fri0, sz_fri1), order='F'),
                           sz_coef_row0, sz_coef_row1),
            convmtx2_valid(np.reshape(beta_loop, (sz_fri0, sz_fri1), order='F'),
                           sz_coef_col0, sz_coef_col1)
        )

        # append to list
        Gt_a_lst.append(Gt_a_loop)
        lu_GtG_lst.append(lu_GtG_loop)
        Tbeta0_lst.append(Tbeta0_loop)

    # initialise loop
    max_iter = 30
    min_error = float('inf')

    rhs = np.concatenate((np.zeros(sz_coef + sz_S0), np.array([1, 1])))

    for ini in range(max_ini):
        # select a subset of size (K + 1) of these coefficients
        # here the indices corresponds to the part of coefficients that are ZERO
        S = planar_sel_coef_subset(num_coef_row, num_coef_col, K)
        S_H = S.conj().T

        # randomly initialised the annihilating filter coefficients
        c_row = np.random.randn(sz_coef_row0, sz_coef_row1) + \
                1j * np.random.randn(sz_coef_row0, sz_coef_row1)  # for row annihilation
        c_col = np.random.randn(sz_coef_col0, sz_coef_col1) + \
                1j * np.random.randn(sz_coef_col0, sz_coef_col1)  # for col annihilation
        c0 = linalg.block_diag(c_row.flatten('F')[:, np.newaxis],
                               c_col.flatten('F')[:, np.newaxis])
        c0_H = c0.conj().T

        R_loop = R_mtx_joint(c_row, c_col, sz_fri0, sz_fri1, mtx_extract_b)
        # Q_H = linalg.qr(R_loop, mode='economic', check_finite=False,
        #                 pivoting=True)[0][:, :sz_R0_effective].conj().T
        # R_loop = np.dot(Q_H, R_loop)
        Q_H = linalg.qr(R_loop, mode='economic',
                        check_finite=False)[0][:, :sz_R0_effective].conj().T
        R_loop = np.dot(Q_H, R_loop)

        # last row in mtx_loop
        mtx_loop_last_row = np.hstack((
            np.vstack((S, c0_H)),
            np.zeros((sz_S0 + 2, sz_S0 + 2), dtype=float)
        ))

        lu_lst = None

        for inner in range(max_iter):
            if inner == 0:
                mtx_loop = np.vstack((
                    np.hstack((
                        planar_compute_mtx_obj_multiband_slow(
                            lu_GtG_lst, Tbeta0_lst, R_loop, Q_H, num_bands, sz_coef
                        ),
                        S_H, c0
                    )),
                    mtx_loop_last_row
                ))
            else:
                mtx_loop[:sz_coef, :sz_coef] = \
                    planar_compute_mtx_obj_multiband_slow(
                        lu_GtG_lst, Tbeta0_lst, R_loop, Q_H, num_bands, sz_coef, lu_lst
                    )

            try:
                c = linalg.solve(mtx_loop, rhs, check_finite=False)[:sz_coef]
            except linalg.LinAlgError:
                break

            c_row = np.reshape(c[:num_coef_row], (sz_coef_row0, sz_coef_row1), order='F')
            c_col = np.reshape(c[num_coef_row:], (sz_coef_col0, sz_coef_col1), order='F')

            # build new R matrix
            R_loop = R_mtx_joint(c_row, c_col, sz_fri0, sz_fri1, mtx_extract_b)
            # Q_H = linalg.qr(R_loop, mode='economic', check_finite=False,
            #                 pivoting=True)[0][:, :sz_R0_effective].conj().T
            # R_loop = np.dot(Q_H, R_loop)
            Q_H = linalg.qr(R_loop, mode='economic',
                            check_finite=False)[0][:, :sz_R0_effective].conj().T
            R_loop = np.dot(Q_H, R_loop)

            # compute the FRI sequence b and / or fitting error
            error_loop, lu_lst, b_recon_lst = planar_compute_b_slow(
                G_lst, lu_GtG_lst, beta_lst, R_loop, num_bands, a_lst
            )

            if error_loop < min_error:
                min_error = error_loop
                # NOTE THAT THE FIRST SEGEMENT CORRESPONDS TO THE THE AMPLITUDE OF THE PREVIOUSLY
                # RECONSTRUCTED DIRAC WHEN UPDATE_G = TRUE
                b_opt_lst = b_recon_lst
                c_row_opt = c_row
                c_col_opt = c_col

            if compute_mse and min_error < noise_level:
                break

        if compute_mse and min_error < noise_level:
            break

    return c_row_opt, c_col_opt, min_error, b_opt_lst, ini


def planar_dirac_recon_alg_joint_ri(G0_ri, a_ri_lst, K, M, N, tau_inter_x, tau_inter_y,
                                    noise_level=0, max_ini=50, stop_cri='mse',
                                    **kwargs):
    """
    reconstruct 2D Dirac deltas with joint annihilation along both horizontal and vertical
    directions.
    :param G0_ri: a list of the linear mapping from the FRI sequence to the visibilities for all subbands
    :param a_ri_lst: visibility measurements
    :param K: number of point sources
    :param M: [M,N] the equivalence of "bandwidth" in time domain (because of duality)
    :param N: [M,N] the equivalence of "bandwidth" in time domain (because of duality)
    :param tau_inter_x: the Fourier domain interpolation step-size is 2 pi / tau_inter
    :param tau_inter_y: the Fourier domain interpolation step-size is 2 pi / tau_inter
    :param noise_level: level of noise
    :param max_ini: maximum number of random initialisations
    :param stop_cri: stopping criteria, either 'mse' or 'max_iter'
    :param kwargs: include
                G_amp_ref_lst: a linear mapping from amplitudes to visibilities based on a
                        reference solution, e.g., previously reconstructed Dirac locations.
    :return:
    """
    compute_mse = (stop_cri == 'mse')
    update_G = False
    mtx_extract_b = None

    num_bands = len(a_ri_lst)  # number of subbands

    # determine the size of the 2D annihilating filters
    sz_coef_row0 = int(np.floor(np.sqrt(K + 1)))
    sz_coef_row1 = int(np.ceil((K + 1) / sz_coef_row0))
    assert sz_coef_row0 * sz_coef_row1 >= K + 1

    if sz_coef_row0 == sz_coef_row1:
        sz_coef_col0, sz_coef_col1 = sz_coef_row0 + 1, sz_coef_row1
    else:
        sz_coef_col1, sz_coef_col0 = sz_coef_row0, sz_coef_row1

    # deal with the specific case when
    # 1) the total coefficient block has even number of entries and
    # 2) the non-zero annihilating filter coefficients (K + 1) is an odd number
    if (sz_coef_row0 * sz_coef_row1) % 2 == 0 and (K + 1) % 2 == 1:
        if sz_coef_row0 % 2 == 0:
            sz_coef_row0 += 1

        if sz_coef_row1 % 2 == 0:
            sz_coef_row1 += 1

    if (sz_coef_col0 * sz_coef_col1) % 2 == 0 and (K + 1) % 2 == 1:
        if sz_coef_col0 % 2 == 0:
            sz_coef_col0 += 1

        if sz_coef_col1 % 2 == 0:
            sz_coef_col1 += 1

    assert sz_coef_col0 >= sz_coef_row0 and sz_coef_row1 >= sz_coef_col1

    # FRI sequence size
    m_limit = int(np.floor(M * tau_inter_x // 2))
    n_limit = int(np.floor(N * tau_inter_y // 2))
    sz_fri1 = 2 * m_limit + 1
    sz_fri0 = 2 * n_limit + 1

    # extension matrix, which maps the first real-valued representation in the first
    # half interval of a Hermitian symmetric matrix/vector to its full range values.
    expand_b_real, expand_b_imag = hermitian_expan_mtx(sz_fri0 * sz_fri1)
    expand_b_mtx = linalg.block_diag(expand_b_real, expand_b_imag)

    # number of coefficients for the annihilating filter
    num_coef_row = sz_coef_row0 * sz_coef_row1
    num_coef_col = sz_coef_col0 * sz_coef_col1

    # similarly for the annihilation coefficients
    expand_c_row_real, expand_c_row_imag = hermitian_expan_mtx(num_coef_row)
    expand_c_row_mtx = linalg.block_diag(expand_c_row_real, expand_c_row_imag)

    expand_c_col_real, expand_c_col_imag = hermitian_expan_mtx(num_coef_col)
    expand_c_col_mtx = linalg.block_diag(expand_c_col_real, expand_c_col_imag)

    # size of various matrices
    sz_S0 = (num_coef_row + num_coef_col - 2 * (K + 1))

    # sz_Tb1 = num_coef_row + num_coef_col

    sz_fri_blk = sz_fri0 * sz_fri1

    sz_R1 = sz_fri0 * sz_fri1

    sz_coef = num_coef_row + num_coef_col

    # shrink the output size to half as both the annihilating filter coeffiicnets and
    # the uniform samples of sinusoids are Hermitian symmetric
    mtx_shrink_row = output_shrink((sz_fri0 - sz_coef_row0 + 1) * (sz_fri1 - sz_coef_row1 + 1))
    mtx_shrink_col = output_shrink((sz_fri0 - sz_coef_col0 + 1) * (sz_fri1 - sz_coef_col1 + 1))
    mtx_shrink_S_row = output_shrink((num_coef_row - (K + 1)))
    mtx_shrink_S_col = output_shrink((num_coef_col - (K + 1)))

    if 'G_amp_ref_lst' in kwargs:
        update_G = True
        G_amp_ref_lst = kwargs['G_amp_ref_lst']
        K_ref = G_amp_ref_lst[0].shape[1]
        sz_G1 = sz_R1 + K_ref
        mtx_extract_b = np.eye(sz_G1)[K_ref:, :]
        G_ri_lst = [
            np.column_stack((
                G_amp_ref_lst[band_count], G0_ri[band_count]
            ))
            for band_count in range(num_bands)
        ]
    else:
        G_ri_lst = G0_ri

    R_test = R_mtx_joint_ri_half(
        np.random.randn(sz_coef_row0, sz_coef_row1) +
        1j * np.random.randn(sz_coef_row0, sz_coef_row1),
        np.random.randn(sz_coef_col0, sz_coef_col1) +
        1j * np.random.randn(sz_coef_col0, sz_coef_col1),
        sz_fri0, sz_fri1,
        expansion_mtx=expand_b_mtx,
        mtx_shrink_row=mtx_shrink_row,
        mtx_shrink_col=mtx_shrink_col,
        mtx_extract_b=mtx_extract_b
    )
    s_test = linalg.svd(R_test, compute_uv=False)
    sz_Tb0_effective = min(R_test.shape[0], R_test.shape[1]) - \
                       np.where(np.abs(s_test) < 1e-12)[0].size

    sz_R0_effective = sz_Tb0_effective

    # pre-compute a few matrices / vectors
    Gt_a_lst = []
    lu_GtG_lst = []
    Tbeta_ri0_lst = []

    if 'beta' in kwargs:
        beta_lst = kwargs['beta']
        compute_beta = False
    else:
        beta_lst = []
        compute_beta = True

    for band_count in range(num_bands):
        G_loop = G_ri_lst[band_count]
        a_loop = a_ri_lst[band_count]

        lu_GtG_loop = linalg.lu_factor(np.dot(G_loop.T, G_loop), check_finite=False)
        Gt_a_loop = np.dot(G_loop.T, a_loop)

        if compute_beta:
            beta_ri_loop = linalg.lu_solve(lu_GtG_loop, Gt_a_loop, check_finite=False)
            beta_lst.append(beta_ri_loop)
        else:
            beta_ri_loop = beta_lst[band_count]

        if update_G:
            beta_ri_loop_full = np.dot(expand_b_mtx, np.dot(mtx_extract_b, beta_ri_loop))
        else:
            beta_ri_loop_full = np.dot(expand_b_mtx, beta_ri_loop)

        beta_cpx_loop = beta_ri_loop_full[:sz_fri_blk] + \
                        1j * beta_ri_loop_full[sz_fri_blk:]

        Tbeta_ri0_loop = T_mtx_joint_ri_half(
            np.reshape(beta_cpx_loop, (sz_fri0, sz_fri1), order='F'),
            sz_coef_row0, sz_coef_row1,
            sz_coef_col0, sz_coef_col1,
            mtx_shrink_row, mtx_shrink_col,
            expand_c_row_mtx, expand_c_col_mtx
        )

        # append to list
        Gt_a_lst.append(Gt_a_loop)
        lu_GtG_lst.append(lu_GtG_loop)
        Tbeta_ri0_lst.append(Tbeta_ri0_loop)

    # initialise loop
    max_iter = 30
    min_error = float('inf')

    rhs = np.concatenate((np.zeros(sz_coef + sz_S0, dtype=float),
                          np.array([1, 1])))

    for ini in range(max_ini):
        # select a subset of size (K + 1) of these coefficients
        # here the indices corresponds to the part of coefficients that are ZERO
        S_ri = planar_sel_coef_subset_ri_half(num_coef_row, num_coef_col, K,
                                              expand_c_row_mtx, expand_c_col_mtx,
                                              mtx_shrink_S_row, mtx_shrink_S_col)
        S_ri_T = S_ri.T

        # randomly initialised the annihilating filter coefficients
        c_row_ri_half = np.random.randn(num_coef_row)
        c_row_ri = np.dot(expand_c_row_mtx, c_row_ri_half)
        c_col_ri_half = np.random.randn(num_coef_col)
        c_col_ri = np.dot(expand_c_col_mtx, c_col_ri_half)
        # for row annihilation
        c_row = np.reshape(c_row_ri[:num_coef_row] + 1j * c_row_ri[num_coef_row:],
                           (sz_coef_row0, sz_coef_row1), order='F')
        # for col annihilation
        c_col = np.reshape(c_col_ri[:num_coef_col] + 1j * c_col_ri[num_coef_col:],
                           (sz_coef_col0, sz_coef_col1), order='F')

        # c0_T = linalg.block_diag(
        #     np.dot(c_row_ri_half[np.newaxis, :], np.dot(expand_c_row_mtx.T, expand_c_row_mtx)),
        #     np.dot(c_col_ri_half[np.newaxis, :], np.dot(expand_c_col_mtx.T, expand_c_col_mtx))
        # )
        c0_T = linalg.block_diag(
            c_row_ri_half[np.newaxis, :],
            c_col_ri_half[np.newaxis, :]
        )

        # c0_T = linalg.block_diag(
        #     np.dot(cpx_mtx2real(c_row.flatten('F')[np.newaxis, :]), expand_c_row_mtx),
        #     np.dot(cpx_mtx2real(c_col.flatten('F')[np.newaxis, :]), expand_c_col_mtx)
        # )
        c0 = c0_T.T

        R_loop_ri = R_mtx_joint_ri_half(
            c_row, c_col, sz_fri0, sz_fri1,
            expansion_mtx=expand_b_mtx,
            mtx_shrink_row=mtx_shrink_row,
            mtx_shrink_col=mtx_shrink_col,
            mtx_extract_b=mtx_extract_b
        )

        Q_H = linalg.qr(R_loop_ri, mode='economic', check_finite=False,
                        pivoting=True)[0][:, :sz_R0_effective].T

        R_loop_ri = np.dot(Q_H, R_loop_ri)

        # last row in mtx_loop
        mtx_loop_last_row = np.hstack((
            np.vstack((S_ri, c0_T)),
            np.zeros((sz_S0 + 2, sz_S0 + 2), dtype=float)
        ))

        lu_lst = None

        for inner in range(max_iter):
            if inner == 0:
                mtx_loop = np.vstack((
                    np.hstack((
                        planar_compute_mtx_obj_multiband_ri(
                            lu_GtG_lst, Tbeta_ri0_lst, R_loop_ri, Q_H, num_bands, sz_coef
                        ),
                        S_ri_T, c0
                    )),
                    mtx_loop_last_row
                ))
            else:
                mtx_loop[:sz_coef, :sz_coef] = \
                    planar_compute_mtx_obj_multiband_ri(
                        lu_GtG_lst, Tbeta_ri0_lst, R_loop_ri, Q_H, num_bands, sz_coef, lu_lst
                    )

            try:
                c_ri = linalg.solve(mtx_loop, rhs, check_finite=False)[:sz_coef]
            except linalg.LinAlgError:
                break

            c_row_ri = np.dot(expand_c_row_mtx, c_ri[:num_coef_row])
            c_row = np.reshape(c_row_ri[:num_coef_row] + 1j * c_row_ri[num_coef_row:],
                               (sz_coef_row0, sz_coef_row1), order='F')

            c_col_ri = np.dot(expand_c_col_mtx, c_ri[num_coef_row:])
            c_col = np.reshape(c_col_ri[:num_coef_col] + 1j * c_col_ri[num_coef_col:],
                               (sz_coef_col0, sz_coef_col1), order='F')

            # build new R matrix
            R_loop_ri = R_mtx_joint_ri_half(
                c_row, c_col, sz_fri0, sz_fri1,
                expansion_mtx=expand_b_mtx,
                mtx_shrink_row=mtx_shrink_row,
                mtx_shrink_col=mtx_shrink_col,
                mtx_extract_b=mtx_extract_b
            )

            Q_H = linalg.qr(R_loop_ri, mode='economic', check_finite=False,
                            pivoting=True)[0][:, :sz_R0_effective].T

            R_loop_ri = np.dot(Q_H, R_loop_ri)

            # compute the FRI sequence b and / or fitting error
            error_loop, lu_lst, b_recon_lst = planar_compute_b_ri(
                G_ri_lst, lu_GtG_lst, beta_lst, R_loop_ri, num_bands, a_ri_lst
            )

            if error_loop < min_error:
                min_error = error_loop
                # NOTE THAT THE FIRST SEGEMENT CORRESPONDS TO THE THE AMPLITUDE OF THE PREVIOUSLY
                # RECONSTRUCTED DIRAC WHEN UPDATE_G = TRUE
                b_opt_ri_lst = b_recon_lst
                c_row_opt = c_row
                c_col_opt = c_col

            if compute_mse and min_error < noise_level:
                break

        if compute_mse and min_error < noise_level:
            break

    return c_row_opt, c_col_opt, min_error, b_opt_ri_lst, ini


def planar_sel_coef_subset(num_coef_row, num_coef_col, K):
    # the selection indices that corresponds to the part where the coeffiicients are zero
    subset_idx_row = np.sort(np.random.permutation(num_coef_row)[K + 2:])
    subset_idx_col = subset_idx_row

    S_row = np.eye(num_coef_row)[subset_idx_row, :]
    S_col = np.eye(num_coef_col)[subset_idx_col, :]

    if S_row.shape[0] == 0 and S_col.shape[0] == 0:
        S = np.zeros((0, num_coef_row + num_coef_col))
    elif S_row.shape[0] == 0 and S_col.shape[0] != 0:
        S = np.column_stack((np.zeros((S_col.shape[0], num_coef_row)), S_col))
    elif S_row.shape[0] != 0 and S_col.shape[0] == 0:
        S = np.column_stack((S_row, np.zeros((S_row.shape[0], num_coef_col))))
    else:
        S = linalg.block_diag(S_row, S_col)

    return S


def planar_sel_coef_subset_ri(num_coef_row, num_coef_col, K):
    # the selection indices that corresponds to the part where the coeffiicients are zero
    subset_idx_row = np.sort(np.random.permutation(num_coef_row)[K + 1:])
    subset_idx_col = np.sort(np.random.permutation(num_coef_col)[K + 1:])

    # # make sure the selection indices are different
    # identical_check = np.all(subset_idx_row == subset_idx_col)
    # while identical_check:
    #     subset_idx_col = np.sort(np.random.permutation(num_coef_col)[K + 1:])
    #     identical_check = np.all(subset_idx_row == subset_idx_col)

    S_row = np.eye(num_coef_row)[subset_idx_row, :]
    S_col = np.eye(num_coef_col)[subset_idx_col, :]

    if S_row.shape[0] == 0 and S_col.shape[0] == 0:
        S = np.zeros((0, num_coef_row + num_coef_col))
    elif S_row.shape[0] == 0 and S_col.shape[0] != 0:
        S = np.column_stack((np.zeros((S_col.shape[0], num_coef_row)), S_col))
    elif S_row.shape[0] != 0 and S_col.shape[0] == 0:
        S = np.column_stack((S_row, np.zeros((S_row.shape[0], num_coef_col))))
    else:
        S = linalg.block_diag(S_row, S_col)

    # we express all matrices and vectors in their real-valued extended form
    if S_row.shape[0] == 0 and S_col.shape[0] == 0:
        S_ri = np.zeros((0, 2 * (num_coef_row + num_coef_col)))
    else:
        S_ri = cpx_mtx2real(S)

    return S_ri


def set_zero_idx(total_num_coef, non_zero_coef):
    # due to Hermitian symmetry, we only specify half of the zero-valued indices.
    # The other half is then extended by symmetry.
    if total_num_coef % 2 == 0:
        half_len = total_num_coef // 2
        if non_zero_coef % 2 == 0:
            subset_idx_half = np.random.permutation(half_len)[non_zero_coef // 2:]
            subset_idx = np.sort(
                np.concatenate((subset_idx_half,
                                total_num_coef - 1 - subset_idx_half))
            )
        else:
            raise ValueError('even number of block size and odd number of non-zero coefficients.')
    else:
        half_len = (total_num_coef - 1) // 2
        if non_zero_coef % 2 == 0:
            subset_idx_half = \
                np.append(
                    np.random.permutation(half_len)[non_zero_coef // 2:], half_len
                )
            subset_idx = np.sort(
                np.concatenate((subset_idx_half,
                                total_num_coef - 1 - subset_idx_half[:-1]))
            )
        else:
            subset_idx_half = np.random.permutation(half_len)[(non_zero_coef - 1) // 2:]
            subset_idx = np.sort(
                np.concatenate((subset_idx_half,
                                total_num_coef - 1 - subset_idx_half))
            )

    return subset_idx


def planar_sel_coef_subset_ri_half(num_coef_row, num_coef_col, K,
                                   expansion_mtx_coef_row,
                                   expansion_mtx_coef_col,
                                   mtx_shrink_S_row,
                                   mtx_shrink_S_col):
    """
    select the portion of annihilation coefficients that are ZERO. here we also enforce
    that the annihilation coefficients are Hermitian symmetric.
    :param num_coef_row: number of coefficients for the row annihilation
    :param num_coef_col: number of coefficients for the column annihilation
    :param K: number of Dirac deltas. The number of non-zero in both coeffcieints are K + 1
    :param expansion_mtx_coef_row: expansion matrix for the Hermitian symmetric annihilation coefficients
    :param expansion_mtx_coef_col: expansion matrix for the Hermitian symmetric annihilation coefficients
    :return:
    """
    assert not (num_coef_row % 2 == 0 and (K + 1) % 2 == 1)  # verify this does not happen
    assert not (num_coef_col % 2 == 0 and (K + 1) % 2 == 1)

    subset_idx_row = set_zero_idx(num_coef_row, K + 1)
    subset_idx_col = set_zero_idx(num_coef_col, K + 1)

    S_row = np.eye(num_coef_row)[subset_idx_row, :]
    S_col = np.eye(num_coef_col)[subset_idx_col, :]

    if S_row.shape[0] == 0 and S_col.shape[0] == 0:
        S_ri = np.zeros((0, num_coef_row + num_coef_col), dtype=int)
    elif S_row.shape[0] == 0 and S_col.shape[0] != 0:
        S_ri = np.dot(mtx_shrink_S_col,
                      np.column_stack((
                          np.zeros((S_col.shape[0] * 2, num_coef_row)),
                          np.dot(cpx_mtx2real(S_col),
                                 expansion_mtx_coef_col)
                      ))
                      )
    elif S_row.shape[0] != 0 and S_col.shape[0] == 0:
        S_ri = np.dot(mtx_shrink_S_row,
                      np.column_stack((
                          np.dot(cpx_mtx2real(S_row),
                                 expansion_mtx_coef_row),
                          np.zeros((S_row.shape[0] * 2, num_coef_col))
                      ))
                      )
    else:
        S_ri = linalg.block_diag(
            np.dot(mtx_shrink_S_row,
                   np.dot(cpx_mtx2real(S_row),
                          expansion_mtx_coef_row)
                   ),
            np.dot(mtx_shrink_S_col,
                   np.dot(cpx_mtx2real(S_col),
                          expansion_mtx_coef_col)
                   )
        )

    return S_ri


def planar_compute_mtx_obj_multiband(GtG_inv_lst, Tbeta0_lst, Rmtx_band, Q_H,
                                     num_bands, sz_coef, lu_lst=None):
    """
    compute T^H (R (G^H G)^{-1} R^H)^{-1} T
    :param GtG_inv_lst: LU decomposition of G^H G
    :param Tbeta0_lst: T(beta) BEFORE left multiplying by a matrix that extracts
            the rows that corresponds to the effective number of equations
    :param Rmtx_band: right dual matrix fro the annihilating filter (same for each block)
    :param Q_H: a rectangular matrix that extracts the effective lines of equations
    :param num_bands: number of sub-bands
    :param sz_coef: size of the annihilating filter coefficients
    :param lu_lst: LU decomposition of R (G^H G)^{-1} R^H
    :return:
    """
    mtx = np.zeros((sz_coef, sz_coef), dtype=complex)
    if lu_lst is None:
        for band_count in range(num_bands):
            Tbeta_loop = np.dot(Q_H, Tbeta0_lst[band_count])
            mtx += np.dot(Tbeta_loop.conj().T,
                          linalg.solve(
                              np.dot(
                                  Rmtx_band,
                                  np.dot(GtG_inv_lst[band_count],
                                         Rmtx_band.conj().T)
                              ),
                              Tbeta_loop, check_finite=False
                          )
                          )
    else:
        for band_count in range(num_bands):
            Tbeta_loop = np.dot(Q_H, Tbeta0_lst[band_count])
            mtx += np.dot(Tbeta_loop.conj().T,
                          linalg.lu_solve(lu_lst[band_count], Tbeta_loop,
                                          check_finite=False)
                          )

    return mtx


def planar_compute_mtx_obj_multiband_slow(lu_GtG_lst, Tbeta0_lst, Rmtx_band, Q_H,
                                          num_bands, sz_coef, lu_lst=None):
    """
    compute T^H (R (G^H G)^{-1} R^H)^{-1} T
    :param lu_GtG_lst: LU decomposition of G^H G
    :param Tbeta0_lst: T(beta) BEFORE left multiplying by a matrix that extracts
            the rows that corresponds to the effective number of equations
    :param Rmtx_band: right dual matrix fro the annihilating filter (same for each block)
    :param Q_H: a rectangular matrix that extracts the effective lines of equations
    :param num_bands: number of sub-bands
    :param sz_coef: size of the annihilating filter coefficients
    :param lu_lst: LU decomposition of R (G^H G)^{-1} R^H
    :return:
    """
    mtx = np.zeros((sz_coef, sz_coef), dtype=complex)
    if lu_lst is None:
        for band_count in range(num_bands):
            Tbeta_loop = np.dot(Q_H, Tbeta0_lst[band_count])
            mtx += np.dot(Tbeta_loop.conj().T,
                          linalg.solve(
                              np.dot(
                                  Rmtx_band,
                                  linalg.lu_solve(
                                      lu_GtG_lst[band_count],
                                      Rmtx_band.conj().T,
                                      check_finite=False)
                              ),
                              Tbeta_loop, check_finite=False
                          )
                          )
    else:
        for band_count in range(num_bands):
            Tbeta_loop = np.dot(Q_H, Tbeta0_lst[band_count])
            mtx += np.dot(Tbeta_loop.conj().T,
                          linalg.lu_solve(lu_lst[band_count], Tbeta_loop,
                                          check_finite=False)
                          )

    return mtx


def planar_compute_mtx_obj_multiband_ri(lu_GtG_lst, Tbeta0_lst, Rmtx_band_ri, Q_H,
                                        num_bands, sz_coef, lu_lst=None):
    """
    compute T^H (R (G^H G)^{-1} R^H)^{-1} T
    :param lu_GtG_lst: LU decomposition of G^H G
    :param Tbeta0_lst: T(beta) BEFORE left multiplying by a matrix that extracts
            the rows that corresponds to the effective number of equations
    :param Rmtx_band_ri: right dual matrix fro the annihilating filter (same for each block)
    :param Q_H: a rectangular matrix that extracts the effective lines of equations
    :param num_bands: number of sub-bands
    :param sz_coef: size of the annihilating filter coefficients
    :param lu_lst: LU decomposition of R (G^H G)^{-1} R^H
    :return:
    """
    mtx = np.zeros((sz_coef, sz_coef), dtype=float)
    if lu_lst is None:
        for band_count in range(num_bands):
            Tbeta_loop = np.dot(Q_H, Tbeta0_lst[band_count])
            mtx += np.dot(Tbeta_loop.T,
                          linalg.solve(
                              np.dot(
                                  Rmtx_band_ri,
                                  linalg.lu_solve(
                                      lu_GtG_lst[band_count],
                                      Rmtx_band_ri.T,
                                      check_finite=False)
                              ),
                              Tbeta_loop, check_finite=False
                          )
                          )
    else:
        for band_count in range(num_bands):
            Tbeta_loop = np.dot(Q_H, Tbeta0_lst[band_count])
            mtx += np.dot(Tbeta_loop.T,
                          linalg.lu_solve(lu_lst[band_count], Tbeta_loop,
                                          check_finite=False)
                          )

    return mtx


def planar_compute_obj_val(GtG_inv_lst, Tbeta0_lst, R_loop, Q_H, c, num_bands, sz_coef):
    """
    compute the fitting error without computing explicitly the reconstructed uniformly
    sampled sinusoids b.
    :param GtG_inv_lst: list of (G^H G)^{-1} for different subbands
    :param Tbeta0_lst: list of Tbeta
    :param R_loop: right dual matrix associated with the annihilation coefficients
    :param c: annihilation coefficients
    :param num_bands: number of subbands
    :param sz_coef: size of the annihilation coefficients
    :return:
    """
    mtx = np.zeros((sz_coef, sz_coef), dtype=complex)

    for band_count in range(num_bands):
        Tbeta_loop = np.dot(Q_H, Tbeta0_lst[band_count])

        mtx += np.dot(Tbeta_loop.conj().T,
                      linalg.solve(
                          np.dot(R_loop,
                                 np.dot(GtG_inv_lst[band_count],
                                        R_loop.conj().T),
                                 ),
                          Tbeta_loop, check_finite=False
                      )
                      )

    fitting_error = np.sqrt(np.dot(c.conj().T, np.dot(mtx, c)).real)

    return fitting_error, mtx


def planar_compute_b(G_lst, GtG_inv_lst, beta_lst, R_loop, num_bands, a_lst):
    """
    compute the uniformly sampled sinusoids b (i.e., the FRI sequence) from the updated
    annihilating filter coefficients.
    :param G_lst: a list of G for different sub-bands
    :param GtG_inv_lst: a list of the LU decomposition of G^H G for different sub-bands
    :param beta_lst: a list of beta-s for different sub-bands
    :param R_loop: right dual matrix for the annihilating filter
            (same for each block -> not a list)
    :param num_bands: number of bands
    :param a: a 2D numpy array. each column corresponds to the measurements
            within a sub-band
    :return:
    """
    b_lst = []
    a_Gb_lst = []
    lu_lst = []

    for band_count in range(num_bands):
        GtG_inv_loop = GtG_inv_lst[band_count]
        beta_loop = beta_lst[band_count]

        lu_loop = linalg.lu_factor(
            np.dot(R_loop, np.dot(GtG_inv_loop, R_loop.conj().T)),
            overwrite_a=True, check_finite=False
        )

        b_loop = beta_loop - np.dot(
            GtG_inv_loop,
            np.dot(R_loop.conj().T,
                   linalg.lu_solve(lu_loop, np.dot(R_loop, beta_loop), check_finite=False))
        )

        # append results to lists
        b_lst.append(b_loop)
        a_Gb_lst.append(a_lst[band_count] - np.dot(G_lst[band_count], b_loop))
        lu_lst.append(lu_loop)

    return linalg.norm(np.concatenate(a_Gb_lst)), lu_lst, b_lst


def planar_compute_b_slow(G_lst, lu_GtG_lst, beta_lst, R_loop, num_bands, a_lst):
    """
    compute the uniformly sampled sinusoids b (i.e., the FRI sequence) from the updated
    annihilating filter coefficients.
    :param G_lst: a list of G for different sub-bands
    :param lu_GtG_lst: a list of the LU decomposition of G^H G for different sub-bands
    :param beta_lst: a list of beta-s for different sub-bands
    :param R_loop: right dual matrix for the annihilating filter
            (same for each block -> not a list)
    :param num_bands: number of bands
    :param a: a 2D numpy array. each column corresponds to the measurements
            within a sub-band
    :return:
    """
    b_lst = []
    a_Gb_lst = []
    lu_lst = []

    for band_count in range(num_bands):
        lu_GtG_loop = lu_GtG_lst[band_count]
        beta_loop = beta_lst[band_count]

        lu_loop = linalg.lu_factor(
            np.dot(R_loop, linalg.lu_solve(lu_GtG_loop, R_loop.conj().T, check_finite=False)),
            overwrite_a=True, check_finite=False
        )

        b_loop = beta_loop - linalg.lu_solve(
            lu_GtG_loop,
            np.dot(R_loop.conj().T,
                   linalg.lu_solve(lu_loop, np.dot(R_loop, beta_loop), check_finite=False)),
            check_finite=False
        )

        # append results to lists
        b_lst.append(b_loop)
        a_Gb_lst.append(a_lst[band_count] - np.dot(G_lst[band_count], b_loop))
        lu_lst.append(lu_loop)

    return linalg.norm(np.concatenate(a_Gb_lst)), lu_lst, b_lst


def planar_compute_b_ri(G_ri_lst, lu_GtG_lst, beta_lst, R_loop_ri, num_bands, a_ri_lst):
    """
    compute the uniformly sampled sinusoids b (i.e., the FRI sequence) from the updated
    annihilating filter coefficients.
    :param G_ri_lst: a list of G for different sub-bands
    :param lu_GtG_lst: a list of the LU decomposition of G^H G for different sub-bands
    :param beta_lst: a list of beta-s for different sub-bands
    :param R_loop_ri: right dual matrix for the annihilating filter
            (same for each block -> not a list)
    :param num_bands: number of bands
    :param a_ri_lst: a 2D numpy array. each column corresponds to the measurements
            within a sub-band
    :return:
    """
    b_lst = []
    a_Gb_lst = []
    lu_lst = []

    for band_count in range(num_bands):
        lu_GtG_loop = lu_GtG_lst[band_count]
        beta_loop = beta_lst[band_count]

        lu_loop = linalg.lu_factor(
            np.dot(R_loop_ri, linalg.lu_solve(lu_GtG_loop, R_loop_ri.T, check_finite=False)),
            overwrite_a=True, check_finite=False
        )

        b_loop = beta_loop - linalg.lu_solve(
            lu_GtG_loop,
            np.dot(R_loop_ri.T,
                   linalg.lu_solve(lu_loop, np.dot(R_loop_ri, beta_loop), check_finite=False)),
            check_finite=False
        )

        # append results to lists
        b_lst.append(b_loop)
        a_Gb_lst.append(a_ri_lst[band_count] - np.dot(G_ri_lst[band_count], b_loop))
        lu_lst.append(lu_loop)

    return linalg.norm(np.concatenate(a_Gb_lst)), lu_lst, b_lst


def joblib_compute_fitting_error_amp_beamforming(
        xk_recon_sel, yk_recon_sel, amp_mtx_sel,
        a_lst, p_x_normalised, p_y_normalised,
        beam_weights_func, num_bands, theano_func, backend):
    return planar_compute_fitting_error_amp_beamforming(
        a_lst, p_x_normalised, p_y_normalised,
        xk_recon_sel, yk_recon_sel, beam_weights_func, num_bands,
        theano_func=theano_func, backend=backend, amp_mtx_lst=amp_mtx_sel)[0]


def planar_compute_fitting_error_amp_beamforming(
        a_lst, p_x_normalised, p_y_normalised,
        xk, yk, beam_weights_func,
        num_bands,
        theano_func=None, backend='cpu',
        amp_mtx_lst=None):
    amplitude = []
    if amp_mtx_lst is None:
        amp_mtx_lst = []
        compute_amp_mtx = True
    else:
        compute_amp_mtx = False
    error_loop = 0

    for band_count in range(num_bands):
        a_loop = a_lst[band_count]
        if compute_amp_mtx:
            amp_mtx_loop = planar_build_mtx_amp_beamforming(
                p_x_normalised[:, :, :, band_count],
                p_y_normalised[:, :, :, band_count],
                xk, yk, beam_weights_func=beam_weights_func,
                theano_func=theano_func, backend=backend)

            amp_mtx_lst.append(amp_mtx_loop)
        else:
            amp_mtx_loop = amp_mtx_lst[band_count]
        # since the amplitude are positive real numbers
        amp_mtx_loop_half_ri = np.vstack((amp_mtx_loop.real, amp_mtx_loop.imag))

        amplitude_band = sp.optimize.nnls(
            np.dot(amp_mtx_loop_half_ri.T, amp_mtx_loop_half_ri),
            np.dot(amp_mtx_loop_half_ri.T, np.concatenate((a_loop.real, a_loop.imag)))
        )[0]

        amplitude.append(amplitude_band)
        error_loop += linalg.norm(a_loop - np.dot(amp_mtx_loop, amplitude_band))

    return error_loop, np.reshape(np.asarray(amplitude), (-1, num_bands), order='F'), amp_mtx_lst


def planar_compute_fitting_error_amp_beamforming_ri(a_ri_lst, p_x_normalised, p_y_normalised,
                                                    xk, yk, beam_weights_func,
                                                    num_station, num_bands, num_sti,
                                                    backend='cpu'):
    amplitude = []
    error_loop = 0

    for band_count in range(num_bands):
        a_ri_loop = a_ri_lst[band_count]

        amp_mtx_loop = planar_build_mtx_amp_ri_beamforming(
            p_x_normalised[:, :, :, band_count], p_y_normalised[:, :, :, band_count],
            xk, yk, beam_weights_func=beam_weights_func,
            num_station=num_station, num_sti=num_sti,
            backend=backend
        )

        amplitude_band = sp.optimize.nnls(
            np.dot(amp_mtx_loop.T, amp_mtx_loop),
            np.dot(amp_mtx_loop.T, a_ri_loop)
        )[0]

        amplitude.append(amplitude_band)
        error_loop += linalg.norm(a_ri_loop - np.dot(amp_mtx_loop, amplitude_band))

    return error_loop, np.reshape(np.asarray(amplitude), (-1, num_bands), order='F')


def planar_build_beta(alpha, xk, yk, m_limit, n_limit, tau_inter_x, tau_inter_y):
    m_grid, n_grid = np.meshgrid(np.arange(-m_limit, m_limit + 1, step=1, dtype=int),
                                 np.arange(-n_limit, n_limit + 1, step=1, dtype=int))
    m_grid = np.reshape(m_grid, (-1, 1), order='F')
    n_grid = np.reshape(n_grid, (-1, 1), order='F')

    alpha = np.reshape(alpha, (-1, 1), order='F')
    xk = np.reshape(xk, (1, -1), order='F')
    yk = np.reshape(yk, (1, -1), order='F')

    beta_cpx = np.dot(np.exp(-1j * (xk * 2 * np.pi / tau_inter_x * m_grid +
                                    yk * 2 * np.pi / tau_inter_y * n_grid)
                             ),
                      alpha).squeeze()

    return beta_cpx


def planar_build_beta_ri(alpha, xk, yk, m_limit, n_limit, tau_inter_x, tau_inter_y):
    m_grid, n_grid = np.meshgrid(np.arange(-m_limit, m_limit + 1, step=1, dtype=int),
                                 np.arange(-n_limit, n_limit + 1, step=1, dtype=int))
    half_size = int((2 * m_limit + 1) * (2 * n_limit + 1) // 2 + 1)
    m_grid = np.reshape(m_grid, (-1, 1), order='F')[:half_size]
    n_grid = np.reshape(n_grid, (-1, 1), order='F')[:half_size]

    alpha = np.reshape(alpha, (-1, 1), order='F')
    xk = np.reshape(xk, (1, -1), order='F')
    yk = np.reshape(yk, (1, -1), order='F')

    beta_cpx = np.dot(np.exp(-1j * (xk * 2 * np.pi / tau_inter_x * m_grid +
                                    yk * 2 * np.pi / tau_inter_y * n_grid)
                             ),
                      alpha).squeeze()

    return np.concatenate((beta_cpx.real, beta_cpx.imag[:-1]))


def planar_extract_innovation(a_lst, K, c_row, c_col,
                              p_x_rotated, p_y_rotated,
                              tau_inter_x, tau_inter_y,
                              partial_beamforming_func,
                              theano_func=None, backend='cpu',
                              **kwargs):
    """
    retrieval Dirac parameters, i.e., Dirac locations (horizontal and vertical) and
    amplitudes from the reconstructed annihilating filter coefficients.
    :return:
    """
    if 'G_amp_ref_lst' in kwargs:
        G_amp_ref_lst = kwargs['G_amp_ref_lst']
    else:
        G_amp_ref_lst = None

    if 'x_ref_rotated' in kwargs and 'y_ref_rotated' in kwargs:
        x_ref_rotated = kwargs['x_ref_rotated']
        y_ref_rotated = kwargs['y_ref_rotated']
    else:
        x_ref_rotated = None
        y_ref_rotated = None

    if 'extract_indicator_lst' in kwargs:
        extract_indicator_lst = kwargs['extract_indicator_lst']
    else:
        extract_indicator_lst = None

    # root finding
    z1, z2 = find_roots(c_row, c_col)
    extraction_index = np.bitwise_or(np.bitwise_not(np.isclose(z1, 0)),
                                     np.bitwise_not(np.isclose(z2, 0)))
    z1 = z1[extraction_index]
    z2 = z2[extraction_index]
    z1 /= np.abs(z1)
    z2 /= np.abs(z2)

    xk_recon = np.angle(z1) * tau_inter_x / (-2 * np.pi)
    yk_recon = np.angle(z2) * tau_inter_y / (-2 * np.pi)
    remove_nan = np.bitwise_or(np.isnan(xk_recon), np.isnan(yk_recon))
    xk_recon = xk_recon[~remove_nan]
    yk_recon = yk_recon[~remove_nan]

    xk_recon_rotated, yk_recon_rotated = \
        planar_select_reliable_recon_internal(
            a_lst, p_x_rotated, p_y_rotated,
            xk_recon.flatten('F'), yk_recon.flatten('F'), K,
            partial_beamforming_func,
            num_removal=xk_recon.size - K,
            theano_func=theano_func, backend=backend,
            extract_indicator_lst=extract_indicator_lst,
            G_amp_ref_lst=G_amp_ref_lst,
            x_ref=x_ref_rotated, y_ref=y_ref_rotated,
            strategy='aggregate' if xk_recon.size > 200 else 'one_by_one'
        )[:2]

    return xk_recon_rotated, yk_recon_rotated


def determine_max_coef_sz(data_sz0, data_sz1):
    """
    determine the annihilating filter size for a given set of data with certain size.
    we choose the filter size in such a way that the number of possible annihilation
    equations is as close as possible to the number of coefficients.
    :param data_sz0: data block size 0
    :param data_sz1: data block size 1
    :return: the equivalent number of Dirac
    """
    return int(np.floor((data_sz0 + 1) * (data_sz1 + 1) /
                        (data_sz0 + data_sz1 + 2)) - 1
               ) ** 2 - 1


def planar_select_reliable_recon_internal(
        a_lst, p_x_normalised, p_y_normalised,
        xk_recon, yk_recon, num_sel,
        beam_weights_func, num_removal,
        theano_func=None, backend='cpu',
        extract_indicator_lst=None,
        G_amp_ref_lst=None, x_ref=None, y_ref=None,
        strategy='aggregate'):
    """
    select reliable reconstruction based on the "leave-one-out" error
    :param a_lst: noisy visibility
    :param p_x_normalised: antenna (horizontal) locations
    :param p_y_normalised: antenna (vertical) locations
    :param xk_recon: reconstructed Dirac locations (horizontal)
    :param yk_recon: reconstructed Dirac locations (vertical)
    :param beam_weights_func: beam forming function
    :param num_station: number of stations
    :param num_bands: number of sub-bands
    :param num_sti: number of STIs
    :param num_removal: number of Diracs to be removed
    :param backend: either 'cpu' or 'gpu'
    :return:
    """
    num_bands = len(a_lst)
    if extract_indicator_lst is None:
        extract_subset = False
    else:
        extract_subset = True

    if G_amp_ref_lst is None or x_ref is None or y_ref is None:
        ref_sol_available = False
        K_ref = 0
    else:
        ref_sol_available = True
        K_ref = G_amp_ref_lst[0].shape[1]

    if strategy == 'aggregate':
        # reshape to use broadcasting
        xk_recon = np.reshape(xk_recon, (1, -1), order='F')
        yk_recon = np.reshape(yk_recon, (1, -1), order='F')

        # find the Dirac amplitudes
        alphak_recon_lst = []
        if ref_sol_available:
            for band_count in range(num_bands):
                if extract_subset:
                    amp_mtx_loop = np.column_stack((
                        G_amp_ref_lst[band_count],
                        planar_build_mtx_amp_beamforming(
                            p_x_normalised[:, :, :, band_count],
                            p_y_normalised[:, :, :, band_count],
                            xk_recon, yk_recon,
                            beam_weights_func=beam_weights_func,
                            theano_func=theano_func, backend=backend
                        )[extract_indicator_lst[band_count], :]
                    ))
                else:
                    amp_mtx_loop = np.column_stack((
                        G_amp_ref_lst[band_count],
                        planar_build_mtx_amp_beamforming(
                            p_x_normalised[:, :, :, band_count],
                            p_y_normalised[:, :, :, band_count],
                            xk_recon, yk_recon,
                            beam_weights_func=beam_weights_func,
                            theano_func=theano_func, backend=backend
                        )
                    ))
                # since the amplitude are positive real numbers
                # amp_mtx_loop_half_ri = np.vstack((amp_mtx_loop.real, amp_mtx_loop.imag))
                amp_mtx_T_amp_mtx = np.dot(amp_mtx_loop.real.T, amp_mtx_loop.real) + \
                                    np.dot(amp_mtx_loop.imag.T, amp_mtx_loop.imag)
                amp_mtx_T_a = np.dot(amp_mtx_loop.real.T, a_lst[band_count].real) + \
                              np.dot(amp_mtx_loop.imag.T, a_lst[band_count].imag)
                del amp_mtx_loop  # free up memory

                alphak_recon_lst.append(
                    sp.optimize.nnls(amp_mtx_T_amp_mtx, amp_mtx_T_a)[0]
                )
        else:
            for band_count in range(num_bands):
                amp_mtx_loop = planar_build_mtx_amp_beamforming(
                    p_x_normalised[:, :, :, band_count],
                    p_y_normalised[:, :, :, band_count],
                    xk_recon, yk_recon,
                    beam_weights_func=beam_weights_func,
                    theano_func=theano_func, backend=backend
                )
                if extract_subset:
                    amp_mtx_loop = amp_mtx_loop[extract_indicator_lst[band_count], :]

                # since the amplitude are positive real numbers
                # amp_mtx_loop_half_ri = np.vstack((amp_mtx_loop.real, amp_mtx_loop.imag))
                amp_mtx_T_amp_mtx = np.dot(amp_mtx_loop.real.T, amp_mtx_loop.real) + \
                                    np.dot(amp_mtx_loop.imag.T, amp_mtx_loop.imag)
                amp_mtx_T_a = np.dot(amp_mtx_loop.real.T, a_lst[band_count].real) + \
                              np.dot(amp_mtx_loop.imag.T, a_lst[band_count].imag)
                del amp_mtx_loop  # free up memory

                alphak_recon_lst.append(
                    sp.optimize.nnls(amp_mtx_T_amp_mtx, amp_mtx_T_a)[0]
                )

        alphak_recon = np.reshape(np.concatenate(alphak_recon_lst),
                                  (-1, num_bands), order='F')

        # determine the correct association of horizontal and vertical locations
        # by extracting the locations that have the largest amplitudes
        alphak_sort_idx = np.argsort(np.abs(alphak_recon[K_ref:, :]), axis=0)[:-num_sel - 1:-1]

        # use majority vote across all sub-bands
        idx_all = np.zeros((alphak_recon.shape[0] - K_ref, alphak_recon.shape[1]), dtype=int)
        for loop in range(num_bands):
            idx_all[alphak_sort_idx[:, loop], loop] = 1
        idx_opt = np.argsort(np.sum(idx_all, axis=1))[:-num_sel - 1:-1]

        if ref_sol_available:
            xk_reliable = np.concatenate((x_ref, (xk_recon.flatten('F'))[idx_opt]))
            yk_reliable = np.concatenate((y_ref, (yk_recon.flatten('F'))[idx_opt]))
        else:
            xk_reliable = (xk_recon.flatten('F'))[idx_opt]
            yk_reliable = (yk_recon.flatten('F'))[idx_opt]
    else:
        if xk_recon.size > 1:
            mask_all = (np.column_stack((
                np.ones((xk_recon.size, K_ref), dtype=int),
                np.ones((xk_recon.size, xk_recon.size), dtype=int) -
                np.eye(xk_recon.size, dtype=int)
            ))).astype(bool)

            # compute the amp_mtx with all Dirac locations
            if ref_sol_available:
                if extract_subset:
                    amp_mtx_full = [
                        np.column_stack((
                            G_amp_ref_lst[band_count],
                            planar_build_mtx_amp_beamforming(
                                p_x_normalised[:, :, :, band_count],
                                p_y_normalised[:, :, :, band_count],
                                xk_recon, yk_recon, beam_weights_func,
                                theano_func=theano_func,
                                backend=backend)[extract_indicator_lst[band_count], :]
                        ))
                        for band_count in range(num_bands)]
                else:
                    amp_mtx_full = [
                        np.column_stack((
                            G_amp_ref_lst[band_count],
                            planar_build_mtx_amp_beamforming(
                                p_x_normalised[:, :, :, band_count],
                                p_y_normalised[:, :, :, band_count],
                                xk_recon, yk_recon, beam_weights_func,
                                theano_func=theano_func, backend=backend)
                        ))
                        for band_count in range(num_bands)]
            else:
                if extract_subset:
                    amp_mtx_full = [
                        planar_build_mtx_amp_beamforming(
                            p_x_normalised[:, :, :, band_count],
                            p_y_normalised[:, :, :, band_count],
                            xk_recon, yk_recon, beam_weights_func,
                            theano_func=theano_func,
                            backend=backend)[extract_indicator_lst[band_count], :]
                        for band_count in range(num_bands)]
                else:
                    amp_mtx_full = [
                        planar_build_mtx_amp_beamforming(
                            p_x_normalised[:, :, :, band_count],
                            p_y_normalised[:, :, :, band_count],
                            xk_recon, yk_recon, beam_weights_func,
                            theano_func=theano_func, backend=backend)
                        for band_count in range(num_bands)]

            leave_one_out_error = [
                planar_compute_fitting_error_amp_beamforming(
                    a_lst, p_x_normalised, p_y_normalised,
                    xk_recon[mask_all[removal_ind, K_ref:]],
                    yk_recon[mask_all[removal_ind, K_ref:]],
                    beam_weights_func,
                    num_bands,
                    theano_func=theano_func,
                    backend=backend,
                    amp_mtx_lst=[
                        amp_mtx_full[band_count][:, mask_all[removal_ind, :]]
                        for band_count in range(num_bands)]
                )[0]
                for removal_ind in range(mask_all.shape[0])]

            idx_opt = np.argsort(np.asarray(leave_one_out_error))[num_removal:]
        else:
            idx_opt = np.array([0])
        if ref_sol_available:
            xk_reliable = np.concatenate((x_ref, xk_recon[idx_opt]))
            yk_reliable = np.concatenate((y_ref, yk_recon[idx_opt]))
        else:
            xk_reliable = xk_recon[idx_opt]
            yk_reliable = yk_recon[idx_opt]

    return xk_reliable, yk_reliable


def planar_select_reliable_recon(a, p_x, p_y, xk_recon, yk_recon,
                                 omega_bands, light_speed, beam_weights_func,
                                 num_station, num_bands, num_sti, num_removal,
                                 theano_func=None, backend='cpu'):
    """
    select reliable reconstruction based on the "leave-one-out" error
    :param a: noisy visibility
    :param p_x: antenna (horizontal) locations
    :param p_y: antenna (vertical) locations
    :param xk_recon: reconstructed Dirac locations (horizontal)
    :param yk_recon: reconstructed Dirac locations (vertical)
    :param beam_weights_func: beam forming function
    :param num_station: number of stations
    :param num_bands: number of sub-bands
    :param num_sti: number of STIs
    :param num_removal: number of Diracs to be removed
    :param backend: either 'cpu' or 'gpu'
    :return:
    """
    a = np.reshape(a, (-1, num_bands), order='F')
    a_lst = [a[:, band_count] for band_count in range(num_bands)]  # the list representation
    K = xk_recon.size  # number of Dirac
    norm_factor = np.reshape(light_speed / omega_bands, (1, 1, 1, -1), order='F')
    # normalised antenna coordinates
    p_x_normalised = np.reshape(p_x, (-1, num_station, num_sti, 1),
                                order='F') / norm_factor
    p_y_normalised = np.reshape(p_y, (-1, num_station, num_sti, 1),
                                order='F') / norm_factor

    if num_removal != 0:
        mask_all = (np.ones((K, K), dtype=int) - np.eye(K, dtype=int)).astype(bool)

        # compute the amp_mtx with all Dirac locations
        amp_mtx_full = [
            planar_build_mtx_amp_beamforming(
                p_x_normalised[:, :, :, band_count],
                p_y_normalised[:, :, :, band_count],
                xk_recon, yk_recon, beam_weights_func,
                theano_func=theano_func, backend=backend)
            for band_count in range(num_bands)]

        leave_one_out_error = []
        for removal_ind in range(K):
            amp_mtx_loop = [amp_mtx_full[band_count][:, mask_all[removal_ind, :]]
                            for band_count in range(num_bands)]
            leave_one_out_error.append(
                planar_compute_fitting_error_amp_beamforming(
                    a_lst, p_x_normalised, p_y_normalised,
                    xk_recon[mask_all[removal_ind, :]],
                    yk_recon[mask_all[removal_ind, :]],
                    beam_weights_func,
                    num_bands,
                    theano_func=theano_func,
                    backend=backend,
                    amp_mtx_lst=amp_mtx_loop
                )[0]
            )

        idx_opt = np.argsort(np.asarray(leave_one_out_error))[num_removal:]
        xk_reliable = xk_recon[idx_opt]
        yk_reliable = yk_recon[idx_opt]
    else:
        xk_reliable = xk_recon
        yk_reliable = yk_recon

    amplitude_reliable = planar_compute_fitting_error_amp_beamforming(
        a_lst, p_x_normalised, p_y_normalised, xk_reliable, yk_reliable,
        beam_weights_func, num_bands,
        theano_func=theano_func, backend=backend
    )[1]

    return xk_reliable, yk_reliable, amplitude_reliable


def planar_calcuate_crb(sigma_noise, p_x_normalised, p_y_normalised,
                        xk, yk, alphak,
                        beam_weights_func=planar_beamforming_func,
                        num_sti=1, num_bands=1):
    """
    calculate the Cramer-Rao lower bound for a given anntenna layout and source locations.
    :param sigma_noise: standard deviation of the complex circulant Gaussian noise.
                For noise = noise_r + 1j * noise_i, then
                noise_r ~ (0, sigma_noise^2 / 2) and noise_i ~ (0, sigma_noise^2 / 2)
    :param p_x_normalised: normalised (by wavelength and 2pi) antenna locations (x-axis)
    :param p_y_normalised: normalised (by wavelength and 2pi) antenna locations (y-axis)
    :param xk: ground truth Dirac delta locations (x-axis)
    :param yk: ground truth Dirac delta locations (y-axis)
    :param beam_weights_func: beamforming function
    :param num_station: number of stations
    :param num_sti: number of integration times
    :param num_bands: number of subbands.
    :return:
    """
    # CAUTION: NOTE THE DEFINITION OF SIGMA_NOISE HERE!!
    eye_3K = np.eye(3 * xk.size)
    crb_lst = []
    for band_count in range(num_bands):
        Phi_cpx_band = np.vstack(
            [
                planar_calcuate_crb_inner(
                    p_x_normalised[:, :, sti_loop, band_count],
                    p_y_normalised[:, :, sti_loop, band_count],
                    xk, yk, alphak, beam_weights_func
                )
                for sti_loop in range(num_sti)
            ]
        )

        # the sum of two dot products is equivalent to the inner product bewteen
        # Phi_ri_band.T and Phi_ri, where Phi_ri = [Phi_cpx_band.real \\ Phi_cpx_band.imag]
        crb_lst.append(
            0.5 * sigma_noise ** 2 *
            linalg.solve(np.dot(Phi_cpx_band.real.T, Phi_cpx_band.real) +
                         np.dot(Phi_cpx_band.imag.T, Phi_cpx_band.imag),
                         eye_3K)
        )
    return crb_lst


def planar_calcuate_crb_inner(p_x_loop, p_y_loop, xk, yk, alphak, beam_weights_func):
    num_antenna, num_station = p_x_loop.shape
    p_x_station_outer = np.reshape(p_x_loop, (-1, 1), order='F')
    p_y_station_outer = np.reshape(p_y_loop, (-1, 1), order='F')

    p_x_station_inner = np.reshape(p_x_loop, (1, -1), order='F')
    p_y_station_inner = np.reshape(p_y_loop, (1, -1), order='F')

    baseline_x = ne.evaluate('p_x_station_outer - p_x_station_inner')
    baseline_y = ne.evaluate('p_y_station_outer - p_y_station_inner')

    # identify antenna pairs that are working;
    # also remove the cross-correlations between antennas within the same station
    valid_idx = np.logical_not(
        np.any(np.dstack((np.isnan(baseline_x), np.isnan(baseline_y),
                          np.kron(np.eye(num_station),
                                  np.ones((num_antenna, num_antenna))).astype(bool)
                          )),
               axis=2)
    )

    # cross beam shape
    cross_beamShape = ne.evaluate('where(valid_idx, local_val, 0)',
                                  local_dict={'local_val':
                                                  beam_weights_func(baseline_x, baseline_y) / num_antenna,
                                              'valid_idx': valid_idx}
                                  )

    baseline_x = ne.evaluate('where(valid_idx, baseline_x, 0)')
    baseline_y = ne.evaluate('where(valid_idx, baseline_y, 0)')

    xk = np.reshape(xk, (1, 1, -1), order='F')
    yk = np.reshape(yk, (1, 1, -1), order='F')
    alphak = np.reshape(alphak, (1, 1, -1), order='F')

    # block views
    cross_beamShape = view_as_blocks(cross_beamShape, (num_antenna, num_antenna))
    baseline_x = view_as_blocks(baseline_x, (num_antenna, num_antenna))
    baseline_y = view_as_blocks(baseline_y, (num_antenna, num_antenna))

    # derivative w.r.t. xk
    effective_rows = [
        [
            np.tensordot(
                cross_beamShape[station_count1, station_count2],
                ne.evaluate(
                    '-alphak * 1j * baseline_x_count * '
                    '(cos(xk * baseline_x_count + yk * baseline_y_count) - '
                    '1j * sin(xk * baseline_x_count + yk * baseline_y_count))',
                    local_dict={
                        'alphak': alphak,
                        'baseline_x_count':
                            baseline_x[station_count1, station_count2][:, :, np.newaxis],
                        'baseline_y_count':
                            baseline_y[station_count1, station_count2][:, :, np.newaxis],
                        'xk': xk,
                        'yk': yk
                    }
                ),
                axes=([0, 1], [0, 1])
            )
            for station_count2 in range(station_count1)  # exploit Hermitian symmetry
        ]
        for station_count1 in range(num_station)
    ]

    mtx_blk_dx = np.empty((num_station * (num_station - 1), xk.size), dtype=complex, order='C')
    count = 0
    for station_count1 in range(num_station):
        for station_count2 in range(num_station):
            if station_count2 > station_count1:
                mtx_blk_dx[count, :] = np.conj(effective_rows[station_count2][station_count1])
                count += 1
            elif station_count2 < station_count1:
                mtx_blk_dx[count, :] = effective_rows[station_count1][station_count2]
                count += 1

    del effective_rows

    # derivative w.r.t. yk
    effective_rows = [
        [
            np.tensordot(
                cross_beamShape[station_count1, station_count2],
                ne.evaluate(
                    '-alphak * 1j * baseline_y_count * '
                    '(cos(xk * baseline_x_count + yk * baseline_y_count) - '
                    '1j * sin(xk * baseline_x_count + yk * baseline_y_count))',
                    local_dict={
                        'alphak': alphak,
                        'baseline_x_count':
                            baseline_x[station_count1, station_count2][:, :, np.newaxis],
                        'baseline_y_count':
                            baseline_y[station_count1, station_count2][:, :, np.newaxis],
                        'xk': xk,
                        'yk': yk
                    }
                ),
                axes=([0, 1], [0, 1])
            )
            for station_count2 in range(station_count1)  # exploit Hermitian symmetry
        ]
        for station_count1 in range(num_station)
    ]

    mtx_blk_dy = np.empty((num_station * (num_station - 1), xk.size), dtype=complex, order='C')
    count = 0
    for station_count1 in range(num_station):
        for station_count2 in range(num_station):
            if station_count2 > station_count1:
                mtx_blk_dy[count, :] = np.conj(effective_rows[station_count2][station_count1])
                count += 1
            elif station_count2 < station_count1:
                mtx_blk_dy[count, :] = effective_rows[station_count1][station_count2]
                count += 1

    del effective_rows

    # derivative w.r.t. alphak
    effective_rows = [
        [
            np.tensordot(
                cross_beamShape[station_count1, station_count2],
                ne.evaluate(
                    '(cos(xk * baseline_x_count + yk * baseline_y_count) - '
                    '1j * sin(xk * baseline_x_count + yk * baseline_y_count))',
                    local_dict={
                        'baseline_x_count':
                            baseline_x[station_count1, station_count2][:, :, np.newaxis],
                        'baseline_y_count':
                            baseline_y[station_count1, station_count2][:, :, np.newaxis],
                        'xk': xk,
                        'yk': yk
                    }
                ),
                axes=([0, 1], [0, 1])
            )
            for station_count2 in range(station_count1)  # exploit Hermitian symmetry
        ]
        for station_count1 in range(num_station)
    ]

    mtx_blk_dalpha = np.empty((num_station * (num_station - 1), xk.size), dtype=complex, order='C')
    count = 0
    for station_count1 in range(num_station):
        for station_count2 in range(num_station):
            if station_count2 > station_count1:
                mtx_blk_dalpha[count, :] = np.conj(effective_rows[station_count2][station_count1])
                count += 1
            elif station_count2 < station_count1:
                mtx_blk_dalpha[count, :] = effective_rows[station_count1][station_count2]
                count += 1

    del effective_rows

    return np.hstack((mtx_blk_dx, mtx_blk_dy, mtx_blk_dalpha))
