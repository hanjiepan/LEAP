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
import warnings
import numpy as np
import numexpr as ne
import scipy as sp
import scipy.special
from scipy import linalg
import time
from joblib import Parallel, delayed
from functools import partial
from skimage.util.shape import view_as_blocks
# import os
# os.environ["THEANO_FLAGS"] = "device=gpu1"
import theano
from theano import tensor as TT
from utils import sph2cart, cpx_mtx2real, periodic_sinc


def beamforming_func(baseline_x, baseline_y, baseline_z, strategy='matched', **kwargs):
    """
    compute beamforming weights (cross-beamshape)
    :param baseline_x: baseline along x-axis
    :param baseline_y: baseline along y-axis
    :param baseline_z: baseline along z-axis
    :param strategy: beamforming strategy, can be 'matched' or ...
    :return:
    """
    if strategy.lower() == 'matched':
        if 'x0' in kwargs:
            x0 = kwargs['x0']
        else:
            x0 = 0

        if 'y0' in kwargs:
            y0 = kwargs['y0']
        else:
            y0 = 0

        if 'z0' in kwargs:
            z0 = kwargs['z0']
        else:
            z0 = 1

        cross_beam = np.exp(1j * (x0 * baseline_x + y0 * baseline_y + z0 * baseline_z))

    else:
        # TODO: incorporate other beamforming strategies later
        raise NameError("Unrecognised beamforming strategy.")

    return cross_beam


def planar_beamforming_func(baseline_x, baseline_y, strategy='matched', **kwargs):
    """
    compute beamforming weights (cross-beamshape)
    :param baseline_x: baseline along x-axis
    :param baseline_y: baseline along y-axis
    :param strategy: beamforming strategy, can be 'matched' or ...
    :return:
    """
    if strategy.lower() == 'matched':
        if 'x0' in kwargs:
            x0 = kwargs['x0']
        else:
            x0 = 0

        if 'y0' in kwargs:
            y0 = kwargs['y0']
        else:
            y0 = 0

        # cross_beam = np.exp(1j * (x0 * baseline_x + y0 * baseline_y))
        # cross_beam = ne.evaluate('exp(1j * (x0 * baseline_x + y0 * baseline_y))')
        cross_beam = ne.evaluate('cos(phase) + 1j * sin(phase)',
                                 local_dict={'phase': x0 * baseline_x + y0 * baseline_y})
    else:
        # TODO: incorporate other beamforming strategies later
        raise NameError("Unrecognised beamforming strategy.")

    return cross_beam


def planar_mtx_fri2visibility_beamforming(mtx_freq2visibility, symb=False, real_value=False, **kwargs):
    """
    build the linear transformation matrix that links the FRI sequence to the visibilities.
    Here when symb is true, then we exploit the fact that the FRI sequence is Hermitian symmetric.
    A real-valued representation of the linear mapping is returned (regarless of symb).
    :param expand_b_mtx: the expansion matrix, which maps the real-value representation of a Hermitian
            symmetric vector as
                [real_part_of_the_first_half (including zero)
                imaginary_part_of_the_first_half (excluding zero)],
            to the full range of vector
                [real_part_of_the_vector
                imaginary_part_of_the_vector]
    :param mtx_freq2visibility: a linear mapping from the Fourier transform to the visibilities
    :param symb: whether to exploit the symmetry in the FRI sequence or not.
    :return:
    """
    # parse inputs
    if 'expand_b_mtx' in kwargs:
        expand_b_mtx = kwargs['expand_b_mtx']
    else:
        symb = False

    if symb:
        real_value = True

    num_bands = len(mtx_freq2visibility)
    if not real_value:
        return mtx_freq2visibility
    elif symb:
        return [np.dot(cpx_mtx2real(mtx_freq2visibility[band_count]), expand_b_mtx)
                for band_count in range(num_bands)]
    else:
        return [cpx_mtx2real(mtx_freq2visibility(band_count))
                for band_count in range(num_bands)]


def planar_mtx_freq2visibility_beamforming(p_x, p_y, M, N, tau_inter_x, tau_inter_y,
                                           beam_weights_func=planar_beamforming_func,
                                           num_station=1, num_sti=1, num_bands=1,
                                           backend='cpu', theano_func=None):
    """
    build the linear transformation matrix that links the Fourier transform on a uniform
    grid, which is arranged column by column, with the measured visibilities.
    :param p_x: antennas' x coordinates
    :param p_y: antennas' y coordinates
    :param M: [M,N] the equivalence of "bandwidth" in time domain (because of duality)
    :param N: [M,N] the equivalence of "bandwidth" in time domain (because of duality)
    :param tau_inter_x: the Fourier domain interpolation step-size is 2 pi / tau_inter
    :param tau_inter_y: the Fourier domain interpolation step-size is 2 pi / tau_inter
    :param beam_weights_func: a function that computes weights associated with the
                beamforming strategy
    :param num_station: total number of stations
    :param num_sti: total number of short-time-intervals (STI)
    :param num_bands: total number of subbands
    :return:
    """
    # we assume the antenna coordinates are arranged in a matrix form with the axises
    # correspond to:
    # (antenna_within_one_station, station_count, sti_index, subband_index)
    p_x = np.reshape(p_x, (-1, num_station, num_sti, num_bands), order='F')
    p_y = np.reshape(p_y, (-1, num_station, num_sti, num_bands), order='F')

    m_limit = int(np.floor(M * tau_inter_x // 2))
    n_limit = int(np.floor(N * tau_inter_y // 2))
    m_len = 2 * m_limit + 1
    n_len = 2 * n_limit + 1

    m_grid, n_grid = np.meshgrid(np.arange(-m_limit, m_limit + 1, step=1, dtype=int),
                                 np.arange(-n_limit, n_limit + 1, step=1, dtype=int))
    m_grid = np.reshape(m_grid, (1, -1), order='F')
    n_grid = np.reshape(n_grid, (1, -1), order='F')

    # a list (over different subbands) of the linear mapping
    G_lst = [
        np.vstack([
            planar_mtx_fri2visibility_beamforming_inner(
                p_x[:, :, sti_loop, band_count],
                p_y[:, :, sti_loop, band_count],
                M, N, tau_inter_x, tau_inter_y,
                m_grid, n_grid, m_len, n_len,
                beam_weights_func,
                backend=backend, theano_func=theano_func
            )
            for sti_loop in range(num_sti)
        ])
        for band_count in range(num_bands)
    ]
    return G_lst


# def planar_mtx_fri2visibility_beamforming_inner(p_x_loop, p_y_loop, M, N,
#                                                 tau_inter_x, tau_inter_y,
#                                                 m_grid, n_grid, m_len, n_len,
#                                                 beam_weights_func):
#     """
#     Inner loop to build the linear mapping from an FRI sequence to the measured visibilities.
#     :param p_x_loop:
#     :param p_y_loop:
#     :param M:
#     :param N:
#     :param tau_inter_x:
#     :param tau_inter_y:
#     :param m_grid:
#     :param n_grid:
#     :param beam_weights_func:
#     :return:
#     """
#     num_antenna, num_station = p_x_loop.shape
#
#     # pre-compute a few entries
#     m_taux = M * tau_inter_x
#     n_tauy = N * tau_inter_y
#     two_pi_m = 2 * np.pi * m_grid
#     two_pi_n = 2 * np.pi * n_grid
#
#     G_blk = np.empty((num_station * (num_station - 1), m_len * n_len), dtype=complex, order='C')
#
#     count_G = 0
#     for station_count1 in range(num_station):
#         # add axis in order to use broadcasting
#         p_x_station1 = p_x_loop[:, station_count1]
#         p_y_station1 = p_y_loop[:, station_count1]
#         # because not all antennas are working, we need to eliminate those from the
#         # calculation. Here non-working antennas has array coordinates nan.
#         failed_antenna_station1 = np.logical_or(np.isnan(p_x_station1),
#                                                 np.isnan(p_y_station1))
#         p_x_station1 = p_x_station1[~failed_antenna_station1][:, np.newaxis]
#         p_y_station1 = p_y_station1[~failed_antenna_station1][:, np.newaxis]
#         for station_count2 in range(num_station):
#             if station_count2 != station_count1:
#                 p_x_station2 = p_x_loop[:, station_count2]
#                 p_y_station2 = p_y_loop[:, station_count2]
#                 # because not all antennas are working, we need to eliminate those from the
#                 # calculation. Here non-working antennas has array coordinates nan.
#                 failed_antenna_station2 = np.logical_or(np.isnan(p_x_station2),
#                                                         np.isnan(p_y_station2))
#                 p_x_station2 = p_x_station2[~failed_antenna_station2][np.newaxis]
#                 p_y_station2 = p_y_station2[~failed_antenna_station2][np.newaxis]
#
#                 # compute baselines
#                 # use C-ordering here as we want the data to be the following order:
#                 # one antenna coordinate in station 1 v.s. all the antennas in station 2;
#                 # then another antenna coordinate in station 1 v.s. all the antennas in station 2
#                 baseline_x = (p_x_station1 - p_x_station2).flatten('C')[:, np.newaxis]
#                 baseline_y = (p_y_station1 - p_y_station2).flatten('C')[:, np.newaxis]
#
#                 # weights from beamforming
#                 cross_beamShape = beam_weights_func(baseline_x, baseline_y) / num_antenna
#
#                 freq_x = 0.5 * (tau_inter_x * baseline_x - two_pi_m)
#                 freq_y = 0.5 * (tau_inter_y * baseline_y - two_pi_n)
#
#                 G_blk[count_G, :] = np.dot(cross_beamShape.T,
#                                            periodic_sinc(freq_x, m_taux) *
#                                            periodic_sinc(freq_y, n_tauy)).squeeze()
#                 count_G += 1
#
#     return G_blk


def planar_mtx_fri2visibility_beamforming_inner(p_x_loop, p_y_loop, M, N,
                                                tau_inter_x, tau_inter_y,
                                                m_grid, n_grid, m_len, n_len,
                                                beam_weights_func, backend='cpu',
                                                theano_func=None):
    num_antenna, num_station = p_x_loop.shape

    # re-ordering indices for Hermitian symmetric entries
    reordering_ind = np.arange(m_len * n_len, step=1, dtype=int)[::-1]

    # pre-compute a few entries
    m_taux = M * tau_inter_x
    n_tauy = N * tau_inter_y

    # reshape to use broadcasting
    m_grid = np.reshape(m_grid, (1, 1, -1), order='F')
    n_grid = np.reshape(n_grid, (1, 1, -1), order='F')
    two_pi_m = 2 * np.pi * m_grid
    two_pi_n = 2 * np.pi * n_grid

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

    # block views
    cross_beamShape = view_as_blocks(cross_beamShape, (num_antenna, num_antenna))
    baseline_x = view_as_blocks(baseline_x, (num_antenna, num_antenna))
    baseline_y = view_as_blocks(baseline_y, (num_antenna, num_antenna))

    if backend == 'cpu':
        effective_rows = [
            [
                np.tensordot(
                    cross_beamShape[station_count1, station_count2],
                    periodic_sinc(
                        0.5 * (tau_inter_x *
                               baseline_x[station_count1, station_count2][:, :, np.newaxis] -
                               two_pi_m
                               ),
                        m_taux) *
                    periodic_sinc(
                        0.5 * (tau_inter_y *
                               baseline_y[station_count1, station_count2][:, :, np.newaxis] -
                               two_pi_n
                               ),
                        n_tauy
                    ),
                    axes=([0, 1], [0, 1])
                )
                for station_count2 in range(station_count1)
            ]
            for station_count1 in range(num_station)
        ]

        G_blk = np.empty((num_station * (num_station - 1), m_len * n_len), dtype=complex, order='C')
        count = 0
        for station_count1 in range(num_station):
            for station_count2 in range(num_station):
                if station_count2 > station_count1:
                    # because periodic sinc is real-valued, we can take conj for the whole row
                    G_blk[count, :] = np.conj(effective_rows[station_count2][station_count1][reordering_ind])
                    count += 1
                elif station_count2 < station_count1:
                    G_blk[count, :] = effective_rows[station_count1][station_count2]
                    count += 1
    else:
        # theano version
        cross_beamShape = np.reshape(cross_beamShape, (-1, num_antenna, num_antenna), order='C')
        baseline_x = np.reshape(baseline_x, (-1, num_antenna, num_antenna), order='C')
        baseline_y = np.reshape(baseline_y, (-1, num_antenna, num_antenna), order='C')

        # indices of the lower triangle (excluding the diagonal)
        lower_tri_idx = np.tril(np.reshape(np.arange(num_station ** 2, dtype=int),
                                           (num_station, num_station), order='C'),
                                k=-1)
        lower_tri_idx = np.extract(lower_tri_idx > 0, lower_tri_idx)

        # indices of the upper triangle (excluding the diagonal)
        upper_tri_idx = np.triu(np.reshape(np.arange(num_station ** 2, dtype=int),
                                           (num_station, num_station), order='C'),
                                k=1).T
        upper_tri_idx = np.extract(upper_tri_idx > 0, upper_tri_idx)

        # indices of all entries but the diagonal
        off_diag_idx_all = (1 - np.eye(num_station, dtype=int)) * \
                           np.reshape(np.arange(num_station ** 2, dtype=int),
                                      (num_station, num_station), order='C')
        off_diag_idx_all = np.extract(off_diag_idx_all > 0, off_diag_idx_all)

        if theano_func is None:
            theano_func = compile_theano_func_build_G_mtx()

        # partition m/n-grid into blocks of length max_mn_blk
        max_mn_blk = 500
        num_mn_blk = m_grid.size // max_mn_blk
        mn_blk_seq = [max_mn_blk] * num_mn_blk
        mn_blk_seq.append(m_grid.size - max_mn_blk * num_mn_blk)

        effective_rows_r = []
        effective_rows_i = []

        mn_bg_idx = 0
        for mn_blk_loop in mn_blk_seq:
            effective_rows_r_loop, effective_rows_i_loop = theano_func(
                tau_inter_x, tau_inter_y, M, N,
                m_grid.squeeze()[mn_bg_idx:mn_bg_idx + mn_blk_loop],
                n_grid.squeeze()[mn_bg_idx:mn_bg_idx + mn_blk_loop],
                baseline_x[lower_tri_idx], baseline_y[lower_tri_idx],
                cross_beamShape[lower_tri_idx].real, cross_beamShape[lower_tri_idx].imag
            )
            effective_rows_r.append(effective_rows_r_loop)
            effective_rows_i.append(effective_rows_i_loop)
            mn_bg_idx += mn_blk_loop

        effective_rows = np.column_stack(effective_rows_r) + \
                         1j * np.column_stack(effective_rows_i)

        G_blk = np.empty((num_station ** 2, m_len * n_len), dtype=complex, order='C')
        G_blk[lower_tri_idx, :] = effective_rows
        G_blk[upper_tri_idx, :] = np.conj(effective_rows[:, reordering_ind])
        G_blk = G_blk[off_diag_idx_all, :]

    return G_blk


def compile_theano_func_build_G_mtx():
    tau_inter_x, tau_inter_y = TT.scalar('tau_inter_x'), TT.scalar('tau_inter_y')
    M, N = TT.scalar('M'), TT.scalar('N')
    m_grid, n_grid = TT.vector('m_grid'), TT.vector('n_grid')
    cross_beamShape_r, cross_beamShape_i = \
        TT.tensor3('cross_beamShape_r'), TT.tensor3('cross_beamShape_i')
    baseline_x, baseline_y = TT.tensor3('baseline_x'), TT.tensor3('baseline_y')
    pi = TT.constant(np.pi)

    def theano_periodic_sinc(in_sig, bandwidth):
        eps = TT.constant(1e-10)
        denominator = TT.mul(TT.sin(TT.true_div(in_sig, bandwidth)), bandwidth)
        idx_modi = TT.lt(TT.abs_(denominator), eps)
        numerator = TT.switch(idx_modi, TT.cos(in_sig), TT.sin(in_sig))
        denominator = TT.switch(idx_modi, TT.cos(TT.true_div(in_sig, bandwidth)), denominator)
        return TT.true_div(numerator, denominator)

    # def theano_periodic_sinc(in_sig, bandwidth):
    #     eps = TT.constant(1e-10)
    #     numerator = TT.sin(in_sig)
    #     denominator = TT.mul(TT.sin(TT.true_div(in_sig, bandwidth)), bandwidth)
    #     out0 = TT.true_div(numerator, denominator)
    #     out1 = TT.true_div(TT.cos(in_sig), TT.cos(TT.true_div(in_sig, bandwidth)))
    #     idx_modi = TT.lt(TT.abs_(denominator), eps)
    #     out = TT.switch(idx_modi, out1, out0)
    #     return out

    # define the function
    def f_inner(cross_beamShape_r, cross_beamShape_i, baseline_x, baseline_y,
                tau_inter_x, tau_inter_y, m_grid, n_grid, M, N):
        periodic_sinc_2d = \
            TT.mul(
                theano_periodic_sinc(
                    0.5 * (TT.shape_padright(tau_inter_x * baseline_x, n_ones=1) -
                           2 * pi * TT.shape_padleft(m_grid, n_ones=2)),
                    M * tau_inter_x
                ),
                theano_periodic_sinc(
                    0.5 * (TT.shape_padright(tau_inter_y * baseline_y, n_ones=1) -
                           2 * pi * TT.shape_padleft(n_grid, n_ones=2)),
                    N * tau_inter_y
                )
            )
        G_mtx_r = TT.tensordot(cross_beamShape_r, periodic_sinc_2d, axes=[[0, 1], [0, 1]])
        G_mtx_i = TT.tensordot(cross_beamShape_i, periodic_sinc_2d, axes=[[0, 1], [0, 1]])

        return G_mtx_r, G_mtx_i

    G_mtx_r, G_mtx_i = theano.map(
        fn=f_inner,
        sequences=(cross_beamShape_r, cross_beamShape_i, baseline_x, baseline_y),
        non_sequences=(tau_inter_x, tau_inter_y, m_grid, n_grid, M, N)
    )[0]

    # compile the function
    func = theano.function([tau_inter_x, tau_inter_y, M, N, m_grid, n_grid,
                            baseline_x, baseline_y,
                            cross_beamShape_r, cross_beamShape_i],
                           [G_mtx_r, G_mtx_i],
                           allow_input_downcast=True)
    return func


def planar_build_mtx_amp_ri_beamforming(p_x_band, p_y_band, xk, yk,
                                        beam_weights_func=planar_beamforming_func,
                                        num_station=1, num_sti=1, backend='cpu'):
    """
    build the matrix that links the Dirac deltas' amplitudes to the visibility measurements
    for each sub-band.
    :param p_x_band: antenna location (x-axis)
    :param p_y_band: antenna location (y-axis)
    :param xk: horizontal location of the Dirac deltas
    :param yk: vertical location of the Dirac deltas
    :param beam_weights_func: beamforming function
    :param num_station: number of stations
    :param num_sti: number of STIs
    :return:
    """
    mtx = planar_build_mtx_amp_beamforming_cpu(p_x_band, p_y_band, xk, yk,
                                               beam_weights_func=beam_weights_func,
                                               num_station=num_station, num_sti=num_sti)
    return np.vstack((mtx.real, mtx.imag))


def planar_build_mtx_amp_beamforming(p_x_band, p_y_band, xk, yk,
                                     beam_weights_func=planar_beamforming_func,
                                     theano_func=None, backend='cpu',
                                     large_k_limit=500):
    if backend == 'cpu':
        num_station, num_sti = p_x_band.shape[1:]
        return planar_build_mtx_amp_beamforming_cpu(
            p_x_band, p_y_band, xk, yk,
            beam_weights_func=planar_beamforming_func,
            num_station=num_station, num_sti=num_sti)
    elif backend == 'gpu':
        if xk.size > large_k_limit:
            return planar_build_mtx_amp_beamforming_theano(
                p_x_band, p_y_band, xk, yk,
                beam_weights_func=beam_weights_func,
                theano_func=theano_func, max_sti_blk=1,
                max_k_blk=500)
        else:
            return planar_build_mtx_amp_beamforming_theano(
                p_x_band, p_y_band, xk, yk,
                beam_weights_func=beam_weights_func,
                theano_func=theano_func, max_sti_blk=25,
                max_k_blk=20)
    else:
        RuntimeError('Unknown backend option: {}'.format(backend))


def planar_build_mtx_amp_beamforming_theano(p_x_band, p_y_band, xk, yk,
                                            beam_weights_func=planar_beamforming_func,
                                            theano_func=None, max_sti_blk=25,
                                            max_k_blk=500, **kwargs):
    num_antenna, num_station, num_sti = p_x_band.shape
    cross_beamShape_all = []
    baseline_x_all = []
    baseline_y_all = []
    for sti_count in range(num_sti):
        p_x_loop = p_x_band[:, :, sti_count]
        p_y_loop = p_y_band[:, :, sti_count]

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

        # append as list
        cross_beamShape_all.append(cross_beamShape[np.newaxis])
        baseline_x_all.append(baseline_x[np.newaxis])
        baseline_y_all.append(baseline_y[np.newaxis])

    cross_beamShape_all = np.concatenate(cross_beamShape_all, axis=0)
    baseline_x_all = np.concatenate(baseline_x_all, axis=0)
    baseline_y_all = np.concatenate(baseline_y_all, axis=0)

    xk = np.reshape(xk, (1, 1, -1), order='F')
    yk = np.reshape(yk, (1, 1, -1), order='F')

    # block views
    cross_beamShape_all = view_as_blocks(cross_beamShape_all, (num_sti, num_antenna, num_antenna))
    baseline_x_all = view_as_blocks(baseline_x_all, (num_sti, num_antenna, num_antenna))
    baseline_y_all = view_as_blocks(baseline_y_all, (num_sti, num_antenna, num_antenna))

    # theano version
    cross_beamShape_all = \
        np.reshape(cross_beamShape_all, (-1, num_sti, num_antenna, num_antenna), order='C')
    baseline_x_all = \
        np.reshape(baseline_x_all, (-1, num_sti, num_antenna, num_antenna), order='C')
    baseline_y_all = \
        np.reshape(baseline_y_all, (-1, num_sti, num_antenna, num_antenna), order='C')

    # indices of the lower triangle (excluding the diagonal)
    lower_tri_idx = np.tril(np.reshape(np.arange(num_station ** 2, dtype=int),
                                       (num_station, num_station), order='C'),
                            k=-1)
    lower_tri_idx = np.extract(lower_tri_idx > 0, lower_tri_idx)

    # indices of the upper triangle (excluding the diagonal)
    upper_tri_idx = np.triu(np.reshape(np.arange(num_station ** 2, dtype=int),
                                       (num_station, num_station), order='C'),
                            k=1).T
    upper_tri_idx = np.extract(upper_tri_idx > 0, upper_tri_idx)

    # indices of all entries but the diagonal
    off_diag_idx_all = (1 - np.eye(num_station, dtype=int)) * \
                       np.reshape(np.arange(num_station ** 2, dtype=int),
                                  (num_station, num_station), order='C')
    off_diag_idx_all = np.extract(off_diag_idx_all > 0, off_diag_idx_all)

    if theano_func is None:
        theano_func = compile_theano_func_build_amp_mtx()

    # partition STIs into blocks of length max_sti_blk
    num_sti_blk = num_sti // max_sti_blk
    sti_blk_seq = [max_sti_blk] * num_sti_blk
    sti_blk_seq.append(num_sti - max_sti_blk * num_sti_blk)

    # partition K number of Diracs into blocks of length max_k_blk
    num_k_blk = xk.size // max_k_blk
    k_blk_seq = [max_k_blk] * num_k_blk
    k_blk_seq.append(xk.size - max_k_blk * num_k_blk)

    effective_rows_r = []
    effective_rows_i = []

    sti_bg_idx = 0
    for sti_blk_loop in sti_blk_seq:
        effective_rows_r_loop = []
        effective_rows_i_loop = []

        k_bg_idx = 0
        for k_blk_loop in k_blk_seq:
            effective_rows_r_k_loop, effective_rows_i_k_loop = \
                theano_func(
                    xk.squeeze()[k_bg_idx:k_bg_idx + k_blk_loop]
                    if xk.size > 1 else np.array([xk.squeeze()]),
                    yk.squeeze()[k_bg_idx:k_bg_idx + k_blk_loop]
                    if yk.size > 1 else np.array([yk.squeeze()]),
                    baseline_x_all[lower_tri_idx].squeeze()[:, sti_bg_idx:sti_bg_idx + sti_blk_loop, :, :],
                    baseline_y_all[lower_tri_idx].squeeze()[:, sti_bg_idx:sti_bg_idx + sti_blk_loop, :, :],
                    cross_beamShape_all[lower_tri_idx][:, sti_bg_idx:sti_bg_idx + sti_blk_loop, :, :].real,
                    cross_beamShape_all[lower_tri_idx][:, sti_bg_idx:sti_bg_idx + sti_blk_loop, :, :].imag
                )
            effective_rows_r_loop.append(effective_rows_r_k_loop)
            effective_rows_i_loop.append(effective_rows_i_k_loop)
            k_bg_idx += k_blk_loop

        effective_rows_r.append(np.concatenate(effective_rows_r_loop, axis=-1))
        effective_rows_i.append(np.concatenate(effective_rows_i_loop, axis=-1))
        sti_bg_idx += sti_blk_loop

    effective_rows = np.concatenate(effective_rows_r, axis=1) + \
                     1j * np.concatenate(effective_rows_i, axis=1)

    mtx_blk = np.empty((num_station * num_station, num_sti, xk.size),
                       dtype=complex, order='C')
    mtx_blk[lower_tri_idx, :, :] = effective_rows
    mtx_blk[upper_tri_idx, :, :] = np.conj(effective_rows)
    mtx_blk = mtx_blk[off_diag_idx_all, :, :]

    mtx_blk = np.vstack([mtx_blk[:, sti_count, :] for sti_count in range(num_sti)])

    return mtx_blk


def compile_theano_func_build_amp_mtx():
    xk, yk = TT.vector('xk'), TT.vector('yk')
    cross_beamShape_r, cross_beamShape_i = \
        TT.tensor4('cross_beamShape_r'), TT.tensor4('cross_beamShape_i')
    baseline_x, baseline_y = TT.tensor4('baseline_x'), TT.tensor4('baseline_y')

    # define the function
    def f_inner(cross_beamShape_r, cross_beamShape_i, baseline_x, baseline_y, xk, yk):
        phase = TT.mul(TT.shape_padleft(xk, n_ones=3), TT.shape_padright(baseline_x, n_ones=1)) + \
                TT.mul(TT.shape_padleft(yk, n_ones=3), TT.shape_padright(baseline_y, n_ones=1))
        cos_phase, sin_phase = TT.cos(phase), TT.sin(phase)

        beamforming_weight_r = \
            TT.batched_tensordot(cos_phase, cross_beamShape_r,
                                 axes=[[1, 2], [1, 2]]) + \
            TT.batched_tensordot(sin_phase, cross_beamShape_i,
                                 axes=[[1, 2], [1, 2]])
        beamforming_weight_i = \
            TT.batched_tensordot(cos_phase, cross_beamShape_i,
                                 axes=[[1, 2], [1, 2]]) - \
            TT.batched_tensordot(sin_phase, cross_beamShape_r,
                                 axes=[[1, 2], [1, 2]])

        return beamforming_weight_r, beamforming_weight_i

    beamforming_mtx_r, beamforming_mtx_i = theano.map(
        fn=f_inner,
        sequences=[cross_beamShape_r, cross_beamShape_i, baseline_x, baseline_y],
        non_sequences=[xk, yk])[0]

    # compile the function
    func = theano.function([xk, yk, baseline_x, baseline_y,
                            cross_beamShape_r, cross_beamShape_i],
                           [beamforming_mtx_r, beamforming_mtx_i],
                           allow_input_downcast=True)

    return func


def planar_build_mtx_amp_beamforming_cpu(p_x_band, p_y_band, xk, yk,
                                         beam_weights_func=planar_beamforming_func,
                                         num_station=1, num_sti=1):
    """
    build the matrix that links the Dirac deltas' amplitudes to the visibility measurements
    for each sub-band.
    :param p_x_band: antenna location (x-axis)
    :param p_y_band: antenna location (y-axis)
    :param xk: horizontal location of the Dirac deltas
    :param yk: vertical location of the Dirac deltas
    :param beam_weights_func: beamforming function
    :param num_station: number of stations
    :param num_sti: number of STIs
    :param backend: either 'cpu' or 'gpu'
    :return:
    """
    p_x_band = np.reshape(p_x_band, (-1, num_station, num_sti), order='F')
    p_y_band = np.reshape(p_y_band, (-1, num_station, num_sti), order='F')

    mtx = np.vstack(
        [
            planar_build_mtx_amp_beamforming_inner(
                p_x_band[:, :, sti_loop], p_y_band[:, :, sti_loop], xk, yk,
                beam_weights_func)
            for sti_loop in range(num_sti)
        ]
    )
    return mtx


# def planar_build_mtx_amp_beamforming_inner(p_x_loop, p_y_loop, xk, yk, beam_weights_func):
#     num_antenna, num_station = p_x_loop.shape
#     K = xk.size
#
#     mtx = np.zeros((num_station * (num_station - 1), K), dtype=complex, order='C')
#     count = 0
#     for station_count1 in range(num_station):
#         # add axis in order to use broadcasting
#         p_x_station1 = p_x_loop[:, station_count1]
#         p_y_station1 = p_y_loop[:, station_count1]
#         # because not all antennas are working, we need to eliminate those from the
#         # calculation. Here non-working antennas has array coordinates nan.
#         failed_antenna_station1 = np.logical_or(np.isnan(p_x_station1),
#                                                 np.isnan(p_y_station1))
#         p_x_station1 = p_x_station1[~failed_antenna_station1][:, np.newaxis]
#         p_y_station1 = p_y_station1[~failed_antenna_station1][:, np.newaxis]
#         for station_count2 in range(num_station):
#             if station_count2 != station_count1:
#                 p_x_station2 = p_x_loop[:, station_count2]
#                 p_y_station2 = p_y_loop[:, station_count2]
#                 # because not all antennas are working, we need to eliminate those from the
#                 # calculation. Here non-working antennas has array coordinates nan.
#                 failed_antenna_station2 = np.logical_or(np.isnan(p_x_station2),
#                                                         np.isnan(p_y_station2))
#                 p_x_station2 = p_x_station2[~failed_antenna_station2][np.newaxis]
#                 p_y_station2 = p_y_station2[~failed_antenna_station2][np.newaxis]
#
#                 # compute baselines
#                 baseline_x = (ne.evaluate('p_x_station1 - p_x_station2')).flatten('C')[:, np.newaxis]
#                 baseline_y = (ne.evaluate('p_y_station1 - p_y_station2')).flatten('C')[:, np.newaxis]
#
#                 cross_beamShape = beam_weights_func(baseline_x, baseline_y) / num_antenna
#
#                 mtx[count, :] = np.dot(cross_beamShape.T,
#                                        ne.evaluate('exp(-1j * (xk * baseline_x + yk * baseline_y))')
#                                        ).squeeze()
#                 count += 1
#
#     return mtx


def planar_build_mtx_amp_beamforming_inner(p_x_loop, p_y_loop,
                                           xk, yk, beam_weights_func):
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

    # block views
    cross_beamShape = view_as_blocks(cross_beamShape, (num_antenna, num_antenna))
    baseline_x = view_as_blocks(baseline_x, (num_antenna, num_antenna))
    baseline_y = view_as_blocks(baseline_y, (num_antenna, num_antenna))

    # if backend == 'cpu':
    effective_rows = [
        [
            np.tensordot(
                cross_beamShape[station_count1, station_count2],
                ne.evaluate(
                    'cos(xk * baseline_x_count + yk * baseline_y_count) - '
                    '1j * sin(xk * baseline_x_count + yk * baseline_y_count)',
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

    mtx_blk = np.empty((num_station * (num_station - 1), xk.size), dtype=complex, order='C')
    count = 0
    for station_count1 in range(num_station):
        for station_count2 in range(num_station):
            if station_count2 > station_count1:
                mtx_blk[count, :] = np.conj(effective_rows[station_count2][station_count1])
                count += 1
            elif station_count2 < station_count1:
                mtx_blk[count, :] = effective_rows[station_count1][station_count2]
                count += 1
    # else:
    #     # theano version
    #     cross_beamShape = np.reshape(cross_beamShape, (-1, num_antenna, num_antenna), order='C')
    #     baseline_x = np.reshape(baseline_x, (-1, num_antenna, num_antenna), order='C')
    #     baseline_y = np.reshape(baseline_y, (-1, num_antenna, num_antenna), order='C')
    #
    #     # indices of the lower triangle (excluding the diagonal)
    #     lower_tri_idx = np.tril(np.reshape(np.arange(num_station ** 2, dtype=int),
    #                                        (num_station, num_station), order='C'),
    #                             k=-1)
    #     lower_tri_idx = np.extract(lower_tri_idx > 0, lower_tri_idx)
    #
    #     # indices of the upper triangle (excluding the diagonal)
    #     upper_tri_idx = np.triu(np.reshape(np.arange(num_station ** 2, dtype=int),
    #                                        (num_station, num_station), order='C'),
    #                             k=1).T
    #     upper_tri_idx = np.extract(upper_tri_idx > 0, upper_tri_idx)
    #
    #     # indices of all entries but the diagonal
    #     off_diag_idx_all = (1 - np.eye(num_station, dtype=int)) * \
    #                        np.reshape(np.arange(num_station ** 2, dtype=int),
    #                                   (num_station, num_station), order='C')
    #     off_diag_idx_all = np.extract(off_diag_idx_all > 0, off_diag_idx_all)
    #
    #     func = compile_theano_func_build_amp_mtx()
    #
    #     effective_rows_r, effective_rows_i = func(xk.squeeze(), yk.squeeze(),
    #                                               baseline_x[lower_tri_idx],
    #                                               baseline_y[lower_tri_idx],
    #                                               cross_beamShape[lower_tri_idx].real,
    #                                               cross_beamShape[lower_tri_idx].imag
    #                                               )
    #     effective_rows = effective_rows_r + 1j * effective_rows_i
    #
    #     mtx_blk = np.empty((num_station * num_station, xk.size),
    #                        dtype=complex, order='C')
    #     mtx_blk[lower_tri_idx, :] = effective_rows
    #     mtx_blk[upper_tri_idx, :] = np.conj(effective_rows)
    #     mtx_blk = mtx_blk[off_diag_idx_all, :]

    return mtx_blk


# def compile_theano_func_build_amp_mtx():
#     xk, yk = TT.vector('xk'), TT.vector('yk')
#     cross_beamShape_r, cross_beamShape_i = \
#         TT.tensor3('cross_beamShape_r'), TT.tensor3('cross_beamShape_i')
#     baseline_x, baseline_y = TT.tensor3('baseline_x'), TT.tensor3('baseline_y')
#
#     # define the function
#     def f_inner(cross_beamShape_r, cross_beamShape_i, baseline_x, baseline_y, xk, yk):
#         phase = TT.shape_padleft(xk, n_ones=2) * TT.shape_padright(baseline_x, n_ones=1) + \
#                 TT.shape_padleft(yk, n_ones=2) * TT.shape_padright(baseline_y, n_ones=1)
#
#         cos_phase, sin_phase = TT.cos(phase), TT.sin(phase)
#
#         beamforming_weight_r = \
#             TT.tensordot(cos_phase, cross_beamShape_r, axes=[[0, 1], [0, 1]]) + \
#             TT.tensordot(sin_phase, cross_beamShape_i, axes=[[0, 1], [0, 1]])
#         beamforming_weight_i = \
#             TT.tensordot(cos_phase, cross_beamShape_i, axes=[[0, 1], [0, 1]]) - \
#             TT.tensordot(sin_phase, cross_beamShape_r, axes=[[0, 1], [0, 1]])
#
#         return beamforming_weight_r, beamforming_weight_i
#
#     beamforming_mtx_r, beamforming_mtx_i = theano.map(
#         fn=f_inner,
#         sequences=(cross_beamShape_r, cross_beamShape_i, baseline_x, baseline_y),
#         non_sequences=(xk, yk))[0]
#
#     # compile the function
#     func = theano.function([xk, yk, baseline_x, baseline_y,
#                             cross_beamShape_r, cross_beamShape_i],
#                            [beamforming_mtx_r, beamforming_mtx_i],
#                            allow_input_downcast=True)
#
#     return func


def planar_update_G_beamforming(xk, yk, M, N, tau_inter_x, tau_inter_y,
                                p_x, p_y, mtx_fri2visibility_lst, beam_weights_func,
                                num_station=1, num_sti=1, num_bands=1,
                                theano_func=None, backend='cpu'):
    p_x = np.reshape(p_x, (-1, num_station, num_sti, num_bands), order='F')
    p_y = np.reshape(p_y, (-1, num_station, num_sti, num_bands), order='F')

    m_limit = np.int(np.floor(M * tau_inter_x // 2))
    n_limit = np.int(np.floor(N * tau_inter_y // 2))

    m_grid, n_grid = np.meshgrid(np.arange(-m_limit, m_limit + 1, step=1, dtype=int),
                                 np.arange(-n_limit, n_limit + 1, step=1, dtype=int))

    # reshape to use broadcasting
    m_grid = np.reshape(m_grid, (-1, 1), order='F')
    n_grid = np.reshape(n_grid, (-1, 1), order='F')

    xk = np.reshape(xk, (1, -1), order='F')
    yk = np.reshape(yk, (1, -1), order='F')

    mtx_amp2freq = np.exp(-1j * (xk * 2 * np.pi / tau_inter_x * m_grid +
                                 yk * 2 * np.pi / tau_inter_y * n_grid))

    mtx_fri2amp = linalg.lstsq(mtx_amp2freq,
                               np.eye(mtx_amp2freq.shape[0]))[0]

    G_updated = []
    for band_count in range(num_bands):
        G0_loop = mtx_fri2visibility_lst[band_count]

        mtx_amp2visibility_loop = planar_build_mtx_amp_beamforming(
            p_x[:, :, :, band_count], p_y[:, :, :, band_count],
            xk, yk, beam_weights_func,
            theano_func=theano_func, backend=backend
        )

        high_freq_mapping = np.dot(mtx_amp2visibility_loop, mtx_fri2amp)

        G_updated.append(
            high_freq_mapping +
            G0_loop -
            np.dot(np.dot(G0_loop, mtx_fri2amp.conj().T),
                   linalg.solve(np.dot(mtx_fri2amp, mtx_fri2amp.conj().T),
                                mtx_fri2amp)
                   )
        )
    return G_updated


def planar_update_G_ri_beamforming(xk, yk, M, N, tau_inter_x, tau_inter_y,
                                   p_x, p_y, mtx_fri2visibility_ri_lst, beam_weights_func,
                                   num_station=1, num_sti=1, num_bands=1):
    p_x = np.reshape(p_x, (-1, num_station, num_sti, num_bands), order='F')
    p_y = np.reshape(p_y, (-1, num_station, num_sti, num_bands), order='F')

    m_limit = np.int(np.floor(M * tau_inter_x // 2))
    n_limit = np.int(np.floor(N * tau_inter_y // 2))

    m_grid, n_grid = np.meshgrid(np.arange(-m_limit, m_limit + 1, step=1, dtype=int),
                                 np.arange(-n_limit, n_limit + 1, step=1, dtype=int))

    # reshape to use broadcasting
    half_size = int((2 * m_limit + 1) * (2 * n_limit + 1) // 2 + 1)
    m_grid = np.reshape(m_grid, (-1, 1), order='F')[:half_size]
    n_grid = np.reshape(n_grid, (-1, 1), order='F')[:half_size]

    xk = np.reshape(xk, (1, -1), order='F')
    yk = np.reshape(yk, (1, -1), order='F')

    mtx_amp2freq_cpx = np.exp(-1j * (xk * 2 * np.pi / tau_inter_x * m_grid +
                                     yk * 2 * np.pi / tau_inter_y * n_grid))
    mtx_amp2freq_ri = np.vstack((mtx_amp2freq_cpx.real, mtx_amp2freq_cpx.imag[:-1, :]))

    mtx_fri2amp_half = linalg.lstsq(mtx_amp2freq_ri,
                                    np.eye(mtx_amp2freq_ri.shape[0]))[0]

    G_updated = []
    for band_count in range(num_bands):
        G0_loop = mtx_fri2visibility_ri_lst[band_count]
        mtx_amp2visibility_loop = planar_build_mtx_amp_ri_beamforming(
            p_x[:, :, :, band_count], p_y[:, :, :, band_count],
            xk, yk, beam_weights_func, num_station, num_sti
        )

        high_freq_mapping = np.dot(mtx_amp2visibility_loop, mtx_fri2amp_half)

        G_updated.append(
            high_freq_mapping +
            G0_loop -
            np.dot(np.dot(G0_loop, mtx_fri2amp_half.T),
                   linalg.solve(np.dot(mtx_fri2amp_half, mtx_fri2amp_half.T),
                                mtx_fri2amp_half)
                   )
        )

    return G_updated


def planar_build_inrange_sel_mtx(p_x_normalised, p_y_normalised,
                                 num_station, num_subband, num_sti,
                                 freq_limit_x, freq_limit_y):
    """
    build the matrix that extract the visibility measurements that fall within the assumed
    period of the spectrum.
    :param p_x_normalised: normalised (by the wavelength) antenna locations (x-axis)
    :param p_y_normalised: normalised (by the wavelength) antenna locations (y-axis)
    :param num_station: number of stations
    :param num_subband: number of subbands
    :param num_sti: number of short time intervals (STIs)
    :param freq_limit_x: one period of the spectrum spans from -freq_limit_x to freq_limit_x along x-axis
    :param freq_limit_y: one period of the spectrum spans from -freq_limit_y to freq_limit_y along y-axis
    :return:
    """
    # we use the centroid of the station as an indication
    # compute the centroid
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        p_x_avg = np.nanmean(p_x_normalised, axis=0)
        p_y_avg = np.nanmean(p_y_normalised, axis=0)

    # reshape to use broadcasting
    p_x_avg_outer = np.reshape(p_x_avg, (num_station, 1, num_sti, num_subband), order='F')
    p_x_avg_inner = np.reshape(p_x_avg, (1, num_station, num_sti, num_subband), order='F')
    p_y_avg_outer = np.reshape(p_y_avg, (num_station, 1, num_sti, num_subband), order='F')
    p_y_avg_inner = np.reshape(p_y_avg, (1, num_station, num_sti, num_subband), order='F')

    # compute baselines
    baseline_x_abs = np.abs(p_x_avg_outer - p_x_avg_inner)
    baseline_y_abs = np.abs(p_y_avg_outer - p_y_avg_inner)

    # extract off-diagonal entries
    extract_cond = np.reshape((1 - np.eye(num_station, dtype=int)).astype(bool), (-1, 1), order='C')

    extract_indicator_lst = []
    len_each_sti = num_station * (num_station - 1)
    for band_count in range(num_subband):
        indicator_band = np.empty(len_each_sti * num_sti, dtype=bool)
        for sti_count in range(num_sti):
            indicator_band[sti_count * len_each_sti: (sti_count + 1) * len_each_sti] = \
                np.logical_and(
                    np.extract(extract_cond, baseline_x_abs[:, :, sti_count, band_count]) <= freq_limit_x,
                    np.extract(extract_cond, baseline_y_abs[:, :, sti_count, band_count]) <= freq_limit_y
                )
        extract_indicator_lst.append(indicator_band)

    return extract_indicator_lst
