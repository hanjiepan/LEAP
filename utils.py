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
import numexpr as ne
import numpy as np
import scipy as sp
import scipy.special
import scipy.misc
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy import linalg
from skimage.util import view_as_blocks
from astropy import units
from astropy.coordinates import SkyCoord


def planar_compute_all_baselines(p_x_normalised, p_y_normalised,
                                 num_antenna, num_station, num_subband, num_sti):
    num_antenna_sq = num_antenna ** 2
    all_baselines_x = np.zeros((num_antenna_sq * num_station * (num_station - 1),
                                num_sti, num_subband),
                               dtype=float)

    all_baselines_y = np.zeros((num_antenna_sq * num_station * (num_station - 1),
                                num_sti, num_subband),
                               dtype=float)

    for band_count in range(num_subband):
        for sti_count in range(num_sti):
            v_index = 0
            for station_count1 in range(num_station):
                p_x_station1 = p_x_normalised[:, station_count1, sti_count, band_count][:, np.newaxis]
                p_y_station1 = p_y_normalised[:, station_count1, sti_count, band_count][:, np.newaxis]
                for station_count2 in range(num_station):
                    if station_count1 != station_count2:
                        p_x_station2 = p_x_normalised[:, station_count2, sti_count, band_count][np.newaxis, :]
                        p_y_station2 = p_y_normalised[:, station_count2, sti_count, band_count][np.newaxis, :]

                        baseline_x = (p_x_station1 - p_x_station2).flatten('F')
                        baseline_y = (p_y_station1 - p_y_station2).flatten('F')

                        all_baselines_x[v_index:v_index + num_antenna_sq, sti_count, band_count] = baseline_x
                        all_baselines_y[v_index:v_index + num_antenna_sq, sti_count, band_count] = baseline_y

                        v_index += num_antenna_sq

    # remove nan from the baselines (some antennas are not working)
    all_baselines_x = all_baselines_x[~np.isnan(all_baselines_x)]
    all_baselines_y = all_baselines_y[~np.isnan(all_baselines_y)]

    return all_baselines_x, all_baselines_y


def planar_distance(x_ref, y_ref, x_recon, y_recon):
    """
    Given two arrays of numbers pt_1 and pt_2, pairs the cells that are the
    closest and provides the pairing matrix index: pt_1[index[0, :]] should be as
    close as possible to pt_2[index[2, :]]. The function outputs the average of the
    absolute value of the differences abs(pt_1[index[0, :]]-pt_2[index[1, :]]).
    :param pt_1: vector 1
    :param pt_2: vector 2
    :return: d: minimum distance between d
             index: the permutation matrix
    """
    pt_1 = x_ref + 1j * y_ref
    pt_2 = x_recon + 1j * y_recon
    pt_1 = np.reshape(pt_1, (1, -1), order='F')
    pt_2 = np.reshape(pt_2, (1, -1), order='F')
    N1 = pt_1.size
    N2 = pt_2.size
    diffmat = np.abs(pt_1 - np.reshape(pt_2, (-1, 1), order='F'))
    min_N1_N2 = np.min([N1, N2])
    index = np.zeros((min_N1_N2, 2), dtype=int)
    if min_N1_N2 > 1:
        for k in range(min_N1_N2):
            d2 = np.min(diffmat, axis=0)
            index2 = np.argmin(diffmat, axis=0)
            index1 = np.argmin(d2)
            index2 = index2[index1]
            index[k, :] = [index1, index2]
            diffmat[index2, :] = float('inf')
            diffmat[:, index1] = float('inf')
        d = np.mean(np.abs(pt_1[:, index[:, 0]] - pt_2[:, index[:, 1]]))
    else:
        d = np.min(diffmat)
        index = np.argmin(diffmat)
        if N1 == 1:
            index = np.array([1, index])
        else:
            index = np.array([index, 1])
    return d, index


def planar_gen_dirac_param(K, num_bands=1, positive_amp=True,
                           focus=(0, 0), fov=np.radians(4), save_param=False):
    # Dirac amplitudes
    if positive_amp:
        amplitude = np.abs(np.random.randn(K, num_bands))
    else:
        amplitude = np.random.randn(K, num_bands)

    # horizontal locations
    xk = focus[0] + fov * (np.random.rand(K) - 0.5)

    # vertical locations
    yk = focus[1] + fov * (np.random.rand(K) - 0.5)

    if save_param:
        np.savez('./data/dirac_param_K{0}.npz'.format(K),
                 amplitude=amplitude, xk=xk, yk=yk)

    return amplitude, xk, yk


def gen_dirac_param(K, num_bands=1, positive_amp=True,
                    focus=(0, 0), fov=np.pi, save_param=True):
    """
    generate Dirac parameters
    :param K: number of Dirac
    :param num_bands: number of subbands
    :param positive_amp: whether Dirac has positive amplitudes or not
    :param focus: the Diracs are centred around this focusing point (azimuth, colatitude) (in radian)
    :param fov: width of field of view (in radian)
    :param save_param: whether to save parameters or not
    :return:
    """
    # Dirac amplitudes
    if positive_amp:
        amplitude = np.abs(np.random.randn(K, num_bands))
    else:
        amplitude = np.random.randn(K, num_bands)

    # azimuths
    azimuth = np.mod(fov * np.random.rand(K) + focus[0] - 0.5 * fov, 2 * np.pi)

    # colatitudes
    min_cos_colatitude = np.cos(focus[1] - 0.5 * fov)
    max_cos_colatitude = np.cos(focus[1] + 0.5 * fov)
    cos_range = np.abs(max_cos_colatitude - min_cos_colatitude)
    cos_colatitude = cos_range * (np.random.rand(K) - 0.5) + \
                     0.5 * (min_cos_colatitude + max_cos_colatitude)
    colatitude = np.arccos(cos_colatitude)

    if save_param:
        np.savez('./data/dirac_param_K{0}.npz'.format(K),
                 amplitude=amplitude, azimuth=azimuth, colatitude=colatitude)

    return amplitude, azimuth, colatitude


# def periodic_sinc(t, M):
#     numerator = np.sin(t)
#     denominator = M * np.sin(t / M)
#     idx = np.abs(denominator) < 1e-12
#     t_idx = t[idx]
#     numerator[idx] = np.cos(t_idx)
#     denominator[idx] = np.cos(t_idx / M)
#     return numerator / denominator


def periodic_sinc(t, M):
    numerator = ne.evaluate('sin(t)')
    denominator = ne.evaluate('M * sin(t / M)')
    idx = ne.evaluate('abs(denominator) < 1e-12')
    t_idx = t[idx]
    numerator[idx] = ne.evaluate('cos(t_idx)')
    denominator[idx] = ne.evaluate('cos(t_idx / M)')
    return ne.evaluate('numerator / denominator')


def hermitian_expansion(len_vec):
    """
    create the expansion matrix such that we expand the vector that is Hermitian symmetric.
    The input vector is the concatenation of the real part and imaginary part
    of the vector in the first half.
    :param len_vec: length of the first half for the real part. Hence, it is 1 element more than
                  that for the imaginary part
    :return: D1: expansion matrix for the real part
             D2: expansion matrix for the imaginary part
    """
    D0 = np.eye(len_vec)
    D1 = np.vstack((D0, D0[1::, ::-1]))
    D2 = np.vstack((D0, -D0[1::, ::-1]))
    D2 = D2[:, :-1]
    return D1, D2


def hermitian_expan_mtx(vec_full_len):
    """
    expansion matrix for an annihilating filter of size K + 1
    :param K: number of Dirac. The filter size is K + 1
    :return:
    """
    if vec_full_len % 2 == 0:
        D0 = np.eye(int(vec_full_len // 2))
        D1 = np.vstack((D0, D0[:, ::-1]))
        D2 = np.vstack((D0, -D0[:, ::-1]))
    else:
        D0 = np.eye(int((vec_full_len + 1) // 2))
        D1 = np.vstack((D0, D0[1:, ::-1]))
        D2 = np.vstack((D0, -D0[1:, ::-1]))[:, :-1]
    return D1, D2


def R_mtx_joint(c_row, c_col, L0, L1, mtx_extract_b=None):
    R_loop_row = convmtx2_valid(c_row, L0, L1)
    R_loop_col = convmtx2_valid(c_col, L0, L1)
    if mtx_extract_b is None:
        return np.vstack((R_loop_row, R_loop_col))
    else:
        return np.dot(np.vstack((R_loop_row, R_loop_col)), mtx_extract_b)


def R_mtx_joint_ri(c_row, c_col, L0, L1, expansion_mtx=None, mtx_extract_b=None):
    R_loop_row = convmtx2_valid(c_row, L0, L1)
    R_loop_col = convmtx2_valid(c_col, L0, L1)
    R_cpx = np.vstack((R_loop_row, R_loop_col))
    if expansion_mtx is None and mtx_extract_b is None:
        return cpx_mtx2real(R_cpx)
    elif expansion_mtx is None and mtx_extract_b is not None:
        return np.dot(cpx_mtx2real(R_cpx), mtx_extract_b)
    elif expansion_mtx is not None and mtx_extract_b is not None:
        return np.dot(cpx_mtx2real(R_cpx), np.dot(expansion_mtx, mtx_extract_b))
    else:
        return np.dot(cpx_mtx2real(R_cpx), expansion_mtx)


def R_mtx_joint_ri_half(c_row, c_col, L0, L1, expansion_mtx,
                        mtx_shrink_row, mtx_shrink_col, mtx_extract_b=None):
    """
    right dual matrix for the case where both the annihilating filter and the FRI sequence
    are Hermitian symmetric. Hence, the output from the annihilation equations are also
    Hermitian symmetric.
    :param c_row: annihilating filter coefficients for the row dimensions
    :param c_col: annihilating filter coefficients for the column dimensions
    :param L0: dimension 0 of the input FRI sequence
    :param L1: deimension 1 of the input FRI sequence
    :param expansion_mtx: expand the real valued represenation of the first half of FRI sequence
            to its full range
    :param mtx_shrink_row: shrink output to (approx.) half the size due to Hermitian symmetry
    :param mtx_shrink_col: shrink output to (approx.) half the size due to Hermitian symmetry
    :param mtx_extract_b: extract the portion of the newly reconstructed b (for cases with
            available reference solutions)
    :return:
    """
    R_loop_row = np.dot(mtx_shrink_row, cpx_mtx2real(convmtx2_valid(c_row, L0, L1)))
    R_loop_col = np.dot(mtx_shrink_col, cpx_mtx2real(convmtx2_valid(c_col, L0, L1)))
    R_mtx = np.vstack((R_loop_row, R_loop_col))
    if mtx_extract_b is not None:
        return np.dot(R_mtx, np.dot(expansion_mtx, mtx_extract_b))
    else:
        return np.dot(R_mtx, expansion_mtx)


def T_mtx_joint_ri_half(b_cpx, sz_coef_row0, sz_coef_row1,
                        sz_coef_col0, sz_coef_col1,
                        mtx_shrink_row, mtx_shrink_col,
                        expansion_mtx_coef_row,
                        expansion_mtx_coef_col):
    """
    (block)Toeplitz matrix associated with the joint annihilaiton cases. Here both the
    FRI sequence and the annihilating fitler are Hermitian symmetric.
    :param b_cpx: COMPLEX-valued FRI sequence.
    :param sz_coef0: dimension 0 of the annihilating filter coefficients
    :param sz_coef1: dimension 1 of the annihilating filter coefficients
    :param mtx_shrink_row: shrink output to (approx.) half the size due to Hermitian symmetry
    :param mtx_shrink_col: shrink output to (approx.) half the size due to Hermitian symmetry
    :return:
    """
    T_mtx_row = np.dot(mtx_shrink_row,
                       cpx_mtx2real(convmtx2_valid(b_cpx, sz_coef_row0, sz_coef_row1))
                       )
    T_mtx_col = np.dot(mtx_shrink_col,
                       cpx_mtx2real(convmtx2_valid(b_cpx, sz_coef_col0, sz_coef_col1))
                       )
    return linalg.block_diag(np.dot(T_mtx_row, expansion_mtx_coef_row),
                             np.dot(T_mtx_col, expansion_mtx_coef_col))


def output_shrink(out_len):
    """
    shrink the convolution output to half the size.
    used when both the annihilating filter and the uniform samples of sinusoids satisfy
    Hermitian symmetric.
    :param out_len: the length of the (complex-valued) output vector
    :return:
    """
    # out_len = L - K
    if out_len % 2 == 0:
        half_out_len = np.int(out_len / 2.)
        mtx_r = np.hstack((np.eye(half_out_len),
                           np.zeros((half_out_len, half_out_len))))
        mtx_i = mtx_r
    else:
        half_out_len = np.int((out_len + 1) / 2.)
        mtx_r = np.hstack((np.eye(half_out_len),
                           np.zeros((half_out_len, half_out_len - 1))))
        mtx_i = np.hstack((np.eye(half_out_len - 1),
                           np.zeros((half_out_len - 1, half_out_len))))
    return linalg.block_diag(mtx_r, mtx_i)


def convmtx2_valid(H, M, N):
    """
    2d convolution matrix with the boundary condition 'valid', i.e., only filter
    within the given data block.
    :param H: 2d filter
    :param M: input signal dimension is M x N
    :param N: input signal dimension is M x N
    :return:
    """
    T = convmtx2(H, M, N)
    s_H0, s_H1 = H.shape
    if M >= s_H0:
        S = np.pad(np.ones((M - s_H0 + 1, N - s_H1 + 1), dtype=bool),
                   ((s_H0 - 1, s_H0 - 1), (s_H1 - 1, s_H1 - 1)),
                   'constant', constant_values=False)
    else:
        S = np.pad(np.ones((s_H0 - M + 1, s_H1 - N + 1), dtype=bool),
                   ((M - 1, M - 1), (N - 1, N - 1)),
                   'constant', constant_values=False)
    T = T[S.flatten('F'), :]
    return T


def convmtx2(H, M, N):
    """
    build 2d convolution matrix
    :param H: 2d filter
    :param M: input signal dimension is M x N
    :param N: input signal dimension is M x N
    :return:
    """
    P, Q = H.shape
    blockHeight = int(M + P - 1)
    blockWidth = int(M)
    blockNonZeros = int(P * M)
    totalNonZeros = int(Q * N * blockNonZeros)

    THeight = int((N + Q - 1) * blockHeight)
    TWidth = int(N * blockWidth)

    Tvals = np.empty((totalNonZeros, 1), dtype=H.dtype)
    Trows = np.empty((totalNonZeros, 1), dtype=int)
    Tcols = np.empty((totalNonZeros, 1), dtype=int)

    c = np.dot(np.diag(np.arange(1, M + 1)), np.ones((M, P), dtype=float))
    r = np.repeat(np.reshape(c + np.arange(0, P)[np.newaxis], (-1, 1), order='F'), N, axis=1)
    c = np.repeat(c.flatten('F')[:, np.newaxis], N, axis=1)

    colOffsets = np.arange(N) * M
    colOffsets = np.reshape(np.repeat(colOffsets[np.newaxis], M * P, axis=0) + c, (-1, 1), order='F') - 1

    rowOffsets = np.arange(N) * blockHeight
    rowOffsets = np.reshape(np.repeat(rowOffsets[np.newaxis], M * P, axis=0) + r, (-1, 1), order='F') - 1

    for k in range(Q):
        val = np.reshape(np.tile((H[:, k]).flatten(), (M, 1)), (-1, 1), order='F')
        first = int(k * N * blockNonZeros)
        last = int(first + N * blockNonZeros)
        Trows[first:last] = rowOffsets
        Tcols[first:last] = colOffsets
        Tvals[first:last] = np.tile(val, (N, 1))
        rowOffsets += blockHeight

    T = np.zeros((THeight, TWidth), dtype=H.dtype)
    T[Trows, Tcols] = Tvals
    return T


def sph2cart(r, colatitude, azimuth):
    """
    spherical to cartesian coordinates
    :param r: radius
    :param colatitude: co-latitude
    :param azimuth: azimuth
    :return:
    """
    r_sin_colatitude = r * np.sin(colatitude)
    x = r_sin_colatitude * np.cos(azimuth)
    y = r_sin_colatitude * np.sin(azimuth)
    z = r * np.cos(colatitude)
    return x, y, z


def UVW2J2000(RA_focus_rad, DEC_focus_rad,
              u_rad, v_rad, w_rad=None, convert_dms=False):
    """
    convert UVW coordinate to the J2000 coordinate
    :param RA_focus_rad: RA of the telescope focus in radian
    :param DEC_focus_rad: DEC of the telescope focus in radian
    :param u_rad: U coordinate in UVW in radian
    :param v_rad: V coordinate in UVW in radian
    :param w_rad: W coordinate in UVW in radian.
            If not given, then w is chosen as 0.
    :return:
    """
    mtx_J2000_to_uvw = np.array([
        [-np.sin(RA_focus_rad), np.cos(RA_focus_rad), 0],
        [-np.cos(RA_focus_rad) * np.sin(DEC_focus_rad),
         -np.sin(RA_focus_rad) * np.sin(DEC_focus_rad),
         np.cos(DEC_focus_rad)],
        sph2cart(1, 0.5 * np.pi - DEC_focus_rad, RA_focus_rad)
    ])
    if w_rad is None:
        coord_rad_J2000 = linalg.solve(
            mtx_J2000_to_uvw,
            np.row_stack((u_rad.squeeze(), v_rad.squeeze(),
                          np.zeros(v_rad.size, dtype=float)
            ))
        )
    else:
        coord_rad_J2000 = linalg.solve(
            mtx_J2000_to_uvw,
            np.row_stack((u_rad.squeeze(), v_rad.squeeze(), w_rad.squeeze()))
        )

    x_rad_J2000 = coord_rad_J2000[0] + np.cos(DEC_focus_rad) * np.cos(RA_focus_rad)
    y_rad_J2000 = coord_rad_J2000[1] + np.cos(DEC_focus_rad) * np.sin(RA_focus_rad)
    z_rad_J2000 = coord_rad_J2000[2] + np.sin(DEC_focus_rad)

    if convert_dms:
        RA_DEC_hmsdms_J2000 = SkyCoord(
            ra=np.arctan2(y_rad_J2000, x_rad_J2000),
            dec=np.arcsin(z_rad_J2000),
            unit=units.radian
        ).to_string('hmsdms')
        RA_hms_J2000 = []
        DEC_dms_J2000 = []
        for ra_dec_loop in RA_DEC_hmsdms_J2000:
            RA_hms_loop, DEC_dms_loop = ra_dec_loop.split(' ')
            RA_hms_J2000.append(RA_hms_loop)
            DEC_dms_J2000.append(DEC_dms_loop)
        return x_rad_J2000, y_rad_J2000, z_rad_J2000, RA_hms_J2000, DEC_dms_J2000
    else:
        return x_rad_J2000, y_rad_J2000, z_rad_J2000


def cpx_mtx2real(mtx):
    """
    extend complex valued matrix to an extended matrix of real values only
    :param mtx: input complex valued matrix
    :return:
    """
    return np.vstack((np.hstack((mtx.real, -mtx.imag)), np.hstack((mtx.imag, mtx.real))))


def sph_extract_off_diag(mtx):
    """
    extract off-diagonal entries in mtx
    The output vector is order in a column major manner
    :param mtx: input matrix to extract the off-diagonal entries
    :return:
    """
    # we transpose the matrix because the function np.extract will first flatten the matrix
    # withe ordering convention 'C' instead of 'F'!!
    Q = mtx.shape[0]
    num_bands = mtx.shape[2]
    extract_cond = np.reshape((1 - np.eye(Q)).T.astype(bool), (-1, 1), order='F')
    return np.column_stack([np.reshape(np.extract(extract_cond, mtx[:, :, band].T),
                                       (-1, 1), order='F')
                            for band in range(num_bands)])


def planar_gen_visibility_beamforming(alpha, xk, yk, p_x, p_y,
                                      beam_weights_func, num_station,
                                      num_subband, num_sti,
                                      snr_data=float('inf')):
    visibility = np.empty((num_station * (num_station - 1), num_sti, num_subband),
                          dtype=complex, order='F')
    visibility_noisy = np.empty((num_station * (num_station - 1), num_sti, num_subband),
                                dtype=complex, order='F')
    for band_count in range(num_subband):
        visi_per_band = []
        visi_noisy_per_band = []

        for sti_loop in range(num_sti):
            visi, visi_noisy = \
                planar_gen_visi_beamforming_inner(
                    alpha[:, band_count], xk, yk,
                    p_x[:, :, sti_loop, band_count],
                    p_y[:, :, sti_loop, band_count],
                    beam_weights_func, snr_data=snr_data
                )
            visi_per_band.append(visi)
            visi_noisy_per_band.append(visi_noisy)

        visibility[:, :, band_count] = np.column_stack(visi_per_band)
        visibility_noisy[:, :, band_count] = np.column_stack(visi_noisy_per_band)

    return visibility, visibility_noisy


def planar_gen_visi_beamforming_inner(alpha_loop, xk, yk, p_x_loop, p_y_loop,
                                      beam_weights_func,
                                      snr_data=float('inf')):
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

    visi_noiseless_effective = [
        [
            np.dot(effective_rows[station_count1][station_count2], alpha_loop)
            for station_count2 in range(station_count1)
        ]
        for station_count1 in range(num_station)
    ]

    # compute the noise variance base on snr
    sigma_noise = linalg.norm(np.concatenate(visi_noiseless_effective)) / \
                  np.sqrt(num_station * (num_station - 1) * 0.5) * 10 ** (-snr_data / 20.)
    noise = sigma_noise / np.sqrt(2) * (
        np.random.randn(num_station * (num_station - 1) // 2) +
        1j * np.random.randn(num_station * (num_station - 1) // 2)
    )

    # add noise to visi_noiseless_effective
    visi_noisy_effective = []
    loop_count = 0
    for station_count1 in range(num_station):
        visi_noisy_loop = []
        for station_count2 in range(station_count1):
            visi_noisy_loop.append(
                visi_noiseless_effective[station_count1][station_count2] +
                noise[loop_count]
            )
            loop_count += 1

        visi_noisy_effective.append(visi_noisy_loop)

    visi_noiseless = np.empty(num_station * (num_station - 1), dtype=complex, order='C')
    visi_noisy = np.empty(num_station * (num_station - 1), dtype=complex, order='C')
    count = 0
    for station_count1 in range(num_station):
        for station_count2 in range(num_station):
            if station_count2 > station_count1:
                visi_noiseless[count] = np.conj(visi_noiseless_effective[station_count2][station_count1])
                visi_noisy[count] = np.conj(visi_noisy_effective[station_count2][station_count1])
                count += 1
            elif station_count2 < station_count1:
                visi_noiseless[count] = visi_noiseless_effective[station_count1][station_count2]
                visi_noisy[count] = visi_noisy_effective[station_count1][station_count2]
                count += 1

    return visi_noiseless, visi_noisy


def partition_stages(K, stage_blk_len, removal_blk_len):
    """
    Partition the reconstruction of K Dirac deltas into several stages.
    :param K: total number of Dirac deltas
    :param stage_blk_len: length of the reconstruction block at each stage
    :param removal_blk_len: number of reconstructed Dirac deltas that is NOT
                carried over to the reconstruction stage
    :return:
    """
    assert removal_blk_len < stage_blk_len
    assert stage_blk_len <= K
    effective_blk_len = stage_blk_len - removal_blk_len
    num_stage = (K - removal_blk_len) // effective_blk_len
    if np.remainder(K - removal_blk_len, effective_blk_len) + effective_blk_len < stage_blk_len:
        K_est_stage_lst = [stage_blk_len] * (num_stage - 1)
        removal_blk_len_lst = [removal_blk_len] * (num_stage - 1)

        K_est_last_stage = np.remainder(K - removal_blk_len, effective_blk_len) + \
                           effective_blk_len + removal_blk_len
        max_last_few_stages = int(np.ceil(K_est_last_stage / effective_blk_len))
        for last_few_stages in range(max_last_few_stages):
            K_est_stage_lst.append(K_est_last_stage)
            removal_blk_len_lst.append(K_est_last_stage - effective_blk_len)
            K_est_last_stage -= effective_blk_len

            if K_est_last_stage <= 1:
                break

        K_est_stage_lst.append(removal_blk_len_lst[-1])

    else:
        K_est_stage_lst = [stage_blk_len] * num_stage
        removal_blk_len_lst = [removal_blk_len] * num_stage

        K_est_last_stage = np.remainder(K - removal_blk_len, effective_blk_len)
        K_est_stage_lst.append(K_est_last_stage)

    assert len(removal_blk_len_lst) + 1 == len(K_est_stage_lst)

    K_est_stage_lst = list(np.trim_zeros(K_est_stage_lst, trim='b'))
    num_stage = len(K_est_stage_lst)
    removal_blk_len_lst = removal_blk_len_lst[:num_stage - 1]

    return K_est_stage_lst, removal_blk_len_lst


def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)

    Reference: http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    Modified by Hanjie Pan
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2, 2)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = np.double(maximum_filter(image, footprint=neighborhood) == image)
    # local_max = maximum_filter(image, footprint=np.ones((15, 15))) == image
    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = (image == 0)

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line1 will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = np.double(binary_erosion(background, structure=neighborhood, border_value=1))

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask
    detected_peaks = local_max - eroded_background
    peak_image = detected_peaks * image
    peak_locs = (np.asarray(np.nonzero(detected_peaks))).astype('int')
    return detected_peaks, peak_image, peak_locs
