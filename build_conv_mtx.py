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
import numpy as np


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
    S = np.zeros((s_H0 + M - 1, s_H1 + N - 1), dtype=bool)
    if M >= s_H0:
        S[s_H0 - 1: M, s_H1 - 1: N] = True
    else:
        S[M - 1: s_H0, N - 1: s_H1] = True
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
    N_blockNonZeros = int(N * blockNonZeros)
    totalNonZeros = Q * N_blockNonZeros

    THeight = int((N + Q - 1) * blockHeight)
    TWidth = int(N * blockWidth)

    Tvals = np.zeros(totalNonZeros, dtype=H.dtype)
    Trows = np.zeros(totalNonZeros, dtype=int)
    Tcols = np.zeros(totalNonZeros, dtype=int)

    c = np.repeat(np.arange(1, M + 1)[:, np.newaxis], P, axis=1)
    r = np.repeat(np.reshape(c + np.arange(0, P)[np.newaxis], (-1, 1), order='F'), N, axis=1)
    c = np.repeat(c.flatten('F')[:, np.newaxis], N, axis=1)

    colOffsets = np.arange(N, dtype=int) * M
    colOffsets = (np.repeat(colOffsets[np.newaxis], M * P, axis=0) + c).flatten('F') - 1

    rowOffsets = np.arange(N, dtype=int) * blockHeight
    rowOffsets = (np.repeat(rowOffsets[np.newaxis], M * P, axis=0) + r).flatten('F') - 1

    for k in range(Q):
        val = (np.tile((H[:, k]).flatten(), (M, 1))).flatten('F')
        first = int(k * N_blockNonZeros)
        last = int(first + N_blockNonZeros)
        Trows[first:last] = rowOffsets
        Tcols[first:last] = colOffsets
        Tvals[first:last] = np.tile(val, N)
        rowOffsets += blockHeight

    T = np.zeros((THeight, TWidth), dtype=H.dtype)
    T[Trows, Tcols] = Tvals
    return T
