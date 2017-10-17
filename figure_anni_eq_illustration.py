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
import matplotlib.pyplot as plt
import matplotlib.patches as patches

if __name__ == '__main__':
    tau = 1
    alpha1 = 0.11
    alpha2 = 0.2
    alpha3 = 0.05
    t1 = 0.15
    t2 = 0.3
    t3 = (t1 + t2) * 0.5 + 0.5 * tau
    alpha_k = np.array([alpha1, alpha2, alpha3])
    tk = np.array([t1, t2, t3])

    # "continuous" time plot
    t = np.linspace(0, tau, num=20000)
    num_samples = 22
    # t_samp = 0.1 + (0.55 - 0.1) * np.arange(num_samples) / num_samples
    t_samp = np.arange(num_samples) / num_samples * tau
    B = 9
    m_grid, t_plt_grid = np.meshgrid(np.arange(-np.floor(B / 2.), 1 + np.floor(B / 2.)), t)
    G = 1. / B * np.exp(2j * np.pi * m_grid * t_plt_grid)
    tk_grid, m_grid_gt = np.meshgrid(tk, np.arange(-np.floor(B / 2.), 1 + np.floor(B / 2.)))
    x_hat_noiseless = np.dot(np.exp(-2j * np.pi * m_grid_gt * tk_grid), alpha_k)
    low_pass_sig = np.real(np.dot(G, x_hat_noiseless))

    # sampling
    m_grid, t_samp_grid = np.meshgrid(np.arange(-np.floor(B / 2.), 1 + np.floor(B / 2.)), t_samp)
    G_samp = 1. / B * np.exp(2j * np.pi * m_grid * t_samp_grid)
    low_pass_sig_samp = np.real(np.dot(G_samp, x_hat_noiseless))

    # annihlating filter (in time)
    Hz = lambda z: (1 - np.exp(-2j * np.pi / tau * t1) / z) * \
                   (1 - np.exp(-2j * np.pi / tau * t2) / z) * \
                   (1 - np.exp(-2j * np.pi / tau * t3) / z)
    mask = Hz(np.exp(-2j * np.pi / tau * t))
    filter_coef = np.zeros(3 + 1, dtype=complex)
    filter_coef[0] = 1
    filter_coef[1] = -np.exp(-2j * np.pi / tau * t1) - \
                     np.exp(-2j * np.pi / tau * t2) - \
                     np.exp(-2j * np.pi / tau * t3)
    filter_coef[2] = np.exp(-2j * np.pi / tau * t1) * np.exp(-2j * np.pi / tau * t2) + \
                     np.exp(-2j * np.pi / tau * t1) * np.exp(-2j * np.pi / tau * t3) + \
                     np.exp(-2j * np.pi / tau * t2) * np.exp(-2j * np.pi / tau * t3)
    filter_coef[3] = -np.exp(-2j * np.pi / tau * t1) * np.exp(-2j * np.pi / tau * t2) * \
                     np.exp(-2j * np.pi / tau * t3)

    fig = plt.figure(figsize=(3.5, 1), dpi=90)
    ax = fig.add_axes([0, 0.11, 1, 0.89])
    markerline, stemlines, baseline = ax.stem(tk, alpha_k)
    plt.setp(stemlines, linewidth=1, color=[0, 0.447, 0.741])
    plt.setp(markerline, marker='', linewidth=1, markersize=8,
             markerfacecolor='k', mec='k')
    plt.setp(baseline, linewidth=0)
    plt.axhline(0, color=[0.4, 0.4, 0.4], linewidth=0.5)
    plt.xlim([0, tau])
    ax.xaxis.set_label_coords(0.5, 0.02)
    plt.tick_params(axis='both',
                    which='both',
                    bottom='off',
                    top='off',
                    left='off',
                    right='off',
                    labelbottom='off',
                    labelleft='off'
                    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    file_name = './result/AA_fig2a_sparse_conti.pdf'
    plt.savefig(file_name, format='pdf', dpi=600, transparent=True)

    fig = plt.figure(figsize=(3.5, 0.78), dpi=90)
    ax = fig.add_axes([0, 0.11, 1, 0.89])
    line = plt.plot(t, low_pass_sig)
    markerline, stemlines, baseline = \
        plt.stem(t_samp, low_pass_sig_samp)
    plt.setp(stemlines, linewidth=0.5, color=[0.466, 0.674, 0.188])
    plt.setp(markerline, marker='o', linewidth=0.5, markersize=2,
             markerfacecolor=[0.466, 0.674, 0.188], mec=[0.466, 0.674, 0.188])
    plt.setp(baseline, linewidth=0)
    plt.axhline(0, color=[0.4, 0.4, 0.4], linewidth=0.5)
    plt.xlim([0, tau])
    plt.setp(line, linestyle='-', color=[0.8, 0.8, 0.8], linewidth=0.5)
    plt.tick_params(axis='both',
                    which='both',
                    bottom='off',
                    top='off',
                    left='off',
                    right='off',
                    labelbottom='off',
                    labelleft='off'
                    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    file_name = './result/AA_fig2b_samp.pdf'
    plt.savefig(file_name, format='pdf', dpi=600, transparent=True)

    fig = plt.figure(figsize=(3.5, 0.78), dpi=90)
    ax = fig.add_axes([0, 0.11, 1, 0.89])
    line = plt.plot(t, np.abs(mask))
    plt.axhline(0, color=[0.4, 0.4, 0.4], linewidth=0.5)
    plt.xlim([0, tau])
    plt.setp(line, linestyle='-', color=[0.85, 0.325, 0.098], linewidth=0.5)

    plt.tick_params(axis='both',
                    which='both',
                    bottom='off',
                    top='off',
                    left='off',
                    right='off',
                    labelbottom='off',
                    labelleft='off'
                    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    zoom_in_factor = 6
    x_lim = plt.gca().get_xlim()
    y_lim = plt.gca().get_ylim()
    ax.add_patch(
        patches.Rectangle(
            (x_lim[0], y_lim[0] / zoom_in_factor),
            x_lim[1] - x_lim[0],
            (y_lim[1] - y_lim[0]) / zoom_in_factor,
            alpha=0.08,
            facecolor='black',
            edgecolor='None'
        )
    )
    file_name = './result/AA_fig2c_mask.pdf'
    plt.savefig(file_name, format='pdf', dpi=600, transparent=True)

    fig = plt.figure(figsize=(3.5, 0.78 / 2), dpi=90)
    ax = fig.add_axes([0, 0.11, 1, 0.89])
    line = plt.plot(t, np.abs(mask))
    plt.xlim([0, tau])
    plt.axhline(0, color=[0.4, 0.4, 0.4], linewidth=0.5)

    plt.ylim([y_lim[0] / zoom_in_factor, y_lim[1] / zoom_in_factor])
    plt.setp(line, linestyle='-', color=[0.85, 0.325, 0.098], linewidth=0.5)

    plt.tick_params(axis='both',
                    which='both',
                    bottom='off',
                    top='off',
                    left='off',
                    right='off',
                    labelbottom='off',
                    labelleft='off'
                    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.add_patch(
        patches.Rectangle(
            (x_lim[0], y_lim[0] / zoom_in_factor),
            x_lim[1] - x_lim[0],
            (y_lim[1] - y_lim[0]) / zoom_in_factor,
            alpha=0.08,
            facecolor='black',
            edgecolor='None'
        )
    )
    file_name = './result/AA_fig2d_mask_zoom.pdf'
    plt.savefig(file_name, format='pdf', dpi=600, transparent=True)
    plt.show()
