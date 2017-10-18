"""
plotter.py: plot functions of the results
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
import re
import os
import subprocess
import numpy as np
from astropy import units
from astropy.coordinates import SkyCoord
from utils import planar_distance, UVW2J2000
import matplotlib

if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')

import matplotlib.colors as mcolors

try:
    which_latex = subprocess.check_output(['which', 'latex'])
    os.environ['PATH'] = \
        os.environ['PATH'] + ':' + \
        os.path.dirname(which_latex.decode('utf-8').rstrip('\n'))
    use_latex = True
except subprocess.CalledProcessError:
    use_latex = False

if use_latex:
    from matplotlib import rcParams

    rcParams['text.usetex'] = True
    rcParams['text.latex.unicode'] = True
    rcParams['text.latex.preamble'] = [r"\usepackage{bm}"]

import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import plot
import plotly.graph_objs as go
import plotly

sns.set_style('ticks',
              {
                  'xtick.major.size': 3.5,
                  'xtick.minor.size': 2,
                  'ytick.major.size': 3.5,
                  'ytick.minor.size': 2,
                  'axes.linewidth': 0.8
              }
              )


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def planar_plot_diracs_zoom_J2000(
        x_plt_grid, y_plt_grid, zoom_box=None,
        RA_focus_rad=0, DEC_focus_rad=0,
        x_ref=None, y_ref=None, amplitude_ref=None, marker_ref='^',
        x_recon=None, y_recon=None, amplitude_recon=None, marker_recon='*',
        max_amp_ref=None, max_amp=None, cmap='magma_r',
        background_img=None, marker_scale=1,
        marker_alpha=0.6, legend_marker_scale=0.7,
        save_fig=False, file_name='sph_recon_2d_dirac',
        reverse_xaxis=True,
        label_ref_sol='ground truth', label_recon='reconstruction', legend_loc=0,
        file_format='pdf', dpi=300, close_fig=True, has_title=True, title_str=None):
    """
    zoom-in plot of the reconstructed point sources in the J2000 coordinates
    :param x_plt_grid: plotting grid on the horizontal axis
    :param y_plt_grid: plotting grid on the vertical axis
    :param zoom_box: the box area to zoom-in. It's a list of 4 element:
            [lower_x, lower_y, width, height]
    :param RA_focus_rad: telescope focus in radian (right ascension)
    :param DEC_focus_rad: telescope focus in radian (declination)
    :param x_ref: ground truth RA of the sources in UVW
    :param y_ref: ground truth DEC of the sources in UVW
    :param amplitude_ref: ground truth intensities of the sources
    :param x_recon: reconstructed RA of the sources in UVW
    :param y_recon: reconstructed DEC of the sources in UVW
    :param amplitude_recon: reconstructed intensities of the sources
    :param max_amp_ref: maximum source intensity (used for noramlization of marker size)
    :param max_amp: maximum source intensity (used for noramlization of marker size)
    :param cmap: colormap
    :param background_img: background image
    :param marker_scale: prescaling factor for marker size
    :param marker_alpha: alpha (transparency) for the marker
    :param save_fig: whether to save the figure or not
    :param file_name: figure file name
    :param reverse_xaxis: whether to reverse the horizontal (RA) axis or not
    :param label_ref_sol: reference solution label name
    :param label_recon: reconstruction label name
    :param legend_loc: location of the legend. Default 'best'
    :param file_format: figure file format
    :param dpi: dpi for the saved figure file
    :param close_fig: close figure or not
    :param has_title: whether to use a figure title or not.
    :param title_str: title string. If has_title is true, the default title is
            the reconstruction error.
    :return:
    """
    # decide the plotting grid and background image based on the zoom-in box
    if zoom_box is None or background_img is None:
        planar_plot_diracs_J2000(
            x_plt_grid=x_plt_grid, y_plt_grid=y_plt_grid,
            RA_focus_rad=RA_focus_rad, DEC_focus_rad=DEC_focus_rad,
            x_ref=x_ref, y_ref=y_ref,
            amplitude_ref=amplitude_ref, marker_ref=marker_ref,
            x_recon=x_recon, y_recon=y_recon,
            amplitude_recon=amplitude_recon, marker_recon=marker_recon,
            max_amp_ref=max_amp_ref, max_amp=max_amp, cmap=cmap,
            background_img=background_img,
            marker_scale=marker_scale,
            legend_marker_scale=legend_marker_scale,
            marker_alpha=marker_alpha, save_fig=save_fig,
            file_name=file_name, reverse_xaxis=reverse_xaxis,
            label_ref_sol=label_ref_sol, label_recon=label_recon,
            legend_loc=legend_loc, file_format=file_format, dpi=dpi,
            close_fig=close_fig, has_title=has_title, title_str=title_str
        )
    else:
        img_sz0, img_sz1 = x_plt_grid.shape
        pixel_idx_row_lower = int(img_sz0 * zoom_box[1])
        pixel_idx_col_left = int(img_sz1 * zoom_box[0])
        pixel_idx_row_upper = int(img_sz0 * (zoom_box[1] + zoom_box[3]))
        pixel_idx_col_right = int(img_sz1 * (zoom_box[0] + zoom_box[2]))
        x_plt_grid_zoom = \
            x_plt_grid[pixel_idx_row_lower:pixel_idx_row_upper,
            pixel_idx_col_left:pixel_idx_col_right]
        y_plt_grid_zoom = \
            y_plt_grid[pixel_idx_row_lower:pixel_idx_row_upper,
            pixel_idx_col_left:pixel_idx_col_right]
        background_img_zoom = \
            background_img[pixel_idx_row_lower:pixel_idx_row_upper,
            pixel_idx_col_left:pixel_idx_col_right]
        planar_plot_diracs_J2000(
            x_plt_grid=x_plt_grid_zoom, y_plt_grid=y_plt_grid_zoom,
            RA_focus_rad=RA_focus_rad, DEC_focus_rad=DEC_focus_rad,
            x_ref=x_ref, y_ref=y_ref,
            amplitude_ref=amplitude_ref, marker_ref=marker_ref,
            x_recon=x_recon, y_recon=y_recon,
            amplitude_recon=amplitude_recon, marker_recon=marker_recon,
            max_amp_ref=max_amp_ref, max_amp=max_amp, cmap=cmap,
            background_img=background_img_zoom,
            marker_scale=marker_scale,
            legend_marker_scale=legend_marker_scale,
            marker_alpha=marker_alpha, save_fig=save_fig,
            file_name=file_name, reverse_xaxis=reverse_xaxis,
            label_ref_sol=label_ref_sol, label_recon=label_recon,
            legend_loc=legend_loc, file_format=file_format, dpi=dpi,
            close_fig=close_fig, has_title=has_title, title_str=title_str
        )


def planar_plot_diracs_J2000(
        x_plt_grid, y_plt_grid,
        RA_focus_rad=0, DEC_focus_rad=0,
        x_ref=None, y_ref=None, amplitude_ref=None, marker_ref='^',
        x_recon=None, y_recon=None, amplitude_recon=None, marker_recon='*',
        max_amp_ref=None, max_amp=None, cmap='magma_r',
        background_img=None, marker_scale=1,
        marker_alpha=0.6, legend_marker_scale=0.7,
        save_fig=False, file_name='sph_recon_2d_dirac',
        reverse_xaxis=True,
        label_ref_sol='ground truth', label_recon='reconstruction', legend_loc=0,
        file_format='pdf', dpi=300, close_fig=True, has_title=True, title_str=None):
    """
    plot the reconstructed point sources in the J2000 coordinates
    :param y_ref: ground truth colatitudes of the Dirac
    :param x_ref: ground truth azimuths of the Dirac
    :param amplitude_ref: ground truth amplitudes of the Dirac
    :param y_recon: reconstructed colatitudes of the Dirac
    :param x_recon: reconstructed azimuths of the Dirac
    :param amplitude_recon: reconstructed amplitudes of the Dirac
    :param lon_0: center of the projection (longitude) <- azimuth
    :param lat_0: center of the projection (latitude) <- pi/2 - co-latitude
    :param save_fig: whether to save figure or not
    :param file_name: figure file name (basename)
    :param file_format: format of the saved figure file
    :return:
    """
    if y_ref is not None and x_ref is not None and amplitude_ref is not None:
        ref_pt_available = True
    else:
        ref_pt_available = False

    if y_recon is not None and x_recon is not None and amplitude_recon is not None:
        recon_pt_available = True
    else:
        recon_pt_available = False

    # convert UVW coordinates to J2000 in [arcmin]
    x_plt_grid_J2000 = x_plt_grid * 180 / np.pi * 60
    y_plt_grid_J2000 = y_plt_grid * 180 / np.pi * 60
    if ref_pt_available:
        x_ref_J2000, y_ref_J2000, z_ref_J2000 = UVW2J2000(
            RA_focus_rad, DEC_focus_rad,
            x_ref, y_ref, convert_dms=False
        )[:3]
        RA_ref_J2000 = np.arctan2(y_ref_J2000, x_ref_J2000)
        DEC_ref_J2000 = np.arcsin(z_ref_J2000)

    if recon_pt_available:
        x_recon_J2000, y_recon_J2000, z_recon_J2000 = UVW2J2000(
            RA_focus_rad, DEC_focus_rad,
            x_recon, y_recon, convert_dms=False
        )[:3]
        RA_recon_J2000 = np.arctan2(y_recon_J2000, x_recon_J2000)
        DEC_recon_J2000 = np.arcsin(z_recon_J2000)

    # plot
    if background_img is not None:
        ax = plt.figure(figsize=(5.5, 4), dpi=dpi).add_subplot(111)
        pos_original = ax.get_position()
        pos_new = [pos_original.x0 + 0.06, pos_original.y0 + 0.01,
                   pos_original.width, pos_original.height]
        ax.set_position(pos_new)
        plt.pcolormesh(x_plt_grid_J2000, y_plt_grid_J2000, background_img,
                       shading='gouraud', cmap=cmap)

    if ref_pt_available:
        if max_amp_ref is not None:
            amplitude_ref_rescaled = amplitude_ref / max_amp_ref
        else:
            amplitude_ref_rescaled = amplitude_ref / np.max(amplitude_ref)

        plt.scatter(RA_ref_J2000 * 180 / np.pi * 60,
                    DEC_ref_J2000 * 180 / np.pi * 60,
                    s=amplitude_ref_rescaled * 200 * marker_scale,  # 350 for '^'
                    marker=marker_ref, edgecolors='k', linewidths=0.5,
                    alpha=marker_alpha, c='w',
                    label=label_ref_sol)

    if recon_pt_available:
        if max_amp is not None:
            amplitude_rescaled = amplitude_recon / max_amp
        else:
            amplitude_rescaled = amplitude_recon / np.max(amplitude_recon)

        plt.scatter(RA_recon_J2000 * 180 / np.pi * 60,
                    DEC_recon_J2000 * 180 / np.pi * 60,
                    s=amplitude_rescaled * 600 * marker_scale,
                    marker=marker_recon, edgecolors='k', linewidths=0.5, alpha=marker_alpha,
                    c=np.tile([0.996, 0.410, 0.703], (x_recon.size, 1)),
                    label=label_recon)

    if has_title and ref_pt_available and recon_pt_available and title_str is None:
        dist_recon = planar_distance(x_ref, y_ref, x_recon, y_recon)[0]

        # in degree, minute, and second representation
        dist_recon_dms = SkyCoord(
            ra=0, dec=dist_recon, unit=units.radian
        ).to_string('dms').split(' ')[1]
        dist_recon_dms = list(filter(None, re.split('[dms]+', dist_recon_dms)))
        dist_recon_dms = (
            '{0}' + u'\u00B0' + '{1}' + u'\u2032' + '{2:.2f}' + u'\u2033'
        ).format(dist_recon_dms[0], dist_recon_dms[1], float(dist_recon_dms[2]))

        plt.title(u'average error = {0}'.format(dist_recon_dms), fontsize=11)
    elif has_title and title_str is not None:
        plt.title(title_str, fontsize=11)
    else:
        plt.title(u'', fontsize=11)

    if ref_pt_available or recon_pt_available:
        plt.legend(scatterpoints=1, loc=legend_loc, fontsize=9,
                   ncol=1, markerscale=legend_marker_scale,
                   handletextpad=0.1, columnspacing=0.1,
                   labelspacing=0.1, framealpha=0.5, frameon=True)

    plt.axis('image')
    plt.xlim((np.min(x_plt_grid_J2000), np.max(x_plt_grid_J2000)))
    plt.ylim((np.min(y_plt_grid_J2000), np.max(y_plt_grid_J2000)))
    plt.xlabel('RA (J2000)')
    plt.ylabel('DEC (J2000)')

    if reverse_xaxis:
        plt.gca().invert_xaxis()

    # extract lablels to convert to hmsdms format
    x_tick_loc, _ = plt.xticks()
    y_tick_loc, _ = plt.yticks()

    x_tick_loc = x_tick_loc[1:-1]
    y_tick_loc = y_tick_loc[1:-1]

    # evaluate a uniform grid of the same size
    x_tick_loc = np.linspace(start=x_tick_loc[0], stop=x_tick_loc[-1],
                             num=x_tick_loc.size, endpoint=True)
    y_tick_loc = np.linspace(start=y_tick_loc[0], stop=y_tick_loc[-1],
                             num=y_tick_loc.size, endpoint=True)

    xlabels_hms_all = []
    for label_idx, xlabels_original_loop in enumerate(x_tick_loc):
        xlabels_original_loop = float(xlabels_original_loop)

        xlabels_hms = SkyCoord(
            ra=xlabels_original_loop, dec=0, unit=units.arcmin
        ).to_string('hmsdms').split(' ')[0]
        xlabels_hms = list(filter(None, re.split('[hms]+', xlabels_hms)))
        if label_idx == 0:
            xlabels_hms = (
                u'{0:.0f}h{1:.0f}m{2:.0f}s'
            ).format(float(xlabels_hms[0]),
                     float(xlabels_hms[1]),
                     float(xlabels_hms[2]))
        else:
            xlabels_hms = (
                u'{0:.0f}m{1:.0f}s'
            ).format(float(xlabels_hms[1]),
                     float(xlabels_hms[2]))

        xlabels_hms_all.append(xlabels_hms)

    ylabels_dms_all = []
    for label_idx, ylabels_original_loop in enumerate(y_tick_loc):
        ylabels_original_loop = float(ylabels_original_loop)
        ylabels_dms = SkyCoord(
            ra=0, dec=ylabels_original_loop, unit=units.arcmin
        ).to_string('hmsdms').split(' ')[1]
        ylabels_dms = list(filter(None, re.split('[dms]+', ylabels_dms)))
        ylabels_dms = (u'{0:.0f}\u00B0{1:.0f}\u2032').format(
            float(ylabels_dms[0]), float(ylabels_dms[1]) + float(ylabels_dms[2]) / 60.
        )
        ylabels_dms_all.append(ylabels_dms)

    plt.axis('image')
    plt.xlim((np.min(x_plt_grid_J2000), np.max(x_plt_grid_J2000)))
    plt.ylim((np.min(y_plt_grid_J2000), np.max(y_plt_grid_J2000)))
    plt.xticks(x_tick_loc)
    plt.yticks(y_tick_loc)
    plt.gca().set_xticklabels(xlabels_hms_all, fontsize=9)
    plt.gca().set_yticklabels(ylabels_dms_all, fontsize=9)

    if reverse_xaxis:
        plt.gca().invert_xaxis()

    if save_fig:
        plt.savefig(filename=(file_name + '.' + file_format), format=file_format,
                    dpi=dpi, transparent=True)

    if close_fig:
        plt.close()


def planar_plot_diracs(
        x_plt_grid, y_plt_grid,
        x_ref=None, y_ref=None, amplitude_ref=None,
        x_recon=None, y_recon=None, amplitude_recon=None,
        max_amp_ref=None, max_amp=None, cmap='magma_r',
        background_img=None, marker_scale=1, marker_alpha=0.6,
        save_fig=False, file_name='sph_recon_2d_dirac',
        xticklabels=None, yticklabels=None, reverse_xaxis=True,
        label_ref_sol='ground truth', label_recon='reconstruction', legend_loc=0,
        file_format='pdf', dpi=300, close_fig=True, has_title=True, title_str=None):
    """
    plot the reconstructed point sources with basemap module
    :param y_ref: ground truth colatitudes of the Dirac
    :param x_ref: ground truth azimuths of the Dirac
    :param amplitude_ref: ground truth amplitudes of the Dirac
    :param y_recon: reconstructed colatitudes of the Dirac
    :param x_recon: reconstructed azimuths of the Dirac
    :param amplitude_recon: reconstructed amplitudes of the Dirac
    :param lon_0: center of the projection (longitude) <- azimuth
    :param lat_0: center of the projection (latitude) <- pi/2 - co-latitude
    :param save_fig: whether to save figure or not
    :param file_name: figure file name (basename)
    :param file_format: format of the saved figure file
    :return:
    """
    if y_ref is not None and x_ref is not None and amplitude_ref is not None:
        ref_pt_available = True
    else:
        ref_pt_available = False

    if y_recon is not None and x_recon is not None and amplitude_recon is not None:
        recon_pt_available = True
    else:
        recon_pt_available = False

    # plot
    x_plt_grid_degree = np.degrees(x_plt_grid)
    y_plt_grid_degree = np.degrees(y_plt_grid)
    if background_img is not None:
        # cmap = sns.cubehelix_palette(dark=0.95, light=0.1, reverse=True,
        #                              start=1, rot=-0.6, as_cmap=True)
        # cmap = sns.cubehelix_palette(dark=0.95, light=0.1, reverse=True,
        #                              start=0.3, rot=-0.6, as_cmap=True)
        # cmap = 'cubehelix_r'  # 'Spectral_r'  # 'BuPu'
        # move the plotting area slight up
        ax = plt.figure(figsize=(5, 4), dpi=dpi).add_subplot(111)
        pos_original = ax.get_position()
        pos_new = [pos_original.x0, pos_original.y0 + 0.01,
                   pos_original.width, pos_original.height]
        ax.set_position(pos_new)
        plt.pcolormesh(x_plt_grid_degree, y_plt_grid_degree, background_img,
                       shading='gouraud', cmap=cmap)

    if ref_pt_available:
        if max_amp_ref is not None:
            amplitude_ref_rescaled = amplitude_ref / max_amp_ref
        else:
            amplitude_ref_rescaled = amplitude_ref / np.max(amplitude_ref)

        plt.scatter(np.degrees(x_ref), np.degrees(y_ref),
                    s=amplitude_ref_rescaled * 350 * marker_scale,
                    marker='^', edgecolors='k', linewidths=0.5, alpha=marker_alpha, c='w',
                    label=label_ref_sol)

    if recon_pt_available:
        if max_amp is not None:
            amplitude_rescaled = amplitude_recon / max_amp
        else:
            amplitude_rescaled = amplitude_recon / np.max(amplitude_recon)

        plt.scatter(np.degrees(x_recon), np.degrees(y_recon),
                    s=amplitude_rescaled * 600 * marker_scale,
                    marker='*', edgecolors='k', linewidths=0.5, alpha=marker_alpha,
                    c=np.tile([0.996, 0.410, 0.703], (x_recon.size, 1)),
                    label=label_recon)

    if has_title and ref_pt_available and recon_pt_available and title_str is None:
        dist_recon = planar_distance(x_ref, y_ref, x_recon, y_recon)[0]

        # in degree, minute, and second representation
        dist_recon_dms = SkyCoord(
            ra=0, dec=dist_recon, unit=units.radian
        ).to_string('dms').split(' ')[1]
        dist_recon_dms = list(filter(None, re.split('[dms]+', dist_recon_dms)))
        dist_recon_dms = (
            '{0}' + u'\u00B0' + '{1}' + u'\u2032' + '{2:.2f}' + u'\u2033'
        ).format(dist_recon_dms[0], dist_recon_dms[1], float(dist_recon_dms[2]))

        plt.title(u'average error = {0}'.format(dist_recon_dms), fontsize=11)
    elif has_title and title_str is not None:
        plt.title(title_str, fontsize=11)
    else:
        plt.title(u'', fontsize=11)

    if ref_pt_available or recon_pt_available:
        plt.legend(scatterpoints=1, loc=legend_loc, fontsize=9,
                   ncol=1, markerscale=0.7,
                   handletextpad=0.1, columnspacing=0.1,
                   labelspacing=0.1, framealpha=0.5, frameon=True)

    plt.axis('image')
    plt.xlim((np.min(x_plt_grid_degree), np.max(x_plt_grid_degree)))
    plt.ylim((np.min(y_plt_grid_degree), np.max(y_plt_grid_degree)))

    if xticklabels is not None:
        # set the number of ticks to match the length of the labels
        ''' from matplotlib documentation: "the number of ticks <= nbins +1" '''
        plt.gca().locator_params(axis='x', nbins=len(xticklabels) - 1)
        plt.gca().set_xticklabels(xticklabels, fontsize=9)

    if yticklabels is not None:
        # set the number of ticks to match the length of the labels
        ''' from matplotlib documentation: "the number of ticks <= nbins +1" '''
        plt.gca().locator_params(axis='y', nbins=len(yticklabels) - 1)
        plt.gca().set_yticklabels(yticklabels, fontsize=9)

    plt.xlabel('RA (J2000)')
    plt.ylabel('DEC (J2000)')

    if reverse_xaxis:
        plt.gca().invert_xaxis()

    if save_fig:
        plt.savefig(filename=(file_name + '.' + file_format), format=file_format,
                    dpi=dpi, transparent=True)

    if close_fig:
        plt.close()


def plot_phase_transition_2dirac(metric_mtx, sep_seq, snr_seq,
                                 save_fig, fig_format, file_name,
                                 fig_title='', dpi=300, cmap=None,
                                 color_bar_min=0, color_bar_max=1,
                                 close_fig=True, plt_line=False):
    """
    plot the phase transition for the reconstructions of two Dirac deltas
    :param metric_mtx: a matrix of the aggregated performance. Here the row indices
                correspond to different separations between two Dirac deltas. The column
                indices corresponds to different noise levels.
    :param sep_seq: a sequence that specifies the separation between two Dirac deltas
    :param snr_seq: a sequence of different SNRs tested
    :param save_fig: whether to save figure or not.
    :param fig_format: file format for the saved figure.
    :param file_name: file name
    :param fig_title: title of the figure
    :param color_bar_min: minimum value for the colorbar
    :param color_bar_max: maximum value for the colorbar
    :return:
    """
    fig = plt.figure(figsize=(5, 3), dpi=90)
    ax = plt.axes([0.19, 0.17, 0.72, 0.72])

    if cmap is None:
        cmap = sns.cubehelix_palette(dark=0.95, light=0.1,
                                     start=0, rot=-0.6, as_cmap=True)
    p_hd = ax.matshow(metric_mtx, cmap=cmap, alpha=1,
                      vmin=color_bar_min, vmax=color_bar_max)
    ax.grid(False)

    # the line that shows at least 50% success rate
    if plt_line:
        mask = (metric_mtx >= 0.5).astype('int')
        line_y = np.array([np.where(mask[:, loop])[0][0]
                           for loop in range(mask.shape[1])])
        line_x = np.arange(mask.shape[1])

        fitting_coef = np.polyfit(line_x, line_y, deg=1)
        x_inter = np.linspace(line_x.min(), line_x.max(), num=100)
        y_inter = np.zeros(x_inter.shape)
        for power_of_x, coef in enumerate(fitting_coef[::-1]):
            y_inter += coef * x_inter ** power_of_x

        plt.plot(line_x, line_y, linestyle='', linewidth=2,
                 color=[0, 0, 1], marker='o', ms=2.5)
        plt.plot(x_inter, y_inter, linestyle=':', linewidth=1.5,
                 color=[1, 1, 0], marker='')

    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(snr_seq.size))
    ax.set_xticklabels(['{:g}'.format(snr_loop) for snr_loop in snr_seq])
    ax.set_yticks(np.arange(sep_seq.size))

    ytick_str = []
    for sep_loop in sep_seq:
        use_degree = True if np.degrees(sep_loop) >= 1 else False
        use_miniute = True if np.degrees(sep_loop) * 60 >= 1 else False
        use_second = True if np.degrees(sep_loop) * 3600 >= 1 else False

        # in degree, minute, and second representation
        sep_loop_dms = SkyCoord(
            ra=0, dec=sep_loop, unit=units.radian
        ).to_string('dms').split(' ')[1]
        sep_loop_dms = list(filter(None, re.split('[dms]+', sep_loop_dms)))
        if use_degree:
            sep_loop_dms = (
                '{0}' + '\u00B0' + '{1}' + '\u2032' + '{2:.0f}' + '\u2033'
            ).format(sep_loop_dms[0].lstrip('0'),
                     sep_loop_dms[1].lstrip('0'),
                     float(sep_loop_dms[2]))
        elif use_miniute:
            sep_loop_dms = (
                '{0}' + '\u2032' + '{1:.0f}' + '\u2033'
            ).format(sep_loop_dms[1].lstrip('0'),
                     float(sep_loop_dms[2]))
        elif use_second:
            sep_loop_dms = (
                '{0:.0f}' + '\u2033'
            ).format(float(sep_loop_dms[2]))

        ytick_str.append(sep_loop_dms)

    ax.set_yticklabels(ytick_str)

    plt.xlabel('SNR (dB)')
    plt.ylabel('source separation')
    ax.set_title(fig_title, position=(0.5, 1.01), fontsize=11)

    p_hdc = fig.colorbar(p_hd, orientation='vertical', use_gridspec=False,
                         anchor=(0, 0.5), shrink=1, spacing='proportional')
    p_hdc.ax.tick_params(labelsize=8.5)
    p_hdc.update_ticks()
    ax.set_aspect('auto')
    if save_fig:
        plt.savefig(file_name, format=fig_format, dpi=dpi, transparent=True)

    if close_fig:
        plt.close()


def planar_plot_diracs_plotly(x_plt, y_plt, img_lsq,
                              y_ref=None, x_ref=None, amplitude_ref=None,
                              y_recon=None, x_recon=None, amplitude_recon=None,
                              file_name='planar_recon_2d_dirac.html',
                              open_browser=False):
    plotly.offline.init_notebook_mode()
    surfacecolor = np.real(img_lsq)  # for plotting purposes

    if y_ref is not None and x_ref is not None and amplitude_ref is not None:
        ref_pt_available = True
    else:
        ref_pt_available = False

    if y_recon is not None and x_recon is not None and amplitude_recon is not None:
        recon_pt_available = True
    else:
        recon_pt_available = False

    trace1 = go.Surface(x=np.degrees(x_plt), y=np.degrees(y_plt),
                        surfacecolor=surfacecolor,
                        opacity=1, colorscale='Portland', hoverinfo='none')

    trace1['contours']['x']['highlightwidth'] = 1
    trace1['contours']['y']['highlightwidth'] = 1
    # trace1['contours']['z']['highlightwidth'] = 1

    np.set_printoptions(precision=3, formatter={'float': '{: 0.2f}'.format})
    if ref_pt_available:
        if hasattr(y_ref, '__iter__'):  # <= not a scalar
            text_str2 = []
            for count, y0 in enumerate(y_ref):
                if amplitude_ref.shape[1] > 1:
                    text_str2.append((
                                         u'({0:.2f}\N{DEGREE SIGN}, ' +
                                         u'{1:.2f}\N{DEGREE SIGN}), </br>' +
                                         u'intensity: {2}').format(np.degrees(y0),
                                                                   np.degrees(x_ref[count]),
                                                                   amplitude_ref.squeeze()[count])
                                     )
                else:
                    text_str2.append((
                                         u'({0:.2f}\N{DEGREE SIGN}, ' +
                                         u'{1:.2f}\N{DEGREE SIGN}), </br>' +
                                         u'intensity: {2:.2f}').format(np.degrees(y0),
                                                                       np.degrees(x_ref[count]),
                                                                       amplitude_ref.squeeze()[count])
                                     )

            trace2 = go.Scatter(mode='markers', name='ground truth',
                                x=np.degrees(x_ref),
                                y=np.degrees(y_ref),
                                text=text_str2,
                                hoverinfo='name+text',
                                marker=dict(size=6, symbol='circle', opacity=0.6,
                                            line=dict(
                                                color='rgb(0, 0, 0)',
                                                width=1
                                            ),
                                            color='rgb(255, 255, 255)'))
        else:
            if amplitude_ref.shape[1] > 1:
                text_str2 = [(u'({0:.2f}\N{DEGREE SIGN}, ' +
                              u'{1:.2f}\N{DEGREE SIGN}) </br>' +
                              u'intensity: {2}').format(np.degrees(y_ref),
                                                        np.degrees(x_ref),
                                                        amplitude_ref)]
            else:
                text_str2 = [(u'({0:.2f}\N{DEGREE SIGN}, ' +
                              u'{1:.2f}\N{DEGREE SIGN}) </br>' +
                              u'intensity: {2:.2f}').format(np.degrees(y_ref),
                                                            np.degrees(x_ref),
                                                            amplitude_ref)]
            trace2 = go.Scatter(mode='markers', name='ground truth',
                                x=[np.degrees(x_ref)],
                                y=[np.degrees(y_ref)],
                                text=text_str2,
                                hoverinfo='name+text',
                                marker=dict(size=6, symbol='circle', opacity=0.6,
                                            line=dict(
                                                color='rgb(0, 0, 0)',
                                                width=1
                                            ),
                                            color='rgb(255, 255, 255)'))

    if recon_pt_available:
        if hasattr(y_recon, '__iter__'):
            text_str3 = []
            for count, y0 in enumerate(y_recon):
                if amplitude_recon.shape[1] > 1:
                    text_str3.append((
                                         u'({0:.2f}\N{DEGREE SIGN}, ' +
                                         u'{1:.2f}\N{DEGREE SIGN}) </br>' +
                                         u'intensity: {2}').format(np.degrees(y0),
                                                                   np.degrees(x_recon[count]),
                                                                   amplitude_recon.squeeze()[count])
                                     )
                else:
                    text_str3.append((
                                         u'({0:.2f}\N{DEGREE SIGN}, ' +
                                         u'{1:.2f}\N{DEGREE SIGN}) </br>' +
                                         u'intensity: {2:.2f}').format(np.degrees(y0),
                                                                       np.degrees(x_recon[count]),
                                                                       np.squeeze(amplitude_recon, axis=1)[count])
                                     )

            trace3 = go.Scatter(mode='markers', name='reconstruction',
                                x=np.degrees(x_recon), y=np.degrees(y_recon),
                                text=text_str3,
                                hoverinfo='name+text',
                                marker=dict(size=6, symbol='diamond', opacity=0.6,
                                            line=dict(
                                                color='rgb(0, 0, 0)',
                                                width=1
                                            ),
                                            color='rgb(255, 105, 180)'))
        else:
            if amplitude_recon.shape[1] > 1:
                text_str3 = [(u'({0:.2f}\N{DEGREE SIGN}, '
                              u'{1:.2f}\N{DEGREE SIGN}) </br>' +
                              u'intensity: {2}').format(np.degrees(y_recon),
                                                        np.degrees(x_recon),
                                                        amplitude_recon)]
            else:
                text_str3 = [(u'({0:.2f}\N{DEGREE SIGN}, '
                              u'{1:.2f}\N{DEGREE SIGN}) </br>' +
                              u'intensity: {2:.2f}').format(np.degrees(y_recon),
                                                            np.degrees(x_recon),
                                                            amplitude_recon)]

            trace3 = go.Scatter(mode='markers', name='reconstruction',
                                x=[np.degrees(x_recon)],
                                y=[np.degrees(y_recon)],
                                text=text_str3,
                                hoverinfo='name+text',
                                marker=dict(size=6, symbol='diamond', opacity=0.6,
                                            line=dict(
                                                color='rgb(0, 0, 0)',
                                                width=1
                                            ),
                                            color='rgb(255, 105, 180)'))

    if ref_pt_available and recon_pt_available:
        data = go.Data([trace1, trace2, trace3])
    elif ref_pt_available and not recon_pt_available:
        data = go.Data([trace1, trace2])
    elif not ref_pt_available and recon_pt_available:
        data = go.Data([trace1, trace3])
    else:
        data = go.Data([trace1])

    if ref_pt_available and recon_pt_available:
        dist_recon = planar_distance(x_ref, y_ref, x_recon, y_recon)[0]
        layout = go.Layout(title=u'average error = {0:.2f}\N{DEGREE SIGN}'.format(np.degrees(dist_recon)),
                           titlefont={'family': 'Open Sans, verdana, arial, sans-serif',
                                      'size': 14,
                                      'color': '#000000'},
                           autosize=False, width=670, height=550, showlegend=True,
                           margin=go.Margin(l=45, r=45, b=55, t=45)
                           )
    else:
        layout = go.Layout(title=u'',
                           titlefont={'family': 'Open Sans, verdana, arial, sans-serif',
                                      'size': 14,
                                      'color': '#000000'},
                           autosize=False, width=670, height=550, showlegend=True,
                           margin=go.Margin(l=45, r=45, b=55, t=45)
                           )

    if ref_pt_available or recon_pt_available:
        layout['legend']['xanchor'] = 'center'
        layout['legend']['yanchor'] = 'top'
        layout['legend']['x'] = 0.5

    layout['scene']['camera']['eye'] = {'x': 0, 'y': 0}

    fig = go.Figure(data=data, layout=layout)
    plot(fig, filename=file_name, auto_open=open_browser)
