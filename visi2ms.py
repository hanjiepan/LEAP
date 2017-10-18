"""
visi2ms.py: write visibility to an MS file
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
import subprocess
import sys
import re
import os
import numpy as np
from astropy.io import fits
from astropy import units
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import matplotlib

if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

if sys.version_info[0] > 2:
    sys.exit('Sorry casacore only runs on Python 2.')
else:
    from casacore import tables as casa_tables


def run_wsclean(ms_file, channel_range=(0, 1), mgain=0.7, FoV=5,
                max_iter=1000, auto_threshold=3, threshold=None,
                output_img_size=512, intermediate_size=1024,
                output_name_prefix='highres', quiet=False,
                run_cs=False, cs_gain=0.1):
    """
    run wsclean algorithm for a given measurement set (ms file).
    :param ms_file: name of the measurement set
    :param channel_range: frequency channel range.
            The first one is inclusive but the second one is exclusive, i.e., (0, 1) select channel0 only.
    :param mgain: gain for the CLEAN iteration
    :param FoV: field of view (in degree). Default 5 degree
    :param max_iter: maximum number of iterations of the CLEAN algorithm
    :param output_img_size: output image size
    :param intermediate_size: intermediate image size (must be no smaller than output_img_size)
    :param output_name_prefix: prefix for various outputs from wsclean
    :return:
    """
    if quiet:
        wsclean_mode = 'wsclean -quiet'
        print('Running wsclean ...')
    else:
        wsclean_mode = 'wsclean'

    assert output_img_size <= intermediate_size
    pixel_size = FoV * 3600 / output_img_size  # in arcsecond

    if run_cs:
        wsclean_mode += ' -iuwt -gain {cs_gain} '.format(cs_gain=cs_gain)

    bash_cmd = '{wsclean} ' \
               '-size {intermediate_size} {intermediate_size} ' \
               '-trim {output_img_size} {output_img_size} ' \
               '-scale {pixel_size}asec ' \
               '-name {output_name_prefix} ' \
               '-datacolumn DATA ' \
               '-channelrange {freq_channel_min} {freq_channel_max} ' \
               '-niter {max_iter} ' \
               '-mgain {mgain} ' \
               '-pol I ' \
               '-weight briggs 0.0 ' \
               '-weighting-rank-filter 3 '.format(
        wsclean=wsclean_mode,
        intermediate_size=intermediate_size,
        output_img_size=output_img_size,
        pixel_size=repr(pixel_size),
        output_name_prefix=output_name_prefix,
        freq_channel_min=channel_range[0],
        freq_channel_max=channel_range[1],
        max_iter=max_iter,
        mgain=mgain,
        auto_threshold=auto_threshold
    )
    if threshold is None:
        bash_cmd += '-auto-threshold {auto_threshold} '.format(auto_threshold=auto_threshold)
    else:
        bash_cmd += '-threshold {threshold} '.format(threshold=threshold)

    bash_cmd += '{ms_file} '.format(ms_file=ms_file)

    # run wsclean
    exitcode = subprocess.call(bash_cmd, shell=True)

    return exitcode


def convert_clean_outputs(clean_output_prefix, result_image_prefix,
                          result_data_prefix, fig_file_format='png', dpi=600):
    """
    convert the FITS images from wsclean to numpy array
    :param clean_output_prefix: prefix of wsCLEAN outputs
    :param result_image_prefix: prefix to be used for the converted numpy array
    :return:
    """
    # CLEAN point sources based one '-model.fits'
    with fits.open(clean_output_prefix + '-model.fits') as handle:
        # FITS data
        src_model = handle[0].data.squeeze()

    # CLEANed image
    with fits.open(clean_output_prefix + '-image.fits') as handle:
        # handle.info()
        # FITS header info.
        img_header = handle['PRIMARY'].header

        # convert to world coordinate
        w = WCS(img_header)

        num_pixel_RA = img_header['NAXIS1']
        num_pixel_DEC = img_header['NAXIS2']
        RA_mesh, DEC_mesh = np.meshgrid(np.arange(num_pixel_RA) + 1,
                                        np.arange(num_pixel_DEC) + 1)
        pixcard = np.column_stack((RA_mesh.flatten('F'), DEC_mesh.flatten('F')))
        RA_DEC_plt = w.dropaxis(3).dropaxis(2).all_pix2world(pixcard, 1)
        RA_plt_grid = np.reshape(RA_DEC_plt[:, 0], (-1, num_pixel_RA), order='F')
        DEC_plt_grid = np.reshape(RA_DEC_plt[:, 1], (-1, num_pixel_DEC), order='F')

        # FITS data
        img_data = handle[0].data.squeeze()

    # dirty image
    with fits.open(clean_output_prefix + '-dirty.fits') as handle:
        dirty_img = handle[0].data.squeeze()

    plt.figure(figsize=(5, 4), dpi=300).add_subplot(111)
    plt.gca().locator_params(axis='x', nbins=6)
    plt.pcolormesh(RA_plt_grid, DEC_plt_grid, img_data,
                   shading='gouraud', cmap='Spectral_r')
    plt.xlabel('RA (J2000)')
    plt.ylabel('DEC (J2000)')
    plt.gca().invert_xaxis()

    xlabels_original = plt.gca().get_xticks().tolist()
    ylabels_original = plt.gca().get_yticks().tolist()
    plt.close()

    # in degree, minute, and second representation
    xlabels_hms_all = []
    for lable_idx, xlabels_original_loop in enumerate(xlabels_original):
        xlabels_original_loop = float(xlabels_original_loop)

        xlabels_dms = SkyCoord(
            ra=xlabels_original_loop, dec=0, unit=units.degree
        ).to_string('hmsdms').split(' ')[0]
        xlabels_dms = list(filter(None, re.split('[hms]+', xlabels_dms)))
        if lable_idx == 1:
            xlabels_dms = (
                u'{0}h{1}m{2}s'
            ).format(xlabels_dms[0], xlabels_dms[1], xlabels_dms[2])
        else:
            xlabels_dms = (
                u'{0}m{1}s'
            ).format(xlabels_dms[1], xlabels_dms[2])

        xlabels_hms_all.append(xlabels_dms)

    ylabels_all = [(u'{0:.2f}' + u'\u00B0').format(ylabels_loop)
                   for ylabels_loop in ylabels_original]

    # use the re-formatted ticklabels to plot the figure again
    plt.figure(figsize=(5, 4), dpi=300).add_subplot(111)
    plt.pcolormesh(RA_plt_grid, DEC_plt_grid, img_data,
                   shading='gouraud', cmap='Spectral_r')
    plt.xlabel('RA (J2000)')
    plt.ylabel('DEC (J2000)')
    plt.gca().invert_xaxis()

    plt.gca().set_xticklabels(xlabels_hms_all, fontsize=9)
    plt.gca().set_yticklabels(ylabels_all, fontsize=9)
    plt.axis('image')

    file_name = result_image_prefix + '-image.' + fig_file_format
    plt.savefig(filename=file_name, format=fig_file_format,
                dpi=dpi, transparent=True)
    plt.close()

    plt.figure(figsize=(5, 4), dpi=300).add_subplot(111)
    plt.pcolormesh(RA_plt_grid, DEC_plt_grid, dirty_img,
                   shading='gouraud', cmap='Spectral_r')
    plt.xlabel('RA (J2000)')
    plt.ylabel('DEC (J2000)')
    plt.gca().invert_xaxis()

    plt.gca().set_xticklabels(xlabels_hms_all, fontsize=9)
    plt.gca().set_yticklabels(ylabels_all, fontsize=9)
    plt.axis('image')

    file_name = result_image_prefix + '-dirty.' + fig_file_format
    plt.savefig(filename=file_name, format=fig_file_format,
                dpi=dpi, transparent=True)
    plt.close()

    # save image data as well as plotting axis labels
    '''
    here we flip the x-axis. in radioastronomy, the convention is that RA (the x-axis)
    DECREASES from left to right.
    By flipping the x-axis, RA INCREASES from left to right.
    '''
    CLEAN_data_file = result_data_prefix + '-CLEAN_data.npz'
    np.savez(
        CLEAN_data_file,
        x_plt_CLEAN_rad=np.radians(RA_plt_grid),
        y_plt_CLEAN_rad=np.radians(DEC_plt_grid),
        img_clean=img_data,
        img_dirty=dirty_img,
        src_model=src_model,
        xlabels_hms_all=xlabels_hms_all,
        ylabels_dms_all=ylabels_all
    )

    return CLEAN_data_file


def update_visi_msfile(reference_ms_file, modified_ms_file,
                       visi, antenna1_idx, antenna2_idx, num_station):
    """
    update a reference ms file with a new visibility data
    :param reference_ms_file: the original ms file
    :param modified_ms_file: the modified ms file with the updated visibilities
    :param visi: new visibilities to be put in the copied ms file
    :param antenna1_idx: coordinate of the first antenna of the visibility measurements
    :param antenna2_idx: coordinate of the second antenna of the visibility measurements
    :return:
    """
    print('Copying table for modifications ...')
    casa_tables.tablecopy(reference_ms_file, modified_ms_file)
    print('Modifying visibilities in the new table ...')
    with casa_tables.table(modified_ms_file, readonly=False, ack=False) as modified_table:
        # swap axis so that:
        # axis 0: cross-correlation index;
        # axis 1: subband index;
        # axis 2: STI index
        visi = np.swapaxes(visi, 1, 2)
        num_bands, num_sti = visi.shape[1:]

        row_count = 0
        for sti_count in range(num_sti):
            visi_loop = np.zeros((num_station, num_station, num_bands), dtype=complex)
            visi_loop[antenna1_idx, antenna2_idx, :] = visi[:, :, sti_count]
            # so that axis 0: subband index;
            # axis 1: cross-correlation index1
            # axis 2: cross-corrleation index2
            visi_loop = visi_loop.swapaxes(2, 0).swapaxes(2, 1)
            for station2 in range(num_station):
                for station1 in range(station2 + 1):
                    # dimension: num_subband x 4 (4 different polarisations: XX, XY, YX, YY)
                    visi_station1_station2 = modified_table.getcell('DATA', rownr=row_count)
                    visi_station1_station2[:num_bands, :] = \
                        visi_loop[:, station1, station2][:, np.newaxis]
                    # visi_station1_station2[:, :] = \
                    #     visi_loop[:, station1, station2][:, np.newaxis]
                    # update visibility in the table
                    modified_table.putcell('DATA', rownr=row_count,
                                           value=visi_station1_station2)
                    flag_station1_station2 = modified_table.getcell('FLAG', rownr=row_count)
                    flag_station1_station2[:num_bands, :] = False
                    modified_table.putcell('FLAG', rownr=row_count,
                                             value=flag_station1_station2)
                    row_count += 1

        assert modified_table.nrows() == row_count  # sanity check


if __name__ == '__main__':
    # for testing purposes
    reference_ms_file = '/home/hpa/Documents/Data/BOOTES24_SB180-189.2ch8s_SIM_every50th.ms'
    modified_ms_file = '/home/hpa/Documents/Data/BOOTES24_SB180-189.2ch8s_SIM_every50th_modi.ms'
    num_station = 24
    num_sti = 63
    num_bands = 1
    visi = np.random.randn(num_station * (num_station - 1), num_sti, num_bands) + \
           1j * np.random.randn(num_station * (num_station - 1), num_sti, num_bands)

    mask_mtx = (1 - np.eye(num_station, dtype=int)).astype(bool)
    antenna2_idx, antenna1_idx = np.meshgrid(np.arange(num_station), np.arange(num_station))
    antenna1_idx = np.extract(mask_mtx, antenna1_idx)
    antenna2_idx = np.extract(mask_mtx, antenna2_idx)

    update_visi_msfile(reference_ms_file, modified_ms_file,
                       visi, antenna1_idx, antenna2_idx, num_station)
