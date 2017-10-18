"""
fits_img2npz.py: convert FITS image to npz
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
import re
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy import units
from astropy.coordinates import SkyCoord

if __name__ == '__main__':
    img_file_name = '/Users/hpa/Google Drive/RadioAstData/highres-image.fits'
    with fits.open(img_file_name) as handle:
        handle.info()
        # FITS header info.
        img_header = handle['PRIMARY'].header

        num_pixel_RA = img_header['NAXIS1']
        RA_center_pixel_idx = int(img_header['CRPIX1']) - 1  # python index from 0
        RA_center_degree = np.mod(img_header['CRVAL1'], 360)
        RA_step_size = img_header['CDELT1']

        num_pixel_DEC = img_header['NAXIS2']
        DEC_center_pixel_idx = int(img_header['CRPIX2']) - 1
        DEC_center_degree = img_header['CRVAL2']
        DEC_step_size = img_header['CDELT2']

        # FITS data
        img_data = handle[0].data.squeeze()

    # extract the associated dirty image
    with fits.open(img_file_name.split('-')[0] + '-dirty.fits') as handle:
        dirty_img = handle[0].data.squeeze()

    # create plotting grid vector for RA and DEC
    RA_plt_vec = (np.arange(num_pixel_RA) - RA_center_pixel_idx) * \
                 RA_step_size + RA_center_degree
    DEC_plt_vec = (np.arange(num_pixel_DEC) - DEC_center_pixel_idx) * \
                  DEC_step_size + DEC_center_degree

    RA_plt_vec = np.mod(RA_plt_vec, 360)
    DEC_plt_vec = np.mod(DEC_plt_vec, 360)

    RA_plt_grid, DEC_plt_grid = np.meshgrid(RA_plt_vec, DEC_plt_vec)

    axes = plt.figure(figsize=(5, 4), dpi=300).add_subplot(111)
    plt.pcolormesh(RA_plt_grid, DEC_plt_grid, img_data,
                   shading='gouraud', cmap='Spectral_r')
    plt.xlabel('RA (J2000)')
    plt.ylabel('DEC (J2000)')

    xlabels_original = axes.get_xticks().tolist()

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
                '{0}' + 'h' + '{1}' + 'm'
            ).format(xlabels_dms[0], xlabels_dms[1])
        else:
            xlabels_dms = (
                '{0}' + 'm'
            ).format(xlabels_dms[1])

        xlabels_hms_all.append(xlabels_dms)

    ylabels_original = axes.get_yticks().tolist()
    ylabels_all = [('{0:.0f}' + '\u00B0').format(ylabels_loop)
                   for ylabels_loop in ylabels_original]

    axes.set_xticklabels(xlabels_hms_all)
    axes.set_yticklabels(ylabels_all)
    plt.axis('image')

    file_name = img_file_name.split('.')[0]
    file_format = 'png'
    dpi = 600
    plt.savefig(filename=(file_name + '.' + file_format), format=file_format,
                dpi=dpi, transparent=True)
    plt.close()

    axes = plt.figure(figsize=(5, 4), dpi=300).add_subplot(111)
    plt.pcolormesh(RA_plt_grid, DEC_plt_grid, dirty_img,
                   shading='gouraud', cmap='Spectral_r')
    plt.xlabel('RA (J2000)')
    plt.ylabel('DEC (J2000)')

    axes.set_xticklabels(xlabels_hms_all)
    axes.set_yticklabels(ylabels_all)
    plt.axis('image')

    file_name = img_file_name.split('-')[0] + '-dirty'
    file_format = 'png'
    dpi = 600
    plt.savefig(filename=(file_name + '.' + file_format), format=file_format,
                dpi=dpi, transparent=True)
    plt.close()

    # save image data as well as plotting axis labels
    '''
    here we flip the x-axis. in radioastronomy, the convention is that RA (the x-axis)
    DECREASES from left to right.
    By flipping the x-axis, RA INCREASES from left to right.
    '''
    np.savez(
        './data/CLEAN_data.npz',
        x_plt_CLEAN_rad=np.radians(RA_plt_grid),
        y_plt_CLEAN_rad=np.radians(DEC_plt_grid),
        x_plt_centered_rad=np.radians(RA_plt_grid[:, ::-1] - RA_center_degree),
        y_plt_centered_rad=np.radians(DEC_plt_grid - DEC_center_degree),
        img_clean=img_data[:, ::-1],
        img_dirty=dirty_img[:, ::-1],
        xlabels_hms_all=xlabels_hms_all,
        ylabels_dms_all=ylabels_all
    )
