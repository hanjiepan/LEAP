"""
convert_catalog2uvw.py: convert J2000 coordinates in a catalog to UVW coordinates
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

------------------------------------------------------------

Description of the data column (type and unit).
------------------------------------------------------------
XTENSION= 'BINTABLE'           / binary table extension                         
BITPIX  =                    8 / array data type                                
NAXIS   =                    2 / number of array dimensions                     
NAXIS1  =                   91 / length of dimension 1                          
NAXIS2  =               623604 / length of dimension 2                          
PCOUNT  =                    0 / number of group parameters                     
GCOUNT  =                    1 / number of groups                               
TFIELDS =                   18 / number of table fields                         
TTYPE1  = 'Source_name'                                                         
TFORM1  = '24A     '                                                            
TTYPE2  = 'RA      '                                                            
TFORM2  = 'E       '                                                            
TTYPE3  = 'E_RA    '                                                            
TFORM3  = 'E       '                                                            
TTYPE4  = 'DEC     '                                                            
TFORM4  = 'E       '                                                            
TTYPE5  = 'E_DEC   '                                                            
TFORM5  = 'E       '                                                            
TTYPE6  = 'Total_flux'                                                          
TFORM6  = 'E       '                                                            
TTYPE7  = 'E_Total_flux'                                                        
TFORM7  = 'E       '                                                            
TTYPE8  = 'Peak_flux'                                                           
TFORM8  = 'E       '                                                            
TTYPE9  = 'E_Peak_flux'                                                         
TFORM9  = 'E       '                                                            
TTYPE10 = 'Maj     '                                                            
TFORM10 = 'E       '                                                            
TTYPE11 = 'E_Maj   '                                                            
TFORM11 = 'E       '                                                            
TTYPE12 = 'Min     '                                                            
TFORM12 = 'E       '                                                            
TTYPE13 = 'E_Min   '                                                            
TFORM13 = 'E       '                                                            
TTYPE14 = 'PA      '                                                            
TFORM14 = 'E       '                                                            
TTYPE15 = 'E_PA    '                                                            
TFORM15 = 'E       '                                                            
TTYPE16 = 'RMS_noise'                                                           
TFORM16 = 'E       '                                                            
TTYPE17 = 'Source_code'                                                         
TFORM17 = '1A      '                                                            
TTYPE18 = 'Mosaic_name'                                                         
TFORM18 = '6A      '                                                            
NAME    = 'TGSSADR1_7sigma_catalog_v3'                                          
TUNIT1  = ''                                                                    
TUNIT2  = 'deg     '                                                            
TUNIT3  = 'arcsec  '                                                            
TUNIT4  = 'deg     '                                                            
TUNIT5  = 'arcsec  '                                                            
TUNIT6  = 'mJy     '                                                            
TUNIT7  = 'mJy     '                                                            
TUNIT8  = 'beam-1 mJy'                                                          
TUNIT9  = 'beam-1 mJy'                                                          
TUNIT10 = 'arcsec  '                                                            
TUNIT11 = 'arcsec  '                                                            
TUNIT12 = 'arcsec  '                                                            
TUNIT13 = 'arcsec  '                                                            
TUNIT14 = 'deg     '                                                            
TUNIT15 = 'deg     '                                                            
TUNIT16 = 'beam-1 mJy'                                                          
TUNIT17 = ''                                                                    
TUNIT18 = ''
"""
from __future__ import division
import setup
import os
import sys
import subprocess
import numpy as np
from astropy.io import fits
from coordinateConversion import J2000_to_UVW_operator

if sys.version_info[0] > 2:
    sys.exit('Sorry casacore only runs on Python 2.')
else:
    from casacore import tables as casa_tables

if __name__ == '__main__':
    catalog = 'tgss'  # 'tgss' or 'nvss'
    parameter_set = {}
    if catalog == 'tgss':
        parameter_set['catalog_basefile_name'] = 'TGSSADR1_7sigma_catalog'
        file_ext = '.fits'
        # column count in the catalog fits table that corresponds to RA, DEC, and Flux
        parameter_set['RA_DEC_Flux_idx'] = [1, 3, 5]
    elif catalog == 'nvss':
        parameter_set['catalog_basefile_name'] = 'NVSS_CATALOG'
        file_ext = '.fit'
        parameter_set['RA_DEC_Flux_idx'] = [0, 1, 2]
    elif catalog == 'bootes':
        parameter_set['catalog_basefile_name'] = 'LOFAR150_BOOTES'
        file_ext = '.fits'
        parameter_set['RA_DEC_Flux_idx'] = [1, 3, 7]
    else:
        RuntimeError('Unknown catalog: {}'.format(catalog))

    data_root_path = os.environ['DATA_ROOT_PATH']
    basefile_name = 'RX42_SB100-109.2ch10s'
    # basefile_name = 'BOOTES24_SB180-189.2ch8s_SIM'
    catalog_basefile_name = parameter_set['catalog_basefile_name']

    ms_file_name = data_root_path + basefile_name + '.ms'
    catalog_file = data_root_path + catalog_basefile_name + file_ext
    npz_catalog_name = data_root_path + catalog_basefile_name + '.npz'

    # pointing direction in radian
    pointingDirection = casa_tables.taql(
        'select REFERENCE_DIR from {msFile}::FIELD'.format(msFile=ms_file_name)
    ).getcol('REFERENCE_DIR').squeeze()
    M = J2000_to_UVW_operator(*pointingDirection)

    # index in the fits table where RA, DEC and Flux are
    RA_DEC_Flux_idx = parameter_set['RA_DEC_Flux_idx']

    with fits.open(catalog_file) as handle:
        # print out a few information about the HDU file and the data column description
        handle.info()
        data_description = handle[1].header
        data_description
        # the actual data
        data = handle[1].data
        # extract data <- verify with the data_description that the correct data is extracted
        # RA and DEC are in J2000 coordinate (in degrees)
        src_RA_DEC_Flux = \
            np.row_stack([[src_data[RA_DEC_Flux_idx[0]],
                           src_data[RA_DEC_Flux_idx[1]],
                           src_data[RA_DEC_Flux_idx[2]]]
                          for src_data in data])

        # convert to radian
        RA_radian = np.radians(src_RA_DEC_Flux[:, 0])
        DEC_radian = np.radians(src_RA_DEC_Flux[:, 1])

        cos_DEC = np.cos(DEC_radian)
        src_x = cos_DEC * np.cos(RA_radian)
        src_y = cos_DEC * np.sin(RA_radian)
        src_z = np.sin(DEC_radian)

        src_u, src_v, src_w = np.dot(M, np.vstack((src_x, src_y, src_z)))

        # save as npz file
        np.savez(npz_catalog_name,
                 Intensities_skyctalog=src_RA_DEC_Flux[:, 2],
                 U_skycatalog=src_u,
                 V_skycatalog=src_v,
                 W_skycatalog=src_w)
