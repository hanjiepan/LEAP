"""
convert_mat2npz.py: convert Matlab .mat file to numpy .npz file for the data obtained from simulator
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

"""
import numpy as np
import scipy.io as sio
from utils import sph_extract_off_diag

num_subband = 1

mat_file_name_sky = './data/simulated_data_sky.mat'
simulation_data_sky = sio.loadmat(mat_file_name_sky)

RA_rad = simulation_data_sky['sky'][0, 0]['RA_rad'][0, 0]
DEC_rad = simulation_data_sky['sky'][0, 0]['DEC_rad'][0, 0]

mat_file_name3 = './data/simulated_data_101STI_50MHz_noise1000_25src.mat'
simulation_data3 = sio.loadmat(mat_file_name3)

array_coordinate_multiSTI = simulation_data3['position_antennas_slots']
num_sti = array_coordinate_multiSTI.shape[2]
# swap axis so that
# axis0 Cartisian coordinate
# axis1 is STI indices
# axis2 of size 3: for (x,y,z)
array_coordinate_multiSTI = np.swapaxes(array_coordinate_multiSTI, 1, 2)
freq_subbands_hz_subband3 = float(simulation_data3['Frequency'][0, 0])
alpha_ks_subband3 = simulation_data3['sources'][0, 0]['int'].squeeze()
visi_noisy_subband3 = np.reshape(np.column_stack(
    [sph_extract_off_diag(simulation_data3['Sigma_slots'][:, :, sti_idx].T[:, :, np.newaxis])
                       for sti_idx in range(num_sti)
     ]), (-1, num_sti, num_subband), order='F')
img_lsq_subband3 = simulation_data3['LS_Image_multi']

latitude_ks = simulation_data3['sources'][0, 0]['delta'].squeeze()
colatitude_ks = np.pi * 0.5 - latitude_ks  # convert to co-latitude
azimuth_ks = simulation_data3['sources'][0, 0]['alpha'].squeeze()

xgrid_plt = simulation_data3['Grid_X_Image']
ygrid_plt = simulation_data3['Grid_Y_Image']
zgrid_plt = simulation_data3['Grid_Z_Image']
colatitude_plt = np.arccos(zgrid_plt)
azimuth_plt = np.arctan2(ygrid_plt, xgrid_plt)

# save files (as if we have multi-bands)
np.savez('./data/simulated_data_101STI_50MHz_noise1000_25src.npz',
         array_coordinate=array_coordinate_multiSTI,
         colatitude_ks=colatitude_ks,
         azimuth_ks=azimuth_ks,
         alpha_ks=alpha_ks_subband3,
         visi_noisy=visi_noisy_subband3,
         img_lsq=img_lsq_subband3,
         colatitude_plt=colatitude_plt,
         azimuth_plt=azimuth_plt,
         freq_subbands_hz=freq_subbands_hz_subband3,
         RA_rad=RA_rad, DEC_rad=DEC_rad)


# mat_file_name1 = './data/simulated_data_15MHz.mat'
# simulation_data1 = sio.loadmat(mat_file_name1)
#
# mat_file_name2 = './data/simulated_data_50MHz.mat'
# simulation_data2 = sio.loadmat(mat_file_name2)
#
# freq_subbands_hz_subband1 = float(simulation_data1['Frequency'][0, 0])
# freq_subbands_hz_subband2 = float(simulation_data2['Frequency'][0, 0])
# freq_subbands_hz = np.array([freq_subbands_hz_subband1, freq_subbands_hz_subband2])
#
# latitude_ks = simulation_data1['sources'][0, 0]['delta'].squeeze()
# colatitude_ks = np.pi * 0.5 - latitude_ks  # convert to co-latitude
# azimuth_ks = simulation_data1['sources'][0, 0]['alpha'].squeeze()
# alpha_ks_subband1 = simulation_data1['sources'][0, 0]['int'].squeeze()
# alpha_ks_subband2 = simulation_data2['sources'][0, 0]['int'].squeeze()
#
# visi_noisy_subband1 = sph_extract_off_diag(simulation_data1['Sigma'].T[:, :, np.newaxis])
# visi_noisy_subband2 = sph_extract_off_diag(simulation_data2['Sigma'].T[:, :, np.newaxis])
#
# img_lsq_subband1 = simulation_data1['LS_Image']
# img_lsq_subband2 = simulation_data2['LS_Image']
# img_lsq = img_lsq_subband1 + img_lsq_subband2
#
# xgrid_plt = simulation_data1['Grid_X_Image']
# ygrid_plt = simulation_data1['Grid_Y_Image']
# zgrid_plt = simulation_data1['Grid_Z_Image']
# colatitude_plt = np.arccos(zgrid_plt)
# azimuth_plt = np.arctan2(ygrid_plt, xgrid_plt)
#
# # save files (as if we have multi-bands)
# np.savez('./data/simulated_data_15MHz.npz',
#          array_coordinate=array_coordinate,
#          colatitude_ks=colatitude_ks,
#          azimuth_ks=azimuth_ks,
#          alpha_ks=alpha_ks_subband1,
#          visi_noisy=visi_noisy_subband1,
#          img_lsq=img_lsq_subband1,
#          colatitude_plt=colatitude_plt,
#          azimuth_plt=azimuth_plt,
#          freq_subbands_hz=freq_subbands_hz_subband1,
#          RA_rad=RA_rad, DEC_rad=DEC_rad)
#
# np.savez('./data/simulated_data_50MHz.npz',
#          array_coordinate=array_coordinate,
#          colatitude_ks=colatitude_ks,
#          azimuth_ks=azimuth_ks,
#          alpha_ks=alpha_ks_subband2,
#          visi_noisy=visi_noisy_subband2,
#          img_lsq=img_lsq_subband2,
#          colatitude_plt=colatitude_plt,
#          azimuth_plt=azimuth_plt,
#          freq_subbands_hz=freq_subbands_hz_subband2,
#          RA_rad=RA_rad, DEC_rad=DEC_rad)
#
# np.savez('./data/simulated_data_15_50MHz.npz',
#          array_coordinate=array_coordinate,
#          colatitude_ks=colatitude_ks,
#          azimuth_ks=azimuth_ks,
#          alpha_ks=np.column_stack((alpha_ks_subband1, alpha_ks_subband2)),
#          visi_noisy=np.column_stack((visi_noisy_subband1, visi_noisy_subband2)),
#          img_lsq=img_lsq,
#          colatitude_plt=colatitude_plt,
#          azimuth_plt=azimuth_plt,
#          freq_subbands_hz=freq_subbands_hz,
#          RA_rad=RA_rad,
#          DEC_rad=DEC_rad)
