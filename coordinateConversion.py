"""
coordinateConversion.py: convert coordinates between J2000 and UVW
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
import numpy as np

def sph2cart(az, el, r):
    """
    Transform spherical coordinates into Cartesian coordinates.
    :param az: vector in [0,2\pi]
    :param el: vector in [-\pi/2,\pi/2]
    :param r: vector in R_{+}
    """
    rcos_theta = r * np.cos(el)
    (x, y, z) = (rcos_theta * np.cos(az), rcos_theta * np.sin(az), r * np.sin(el))
    return (x, y, z)


def cart2sph(x, y, z):
    """
    Transform Cartesian coordinates to spherical coordinates.
    This function is the inverse of sph2cart().
    """
    hxy = np.sqrt(x ** 2 + y ** 2)
    (r, el, az) = (np.hypot(hxy, z), np.arctan2(z, hxy), np.arctan2(y, x))
    return (az, el, r)


def J2000_to_UVW_operator(az, el):
    """
    Create the (3,3) change-of-basis matrix to go from J2000
    coordinates to UVW coordinates.
    :param az: scalar in [0,2\pi]
    :param el: scalar in [-\pi/2,\pi/2]
    :return: (3,3) change-of-basis matrix
    """
    M = np.array([
        [-np.sin(az), np.cos(az), 0],
        [-np.cos(az) * np.sin(el), -np.sin(az) * np.sin(el), np.cos(el)],
        sph2cart(az, el, 1)
    ])
    return M
