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
