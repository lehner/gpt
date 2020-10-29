#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2020  Lorenzo Barca    (lorenzo1.barca@ur.de)
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import gpt as g
import numpy as np
import h5py


def _check_meas(hdf5):
    ''' Assert existence of the measurement, create otherwise and return the hdf5 group '''

    if "barspec" not in hdf5.keys():
        grp1 = hdf5.create_group("barspec")
    grp1 = hdf5["barspec"]
    n = 0
    e = True
    while (e == True):
        meas = "meas" + str(n)
        if meas not in grp1.keys():
            e = False
            grp2 = grp1.create_group(meas)
        n += 1
    return grp2



def _write_hdf5dset(correlators, file):
    ''' The correlators have dimension [n_baryons * 2(n_time_rev)][#moms][Nt] '''

    n_baryons = correlators.shape[0]
    moms = correlators.shape[1]
    Nt = correlators.shape[-1]

    with h5py.File(file, 'a') as hdf5_file:
        group_meas = _check_meas(hdf5_file)

        # Add attributes
#        dset.attrs['title'] = '2-point function for baryon spectrum'
#        dset.attrs['su(n)'] = 'SU(' + str(suN) + ')'
#        dset.attrs['quarks'] = quarks_list
#        dset.attrs['kappa'] = kappa_list

        print(group_meas)
        offset = 0
        if "data" not in group_meas.keys():
            dset = group_meas.create_dataset("data",
                                      correlators.shape,
                                      dtype=complex,
                                      chunks=True,
                                      fletcher32=True)
        else:
            dset = group_meas["data"]
            offset = 1

        dset[...] = correlators[:]

