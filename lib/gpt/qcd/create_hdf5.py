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


def _check_meas(name_hdf5, measurement_spectrum):
    ''' Assert existence of the measurement, 
       create otherwise and return the name of the source measurement'''

    with h5py.File(name_hdf5, 'a') as hdf5:
        if measurement_spectrum not in hdf5.keys():
            grp1 = hdf5.create_group(measurement_spectrum)
        grp1 = hdf5[measurement_spectrum]
        n = 0
        tmp = True
        while (tmp == True):
            meas_tsrc = "meas" + str(n)
            if meas_tsrc not in grp1.keys():
                tmp = False
                grp2 = grp1.create_group(meas_tsrc)
            n += 1
    return meas_tsrc


def _check_propset(name_hdf5, spectrum_meas, source_meas, propset_meas):
    ''' Assert existence of the measurement, 
       create otherwise and return the name of the source measurement'''

    hdf5 = h5py.File(name_hdf5, 'a')
    grp1 = hdf5[spectrum_meas]
    grp2 = grp1[source_meas]
    grp2.create_group(propset_meas)


def _write_hdf5dset_baryon(correlators, data_file, tsrc_meas):
    ''' The correlators have dimension [n_baryons * 2(time_rev)][#moms][Nt] '''

    n_baryons = correlators.shape[0]
    moms = correlators.shape[1]
    Nt = correlators.shape[-1]

    hdf5_file = h5py.File(data_file, 'a')
    spectrum_group = hdf5_file["baryonspec"]
    tsrc_group = spectrum_group[tsrc_meas]

    if "data" not in tsrc_group.keys():
        dset = tsrc_group.create_dataset("data",
                                      correlators.shape,
                                      dtype=complex,
                                      chunks=True,
                                      fletcher32=True)
    else:
        dset = tsrc_group["data"]

    dset[...] = correlators[:]
    hdf5_file.close()


def _write_hdf5dset_meson(correlators, data_file, tsrc_meas, propset_meas):

    n_mesons = correlators.shape[0]
    moms = correlators.shape[1]
    Nt = correlators.shape[-1]

    hdf5_file = h5py.File(data_file, 'a')
    spectrum_meas_grp = hdf5_file["mesonspec"]
    source_meas_grp = spectrum_meas_grp[tsrc_meas]
    propset_meas_grp = source_meas_grp[propset_meas]

    if "data" not in propset_meas_grp.keys():
        dset = propset_meas_grp.create_dataset("data",
                                      correlators.shape,
                                      dtype=complex,
                                      chunks=True,
                                      fletcher32=True)
    else:
        dset = propset_meas_grp["data"]

    dset[...] = correlators[:]
    hdf5_file.close()


