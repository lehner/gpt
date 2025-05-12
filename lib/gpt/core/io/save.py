#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
#
import gpt, cgpt
from gpt.params import params_convention


# format
class format:
    # gpt general purpose format
    class gpt:
        @params_convention(mpi=None)
        def __init__(self, params):
            self.params = params

    # lime general purpose format
    class lime:
        @params_convention(binary_data_tag="gpt-binary-data", tags={}, tag_order=[])
        def __init__(self, params):
            self.params = params

    # lattice QCD specific file formats
    class cevec:
        @params_convention(nsingle=None, max_read_blocks=None, mpi=None)
        def __init__(self, params):
            self.params = params

    class nersc:
        @params_convention(label="", id="gpt", sequence_number=1)
        def __init__(self, params):
            self.params = params

    grid_scidac = lime(
        binary_data_tag="ildg-binary-data",
        tag_order=["grid-format", "scidac-record-xml", "scidac-private-record-xml"],
        tags={
            "scidac-private-record-xml": """<scidacRecord><version>1</version><date/><recordtype>0</recordtype><datatype>0</datatype><precision>0</precision>
            <colors>0</colors><spins>0</spins><typesize>0</typesize><datacount>0</datacount></scidacRecord>""",
            "scidac-record-xml": """<emptyUserRecord><dummy>0</dummy></emptyUserRecord>""",
            "grid-format": """<FieldMetaData><nd>0</nd><dimension/><boundary/><data_start>0</data_start><hdr_version/><storage_format/><link_trace>0</link_trace><plaquette>0</plaquette><checksum>0</checksum>
            <scidac_checksuma>0</scidac_checksuma><scidac_checksumb>0</scidac_checksumb><sequence_number>0</sequence_number><data_type/><ensemble_id/><ensemble_label/><ildg_lfn/><creator/><creator_hardware/>
            <creation_date/><archive_date/><floating_point/></FieldMetaData>""",
        },
    )


# output
def save(filename, objs, fmt=format.gpt()):
    if isinstance(fmt, format.gpt):
        return gpt.core.io.gpt_io.save(filename, objs, fmt.params)
    elif isinstance(fmt, format.cevec):
        return gpt.core.io.cevec_io.save(filename, objs, fmt.params)
    elif isinstance(fmt, format.lime):
        return gpt.core.io.lime_io.save(filename, objs, fmt.params)
    elif isinstance(fmt, format.nersc):
        return gpt.core.io.nersc_io.save(filename, objs, fmt.params)

    return cgpt.save(filename, objs, fmt, gpt.default.is_verbose("io"))
