import gpt
from gpt.qcd.hadron.quarkcontract import quark_contract_13


# TODO clean up and check
def contract_one_flavor_baryon(prop_1, prop_2, prop_3, pol_matrix, source_spin_matrix, sink_spin_matrix):
    di_quark = quark_contract_13(gpt.eval(prop_3 * source_spin_matrix), gpt.eval(sink_spin_matrix * prop_2))
    return gpt.trace(
        gpt.eval(pol_matrix * gpt.color_trace(prop_1 * gpt.spin_trace(di_quark))) +
        gpt.eval(pol_matrix * gpt.color_trace(prop_1 * di_quark))
    )


# TODO clean up and check
def contract_two_flavor_baryon(prop_1, prop_2, prop_3, pol_matrix, source_spin_matrix, sink_spin_matrix):
    di_quark = quark_contract_13(gpt.eval(prop_3 * source_spin_matrix), gpt.eval(sink_spin_matrix * prop_2))
    return gpt.trace(
        gpt.eval(pol_matrix * gpt.color_trace(prop_1 * gpt.spin_trace(di_quark))) +
        gpt.eval(pol_matrix * gpt.color_trace(prop_1 * di_quark))
    )


# TODO clean up and check
def contract_three_flavor_baryon(prop_1, prop_2, prop_3, pol_matrix, source_spin_matrix, sink_spin_matrix):
    di_quark = quark_contract_13(gpt.eval(prop_1 * source_spin_matrix), gpt.eval(sink_spin_matrix * prop_2))
    return gpt.trace(gpt.eval(pol_matrix * gpt.color_trace(prop_3 * gpt.spin_trace(di_quark))))
