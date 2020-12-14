import gpt
from gpt.qcd.hadron.quarkcontract import quark_contract_13
from gpt.qcd.hadron.spinmatrices import charge_conjugation, gamma_minus


def contract_sigma_plus_star(prop_up, prop_strange, pol_matrix):
    cgminus = charge_conjugation() * gamma_minus()

    di_quark = quark_contract_13(gpt.eval(prop_up * cgminus), gpt.eval(cgminus * prop_strange))
    contraction = gpt.trace(
        gpt.eval(pol_matrix * gpt.color_trace(prop_strange * gpt.spin_trace(di_quark))) +
        gpt.eval(pol_matrix * gpt.color_trace(prop_strange * di_quark))
    )

    di_quark = quark_contract_13(gpt.eval(prop_strange * cgminus), gpt.eval(cgminus * prop_up))
    contraction += gpt.trace(gpt.eval(pol_matrix * gpt.color_trace(prop_strange * di_quark)))

    di_quark = quark_contract_13(gpt.eval(prop_strange * cgminus), gpt.eval(cgminus * prop_strange))
    contraction += gpt.trace(gpt.eval(pol_matrix * gpt.color_trace(prop_up * di_quark)))
    contraction *= 2

    contraction += gpt.trace(gpt.eval(pol_matrix * gpt.color_trace(prop_up * gpt.spin_trace(di_quark))))
    return contraction


def contract_delta_plus(prop_up, prop_down, pol_matrix):
    return contract_sigma_plus_star(prop_up, prop_down, pol_matrix)


def contract_xi_zero_star(prop_up, prop_strange, pol_matrix):
    return contract_sigma_plus_star(prop_strange, prop_up, pol_matrix)


def contract_omega(prop_strange, pol_matrix):
    # TODO: normalization ?
    return contract_sigma_plus_star(prop_strange, prop_strange, pol_matrix)
