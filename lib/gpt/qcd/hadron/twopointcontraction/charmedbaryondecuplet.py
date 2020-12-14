import gpt
from gpt.qcd.hadron.twopointcontraction.baryonoctet import baryon_base_contraction
from gpt.qcd.hadron.twopointcontraction.baryondecuplet import contract_sigma_plus_star
from gpt.qcd.hadron.spinmatrices import charge_conjugation


def contract_charmed_sigma_star_zero(prop_down, prop_charm, pol_matrix):
    return contract_sigma_plus_star(prop_down, prop_charm, pol_matrix)


def contract_charmed_xi_zero_star(prop_down, prop_strange, prop_charm, pol_matrix):
    cg5 = charge_conjugation() * gpt.gamma[5]
    return (
        2 * baryon_base_contraction(prop_charm, prop_strange, prop_down, pol_matrix, cg5) +
        2 * baryon_base_contraction(prop_charm, prop_down, prop_strange, pol_matrix, cg5) +
        2 * baryon_base_contraction(prop_strange, prop_charm, prop_down, pol_matrix, cg5) +
        2 * baryon_base_contraction(prop_down, prop_charm, prop_strange, pol_matrix, cg5) -
        baryon_base_contraction(prop_strange, prop_down, prop_charm, pol_matrix, cg5) -
        baryon_base_contraction(prop_down, prop_strange, prop_charm, pol_matrix, cg5)
    )


def contract_double_charmed_xi_plus_star(prop_down, prop_charm, pol_matrix):
    return contract_sigma_plus_star(prop_charm, prop_down, pol_matrix)


def contract_charmed_omega_star(prop_strange, prop_charm, pol_matrix):
    return contract_sigma_plus_star(prop_strange, prop_charm, pol_matrix)


def contract_double_charmed_omega_star(prop_strange, prop_charm, pol_matrix):
    return contract_sigma_plus_star(prop_charm, prop_strange, pol_matrix)


def contract_triple_charmed_omega(prop_charm, pol_matrix):
    # TODO normalization?
    return contract_sigma_plus_star(prop_charm, prop_charm, pol_matrix)
