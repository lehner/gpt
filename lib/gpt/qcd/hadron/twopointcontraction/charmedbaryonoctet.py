import gpt
from gpt.qcd.hadron.twopointcontraction.baryonoctet import \
  baryon_base_contraction, contract_lambda, contract_lambda_naive, contract_lambda_to_sigma_zero
from gpt.qcd.hadron.spinmatrices import charge_conjugation


def contract_charmed_lambda(prop_up, prop_down, prop_charm, pol_matrix):
    return contract_lambda(prop_up, prop_down, prop_charm, pol_matrix)


def contract_charmed_lambda_naive(prop_up, prop_down, prop_charm, pol_matrix):
    return contract_lambda_naive(prop_up, prop_down, prop_charm, pol_matrix)


def contract_charmed_sigma_zero(prop_down, prop_charm, pol_matrix):
    cg5 = charge_conjugation() * gpt.gamma[5]
    return baryon_base_contraction(prop_down, prop_down, prop_charm, pol_matrix, cg5)


def contract_charmed_xi_zero(prop_down, prop_strange, prop_charm, pol_matrix):
    return contract_lambda(prop_down, prop_strange, prop_charm, pol_matrix)


def contract_charmed_xi_prime_zero(prop_down, prop_strange, prop_charm, pol_matrix):
    cg5 = charge_conjugation() * gpt.gamma[5]
    return 0.5 * (
        baryon_base_contraction(prop_down, prop_strange, prop_charm, pol_matrix, cg5) +
        baryon_base_contraction(prop_strange, prop_down, prop_charm, pol_matrix, cg5)
    )


def contract_double_charmed_xi_plus(prop_down, prop_charm, pol_matrix):
    cg5 = charge_conjugation() * gpt.gamma[5]
    return baryon_base_contraction(prop_charm, prop_charm, prop_down, pol_matrix, cg5)


def contract_charmed_omega(prop_strange, prop_charm, pol_matrix):
    cg5 = charge_conjugation() * gpt.gamma[5]
    return baryon_base_contraction(prop_strange, prop_strange, prop_charm, pol_matrix, cg5)


def contract_double_charmed_omega(prop_strange, prop_charm, pol_matrix):
    cg5 = charge_conjugation() * gpt.gamma[5]
    return baryon_base_contraction(prop_charm, prop_charm, prop_strange, pol_matrix, cg5)


def charmed_xi_to_xi_prime(prop_down, prop_strange, prop_charm, pol_matrix):
    return contract_lambda_to_sigma_zero(prop_down, prop_strange, prop_charm, pol_matrix)
