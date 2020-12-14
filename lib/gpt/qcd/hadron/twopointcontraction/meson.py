import gpt


def contract_meson(prop1, prop2, gamma_source, gamma_sink):
    return gpt.trace(
        gamma_source * prop1 * gamma_sink * gpt.gamma[5] * gpt.adj(prop2) * gpt.gamma[5]
    )
