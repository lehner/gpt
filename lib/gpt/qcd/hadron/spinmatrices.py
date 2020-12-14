import gpt


def positive_parity_unpolarized():
    return 0.5 * (gpt.gamma["I"] + gpt.gamma["T"])


def positive_parity_xpolarized():
    return positive_parity_unpolarized() * 1j * gpt.gamma["X"] * gpt.gamma[5]


def positive_parity_ypolarized():
    return positive_parity_unpolarized() * 1j * gpt.gamma["Y"] * gpt.gamma[5]


def positive_parity_zpolarized():
    return positive_parity_unpolarized() * 1j * gpt.gamma["Z"] * gpt.gamma[5]


def negative_parity_unpolarized():
    return 0.5 * (gpt.gamma["I"] - gpt.gamma["T"])


def negative_parity_xpolarized():
    return negative_parity_unpolarized() * 1j * gpt.gamma["X"] * gpt.gamma[5]


def negative_parity_ypolarized():
    return negative_parity_unpolarized() * 1j * gpt.gamma["Y"] * gpt.gamma[5]


def negative_parity_zpolarized():
    return negative_parity_unpolarized() * 1j * gpt.gamma["Z"] * gpt.gamma[5]


def charge_conjugation():
    return gpt.gamma["Y"] * gpt.gamma["T"]


def gamma_minus():
    return 0.5 * (gpt.gamma["Y"] + 1j * gpt.gamma["X"])
