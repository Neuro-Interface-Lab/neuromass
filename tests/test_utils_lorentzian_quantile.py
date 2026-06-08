import numpy as np

from neuromass.utils import LorentzianFrequencyGenerator


def test_lorentzian_quantile_sample_is_sorted_and_centered() -> None:
    generator = LorentzianFrequencyGenerator(x0=0.0, gamma=1.0)
    sample = generator.sample(11, method="quantile")

    assert np.all(np.diff(sample) > 0.0)
    assert np.isclose(sample[5], 0.0, atol=1e-12)


def test_lorentzian_quantile_truncated_sample_respects_cutoff() -> None:
    generator = LorentzianFrequencyGenerator(x0=0.0, gamma=1.0)
    sample = generator.sample(32, method="quantile", truncated=True, cutoff=3.0)

    assert np.all(np.abs(sample) <= 3.0)
