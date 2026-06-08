import numpy as np

from neuromass.utils import LorentzianFrequencyGenerator


def test_lorentzian_sample_shape() -> None:
    generator = LorentzianFrequencyGenerator(seed=0)
    sample = generator.sample(10)
    assert sample.shape == (10,)


def test_lorentzian_truncated_sample_respects_cutoff() -> None:
    generator = LorentzianFrequencyGenerator(x0=1.0, gamma=0.5, seed=0)
    sample = generator.sample(50, truncated=True, cutoff=2.0)
    assert np.all(np.abs(sample - 1.0) <= 2.0)
