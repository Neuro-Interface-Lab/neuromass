"""Lorentzian-distributed generators for network model parameters."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class LorentzianFrequencyGenerator:
    """Generate natural frequencies following a Lorentzian distribution."""

    x0: float = 0.0
    gamma: float = 1.0
    symmetric: bool = False
    seed: int | None = None
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.gamma <= 0.0:
            raise ValueError("`gamma` must be strictly positive.")
        self.rng = np.random.default_rng(self.seed)

    def sample(
        self,
        n_samples: int,
        truncated: bool = False,
        cutoff: float | None = None,
        method: str = "random",
        seed: int | None = None,
    ) -> np.ndarray:
        """Generate Lorentzian-distributed samples."""

        if n_samples <= 0:
            raise ValueError("`n_samples` must be strictly positive.")

        if seed is not None:
            self.reset_rng(seed)

        if method not in {"random", "quantile"}:
            raise ValueError("`method` must be either 'random' or 'quantile'.")

        if method == "quantile":
            if truncated:
                if cutoff is None or cutoff <= 0.0:
                    raise ValueError("A strictly positive `cutoff` is required for truncation.")
                return self._sample_quantile_truncated(n_samples, cutoff)
            return self._sample_quantile(n_samples)

        if truncated:
            if cutoff is None or cutoff <= 0.0:
                raise ValueError("A strictly positive `cutoff` is required for truncation.")
            return self._sample_truncated(n_samples, cutoff)

        if self.symmetric:
            return self._sample_symmetric(n_samples)
        return self._sample_standard(n_samples)

    def reset_rng(self, seed: int | None = None) -> None:
        """Reset the random number generator."""

        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _sample_standard(self, n_samples: int) -> np.ndarray:
        return self.x0 + self.gamma * self.rng.standard_cauchy(n_samples)

    def _sample_symmetric(self, n_samples: int) -> np.ndarray:
        half = n_samples // 2
        base = self.gamma * self.rng.standard_cauchy(half)
        symmetric_samples = np.concatenate([base, -base])

        if n_samples % 2 == 1:
            symmetric_samples = np.append(symmetric_samples, 0.0)

        return self.x0 + symmetric_samples

    def _sample_truncated(self, n_samples: int, cutoff: float) -> np.ndarray:
        samples: list[float] = []

        while len(samples) < n_samples:
            if self.symmetric:
                batch = self._sample_symmetric(n_samples)
            else:
                batch = self._sample_standard(n_samples)

            valid = batch[np.abs(batch - self.x0) <= cutoff]
            samples.extend(valid.tolist())

        return np.asarray(samples[:n_samples], dtype=np.float64)

    def _sample_quantile(self, n_samples: int) -> np.ndarray:
        probabilities = (np.arange(1, n_samples + 1, dtype=np.float64) - 0.5) / n_samples
        samples = self.x0 + self.gamma * np.tan(np.pi * (probabilities - 0.5))
        return np.asarray(samples, dtype=np.float64)

    def _sample_quantile_truncated(self, n_samples: int, cutoff: float) -> np.ndarray:
        lower_probability = 0.5 + np.arctan(-cutoff / self.gamma) / np.pi
        upper_probability = 0.5 + np.arctan(cutoff / self.gamma) / np.pi
        probabilities = np.linspace(
            lower_probability,
            upper_probability,
            n_samples,
            endpoint=False,
            dtype=np.float64,
        )
        probabilities += (upper_probability - lower_probability) / (2.0 * n_samples)
        centered_samples = self.gamma * np.tan(np.pi * (probabilities - 0.5))
        return np.asarray(self.x0 + centered_samples, dtype=np.float64)

