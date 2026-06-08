"""Minimal example placeholder for the naive Kuramoto model."""

import neuromass as nm


def main() -> None:
    model = nm.models.kuramoto.NaiveKuramotoModel()
    print(model)


if __name__ == "__main__":
    main()

