import neuromass as nm


def test_package_import() -> None:
    assert nm.__version__ == "0.1.0"

