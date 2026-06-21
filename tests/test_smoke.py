import agastya


def test_package_version_present():
    assert isinstance(agastya.__version__, str)
    assert agastya.__version__
