"""Test the O*NET data manager."""

import logging

import pytest

from workrb.data import ONET

OCCUPATIONS_TSV = (
    "O*NET-SOC Code\tTitle\tDescription\n"
    "11-1011.00\tChief Executives\tDetermine and formulate policies.\n"
    "15-1252.00\tSoftware Developers\tResearch, design, and develop software.\n"
    "13-2011.00\tAccountants and Auditors\tExamine and analyze accounting records.\n"
)


@pytest.fixture
def onet_dir(tmp_path):
    """Create a data directory holding a tiny O*NET 30.3 occupation file."""
    version_dir = tmp_path / "30.3"
    version_dir.mkdir()
    (version_dir / "occupations.txt").write_text(OCCUPATIONS_TSV)
    return tmp_path


def test_occupations_vocabulary_is_sorted(onet_dir):
    """The vocabulary holds all occupation titles in sorted order."""
    onet = ONET(version="30.3", data_dir=str(onet_dir), auto_download=False)

    assert onet.get_occupations_vocabulary() == [
        "Accountants and Auditors",
        "Chief Executives",
        "Software Developers",
    ]


def test_occupations_codes(onet_dir):
    """Each occupation title maps to its O*NET-SOC code."""
    onet = ONET(version="30.3", data_dir=str(onet_dir), auto_download=False)

    assert onet.get_occupations_codes() == {
        "Chief Executives": "11-1011.00",
        "Software Developers": "15-1252.00",
        "Accountants and Auditors": "13-2011.00",
    }
    assert onet.get_info()["num_occupations"] == 3


def test_default_version_alias(onet_dir):
    """The 'default' version resolves to the default O*NET release."""
    onet = ONET(version="default", data_dir=str(onet_dir), auto_download=False)

    assert onet.version == ONET.DEFAULT_VERSION


@pytest.fixture
def propagate_workrb_logs():
    """Enable propagation on the 'workrb' logger so caplog sees its records."""
    workrb_logger = logging.getLogger("workrb")
    original = workrb_logger.propagate
    workrb_logger.propagate = True
    yield
    workrb_logger.propagate = original


@pytest.mark.parametrize(("version", "expect_warning"), [("25.0", True), ("25.1", False)])
def test_warns_for_versions_before_soc_2019(
    version, expect_warning, tmp_path, caplog, propagate_workrb_logs
):
    """Releases older than 25.1 warn that they use an older taxonomy."""
    version_dir = tmp_path / version
    version_dir.mkdir()
    (version_dir / "occupations.txt").write_text(OCCUPATIONS_TSV)

    with caplog.at_level(logging.WARNING, logger="workrb.data.onet"):
        ONET(version=version, data_dir=str(tmp_path), auto_download=False)

    warned = any("predates the O*NET-SOC 2019 taxonomy" in r.message for r in caplog.records)
    assert warned == expect_warning


def test_missing_data_without_auto_download_raises(tmp_path):
    """Missing data must raise instead of downloading when auto_download=False."""
    with pytest.raises(RuntimeError, match="auto_download=False"):
        ONET(version="30.3", data_dir=str(tmp_path), auto_download=False)


@pytest.mark.parametrize("version", ["30", "30.3.1", "v30.3", "latest"])
def test_invalid_version_format_raises(version, tmp_path):
    """Versions must look like '<major>.<minor>'."""
    with pytest.raises(AssertionError, match="Invalid version format"):
        ONET(version=version, data_dir=str(tmp_path), auto_download=False)


def test_unknown_version_download_raises(tmp_path):
    """A release that does not exist upstream fails with a clear error."""
    with pytest.raises(RuntimeError, match=r"version 99\.9"):
        ONET(version="99.9", data_dir=str(tmp_path))


def test_download_default_version(tmp_path):
    """The default release downloads and holds the full occupation list."""
    onet = ONET(data_dir=str(tmp_path))

    assert len(onet.get_occupations_vocabulary()) == 1016
    assert onet.get_occupations_codes()["Chief Executives"] == "11-1011.00"
