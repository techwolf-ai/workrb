"""O*NET data manager, the O*NET counterpart of the ESCO data manager."""

import logging
import re
from pathlib import Path

import appdirs
import pandas as pd
import requests

logger = logging.getLogger(__name__)


class ONET:
    """
    O*NET data manager with flexible data location and caching.

    Downloads the occupation data of a given O*NET database release to a cache
    directory outside the package, mirroring the ESCO data manager interface.
    """

    BASE_URL = "https://www.onetcenter.org/dl_files/database/"
    DEFAULT_VERSION = "30.3"

    # First release on the O*NET-SOC 2019 taxonomy; older releases use O*NET-SOC 2010 or earlier
    SOC_2019_FIRST_VERSION = (25, 1)

    CODE_COL = "O*NET-SOC Code"
    TITLE_COL = "Title"

    def __init__(
        self,
        version: str = DEFAULT_VERSION,
        data_dir: str | None = None,
        auto_download: bool = True,
    ):
        """
        Initialize O*NET data manager.

        Args:
            version: O*NET database release (e.g., "30.3", "25.0")
            data_dir: Custom data directory (defaults to cache dir)
            auto_download: Whether to auto-download missing data
        """
        # Map "default" to latest version
        if version == "default":
            version = self.DEFAULT_VERSION
        assert re.fullmatch(r"\d+\.\d+", version), f"Invalid version format: {version!r}"

        self.version = version
        self.auto_download = auto_download

        major, minor = version.split(".")
        if (int(major), int(minor)) < self.SOC_2019_FIRST_VERSION:
            logger.warning(
                "O*NET v%s predates the O*NET-SOC 2019 taxonomy (introduced in v25.1). Its "
                "occupation codes and titles differ from later releases, so mapping labels "
                "to or from this version will drop occupations without a match.",
                version,
            )

        # Set up data paths - use cache directory by default
        if data_dir:
            self.base_path = Path(data_dir)
        else:
            self.base_path = Path(appdirs.user_cache_dir("workrb")) / "onet"

        self.data_path = self.base_path / version
        self.occupations_path = self.data_path / "occupations.txt"

        self._initialize_data()

    def _initialize_data(self) -> None:
        """Initialize O*NET data, downloading if necessary."""
        if not self.occupations_path.exists():
            if not self.auto_download:
                raise RuntimeError(
                    f"O*NET data not found at {self.occupations_path} and auto_download=False"
                )
            try:
                logger.debug(f"O*NET v{self.version} not found in cache.")
                self._download_onet_data()
            except requests.exceptions.RequestException as e:
                raise RuntimeError(
                    f"Could not download O*NET data for version {self.version}. Check that "
                    f"the version exists, or manually download the occupation data to "
                    f"{self.occupations_path}"
                ) from e

        # Tab-separated file with columns: O*NET-SOC Code, Title, Description
        self.occupations_df = pd.read_csv(self.occupations_path, sep="\t")
        logger.debug(f"Loaded O*NET v{self.version} from {self.occupations_path}")

    def _create_download_url(self) -> str:
        """Create the download URL for the occupation data of the specific version."""
        return f"{self.BASE_URL}db_{self.version.replace('.', '_')}_text/Occupation%20Data.txt"

    def _download_onet_data(self) -> None:
        """Download the O*NET occupation data for the specified version."""
        url = self._create_download_url()
        logger.debug(f"Downloading {url} to {self.occupations_path}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        self.data_path.mkdir(parents=True, exist_ok=True)
        self.occupations_path.write_bytes(response.content)

    def get_occupations_vocabulary(self) -> list[str]:
        """Get sorted list of unique occupation titles."""
        return sorted(self.occupations_df[self.TITLE_COL].unique().tolist())

    def get_occupations_codes(self) -> dict[str, str]:
        """Get mapping from occupation titles to O*NET-SOC codes."""
        return dict(
            zip(
                self.occupations_df[self.TITLE_COL],
                self.occupations_df[self.CODE_COL],
                strict=True,
            )
        )

    def get_info(self) -> dict:
        """Get information about the loaded O*NET data."""
        return {
            "version": self.version,
            "data_path": str(self.data_path),
            "num_occupations": len(self.occupations_df),
        }
