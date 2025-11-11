"""
Enhanced ESCO data manager for the hybrid approach.
"""

import logging
import urllib.parse
import zipfile
from pathlib import Path

import appdirs
import pandas as pd
import requests
from tqdm import tqdm

from workrb.types import Language

logger = logging.getLogger(__name__)


class ESCO:
    """
    ESCO data manager with flexible data location and caching.

    Integrates with the hybrid data approach - downloads to cache directory
    outside the package while maintaining the same interface.
    """

    BASE_URL = "https://ec.europa.eu/esco/download/"
    SUPPORTED_ESCO_LANGUAGES: tuple[Language, ...] = (
        Language.BG,  # Bulgarian
        Language.ES,  # Spanish
        Language.CS,  # Czech
        Language.DA,  # Danish
        Language.DE,  # German
        Language.ET,  # Estonian
        Language.EL,  # Greek
        Language.EN,  # English
        Language.FR,  # French
        Language.GA,  # Irish
        Language.HR,  # Croatian
        Language.IT,  # Italian
        Language.LV,  # Latvian
        Language.LT,  # Lithuanian
        Language.HU,  # Hungarian
        Language.MT,  # Maltese
        Language.NL,  # Dutch
        Language.PL,  # Polish
        Language.PT,  # Portuguese
        Language.RO,  # Romanian
        Language.SK,  # Slovak
        Language.SL,  # Slovenian
        Language.FI,  # Finnish
        Language.SV,  # Swedish
        Language.IS,  # Icelandic
        Language.NO,  # Norwegian
        Language.AR,  # Arabic
        Language.UK,  # Ukrainian
    )

    def __init__(
        self,
        version: str = "1.2.0",
        language: Language = Language.EN,
        data_dir: str | None = None,
        auto_download: bool = True,
    ):
        """
        Initialize ESCO data manager.

        Args:
            version: ESCO version (e.g., "1.2.0", "1.1.0")
            language: Language code (e.g., "en", "fr", "de")
            data_dir: Custom data directory (defaults to cache dir)
            auto_download: Whether to auto-download missing data
        """
        # Map "default" to latest version
        if version == "default":
            version = "1.2.0"

        self.version = version
        self.auto_download = auto_download

        self.language = language
        assert self.language in self.SUPPORTED_ESCO_LANGUAGES, (
            f"Language {self.language.value} not supported by ESCO. "
            f"Supported languages: {[lang.value for lang in self.SUPPORTED_ESCO_LANGUAGES]}"
        )

        # Set up data paths - use cache directory by default
        if data_dir:
            self.base_path = Path(data_dir)
        else:
            cache_dir = appdirs.user_cache_dir("wteb")
            self.base_path = Path(cache_dir) / "esco"

        self.data_path = self.base_path / version / self.language

        # Initialize data
        self._initialize_data()

    def _initialize_data(self):
        """Initialize ESCO data, downloading if necessary."""
        # Try to load from data path
        if self._load_data_from_path(self.data_path):
            return

        # Try auto-download if enabled
        if not self.auto_download:
            raise RuntimeError(f"ESCO data not found at {self.data_path} and auto_download=False")

        try:
            logger.debug(f"ESCO v{self.version} ({self.language.value}) not found in cache.")
            logger.debug(f"Downloading to {self.data_path}...")
            self._download_esco_data()

            if self._load_data_from_path(self.data_path):
                logger.debug("Successfully downloaded and loaded ESCO data.")
                return
        except Exception as e:
            logger.debug(f"Failed to download ESCO data: {e}")
            raise RuntimeError(
                f"Could not load or download ESCO data for version {self.version}, "
                f"language {self.language.value}. Try setting auto_download=True or "
                f"manually downloading data to {self.data_path}"
            ) from e

    def _load_data_from_path(self, path: Path) -> bool:
        """Try to load ESCO data from the given path. Returns True if successful."""
        skills_path = path / f"skills_{self.language.value}.csv"
        occupations_path = path / f"occupations_{self.language.value}.csv"
        relations_path = path / f"occupationSkillRelations_{self.language.value}.csv"

        if not all(p.exists() for p in [skills_path, occupations_path, relations_path]):
            logger.debug(f"ESCO data not found at {path}")
            return False

        # Load the data
        self.skills_df = self._load_skills_df(skills_path)
        self.uri_to_skill = {
            row["conceptUri"]: row["preferredLabel"] for _, row in self.skills_df.iterrows()
        }

        self.occupations_df = pd.read_csv(occupations_path)
        self.uri_to_occupation = {
            row["conceptUri"]: row["preferredLabel"] for _, row in self.occupations_df.iterrows()
        }

        self.relations_df = pd.read_csv(relations_path)

        logger.debug(f"Loaded ESCO v{self.version} ({self.language.value}) from {path}")
        return True

    def _load_skills_df(self, skills_path: Path) -> pd.DataFrame:
        """
        Load the skills DataFrame from the given path.

        Postprocess alternative labels that are multi-line with whitespaces and empty lines.
        """
        skills_df = pd.read_csv(skills_path)

        # Remove NaNs
        skills_df = skills_df[skills_df["preferredLabel"].notna()]

        def parse_alt_labels_to_list(str_list: str) -> list[str]:
            """Reformat alternative labels that are multi-line with whitespaces and empty lines."""
            if pd.isna(str_list):
                return []
            assert isinstance(str_list, str), (
                f"Alt label should be a str but is {type(str_list)}: '{str_list}'"
            )
            skill_parts = []
            for skill_part_raw in str_list.split("\n"):
                skill_part = skill_part_raw.strip()
                if len(skill_part) > 1 and skill_part not in skill_parts:
                    skill_parts.append(skill_part)
            return skill_parts

        skills_df["altLabels"] = skills_df["altLabels"].apply(parse_alt_labels_to_list)

        # Remove self from alternative labels
        skills_df["altLabels"] = skills_df[["preferredLabel", "altLabels"]].apply(
            lambda x: [alt for alt in x["altLabels"] if alt != x["preferredLabel"]],
            axis=1,
        )

        return skills_df

    def _create_download_url(self) -> str:
        """Create the download URL for the specific version and language."""
        filename = (
            f"ESCO dataset - v{self.version} - classification - {self.language.value} - csv.zip"
        )
        encoded_filename = urllib.parse.quote(filename)
        return f"{self.BASE_URL}{encoded_filename}"

    def _download_file(self, url: str, output_path: Path) -> bool:
        """Download a file from URL to output_path. Returns True if successful."""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            output_path.parent.mkdir(parents=True, exist_ok=True)

            total_size = int(response.headers.get("content-length", 0))

            with (
                open(output_path, "wb") as f,
                tqdm(
                    desc=f"Downloading ESCO v{self.version} ({self.language.value})",
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar,
            ):
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            return True

        except requests.exceptions.RequestException as e:
            logger.warning(f"Download failed: {e}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected error during download: {e}")
            return False

    def _unzip_file(self, zip_path: Path) -> bool:
        """Extract zip file to the appropriate dataset directory."""
        try:
            # Create target directory
            self.data_path.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.data_path)

            return True

        except zipfile.BadZipFile as e:
            logger.warning(f"Invalid zip file: {e}")
            return False
        except Exception as e:
            logger.warning(f"Extraction failed: {e}")
            return False

    def _download_esco_data(self) -> None:
        """Download ESCO data for the specified version and language."""
        # Create download cache directory
        download_cache = self.base_path / "downloads"
        download_cache.mkdir(parents=True, exist_ok=True)

        # Create filename and path
        filename = f"ESCO_dataset_v{self.version}_classification_{self.language.value}_csv.zip"
        zip_path = download_cache / filename

        # Download if not already cached
        if not zip_path.exists():
            url = self._create_download_url()

            if not self._download_file(url, zip_path):
                raise RuntimeError(f"Failed to download ESCO data from {url}")

        # Extract the zip file
        if not self._unzip_file(zip_path):
            raise RuntimeError(f"Failed to extract ESCO data from {zip_path}")

    # Keep all the existing data access methods from your original ESCO class
    def get_skills_vocabulary(self) -> list[str]:
        """Get list of all skill preferred labels in deterministic sorted order."""
        return sorted(self.skills_df["preferredLabel"].unique().tolist())

    def get_skills_uris(self) -> dict[str, str]:
        """Get dictionary of all skill URIs."""
        return dict(
            zip(
                self.skills_df["preferredLabel"].tolist(),
                self.skills_df["conceptUri"].tolist(),
                strict=True,
            )
        )

    def get_occupations_vocabulary(self) -> list[str]:
        """Get list of all occupation preferred labels in deterministic sorted order."""
        return sorted(self.occupations_df["preferredLabel"].unique().tolist())

    def get_occupations_uris(self) -> dict[str, str]:
        """Get dictionary of all occupation URIs."""
        return dict(
            zip(
                self.occupations_df["preferredLabel"].tolist(),
                self.occupations_df["conceptUri"].tolist(),
                strict=True,
            )
        )

    def get_occupation_skill_relations(
        self, relation_type: str | None = None
    ) -> list[tuple[str, str, str]]:
        """Get occupation-skill relationships."""
        relations_df = self.relations_df.copy()
        if relation_type:
            relations_df = relations_df[relations_df["relationType"] == relation_type]

        return list(
            relations_df[["occupationUri", "skillUri", "relationType"]].itertuples(
                index=False, name=None
            )
        )

    def get_skills_for_occupation(
        self, occupation_uri: str, relation_type: str | None = None
    ) -> list[str]:
        """Get skills associated with a specific occupation."""
        occupation_relations = self.relations_df[
            self.relations_df["occupationUri"] == occupation_uri
        ]

        if relation_type:
            occupation_relations = occupation_relations[
                occupation_relations["relationType"] == relation_type
            ]

        return occupation_relations["skillUri"].tolist()

    def uri_to_preferred_label(self, uri: str, entity_type: str = "skill") -> str | None:
        """Convert URI to preferred label."""
        if entity_type == "skill":
            return self.uri_to_skill.get(uri)
        if entity_type == "occupation":
            return self.uri_to_occupation.get(uri)

    def create_job2skills_dataset(self, relation_type: str = "essential") -> list[dict]:
        """Create job-to-skills dataset for benchmarking."""
        filtered_relations = self.relations_df[self.relations_df["relationType"] == relation_type]
        dataset = []

        for _, occupation in self.occupations_df.iterrows():
            occ_uri = occupation["conceptUri"]
            occ_label = occupation["preferredLabel"]

            # Get skills for this occupation
            occ_skills = filtered_relations[filtered_relations["occupationUri"] == occ_uri][
                "skillUri"
            ].tolist()

            if occ_skills:  # Only include occupations that have skills
                # Convert skill URIs to labels
                skill_labels = []
                for skill_uri in occ_skills:
                    skill_label = self.uri_to_skill.get(skill_uri)
                    if skill_label:
                        skill_labels.append(skill_label)

                if skill_labels:
                    dataset.append({"occupation": occ_label, "skills": skill_labels})

        return dataset

    def get_skills_with_alternatives(self) -> dict[str, list[str]]:
        """Get skills with their alternative labels for skill normalization tasks."""
        # Only keep skills with alternative labels
        skills_alt_df: pd.DataFrame = self.skills_df.copy()
        # Each at least 1 alternative label
        skills_alt_df = pd.DataFrame(skills_alt_df[skills_alt_df["altLabels"].apply(len) > 0])

        return dict(
            zip(
                skills_alt_df["preferredLabel"].tolist(),
                skills_alt_df["altLabels"].tolist(),
                strict=True,
            )
        )

    def get_info(self) -> dict:
        """Get information about loaded ESCO data."""
        return {
            "version": self.version,
            "language": self.language,
            "data_path": str(self.data_path),
            "num_skills": len(self.skills_df),
            "num_occupations": len(self.occupations_df),
            "num_relations": len(self.relations_df),
        }
