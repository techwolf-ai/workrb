"""BiEncoder model wrapper for WorkRB."""

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device
from tqdm.auto import tqdm

from workrb.models.base import ModelInterface
from workrb.registry import register_model
from workrb.types import ModelInputType


@register_model()
class BiEncoderModel(ModelInterface):
    """BiEncoder model using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", **kwargs):
        self.base_model_name = model_name
        self.model = SentenceTransformer(model_name)

        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.model.eval()

    @property
    def name(self) -> str:
        return f"BiEncoder-{self.base_model_name.split('/')[-1]}"

    @property
    def description(self) -> str:
        return "BiEncoder model using sentence-transformers for ranking and classification tasks."

    def _compute_rankings(
        self,
        queries: list[str],
        targets: list[str],
        query_input_type: ModelInputType | None = None,
        target_input_type: ModelInputType | None = None,
    ) -> torch.Tensor:
        """Compute ranking scores using cosine similarity."""
        query_embeddings = self.model.encode(queries, convert_to_tensor=True)
        target_embeddings = self.model.encode(targets, convert_to_tensor=True)

        # Normalize for cosine similarity
        query_norm = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
        target_norm = torch.nn.functional.normalize(target_embeddings, p=2, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.mm(query_norm, target_norm.t())
        return similarity_matrix

    def _compute_classification(
        self,
        texts: list[str],
        targets: list[str],
        input_type: ModelInputType,
        target_input_type: ModelInputType | None = None,
    ) -> torch.Tensor:
        """Compute classification scores by ranking texts against target labels.

        Args:
            texts: List of input texts to classify
            targets: List of target class labels (as text)
            input_type: Type of input (e.g., JOB_TITLE)
            target_input_type: Type of target (e.g., SKILL_NAME). If None, uses input_type.

        Returns
        -------
            Tensor of shape (n_texts, n_classes) with similarity scores
        """
        if target_input_type is None:
            target_input_type = input_type

        # Use ranking mechanism to compute similarity between texts and class labels
        return self._compute_rankings(
            queries=texts,
            targets=targets,
            query_input_type=input_type,
            target_input_type=target_input_type,
        )

    @property
    def classification_label_space(self) -> list[str] | None:
        """BiEncoder models do not have classification heads."""
        return None

    @property
    def citation(self) -> str | None:
        """BiEncoder models based on sentence-transformers."""
        return """
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "http://arxiv.org/abs/1908.10084",
}
"""


@register_model()
class JobBERTModel(ModelInterface):
    """BiEncoder model using sentence-transformers."""

    def __init__(self, model_name: str = "TechWolf/JobBERT-v2", **kwargs):
        self.base_model_name = model_name
        self.model = SentenceTransformer(model_name)

        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.model.eval()

        # Map input types to branches
        anchor_branch = "anchor"
        positive_branch = "positive"

        self.default_branch = anchor_branch
        self.input_type_to_branch = {
            ModelInputType.JOB_TITLE: anchor_branch,
            ModelInputType.SKILL_NAME: positive_branch,
            ModelInputType.SKILL_SENTENCE: positive_branch,
        }

    @property
    def name(self) -> str:
        return self.base_model_name.split("/")[-1]

    @property
    def description(self) -> str:
        return (
            "Job-Normalization BiEncoder from Techwolf: https://huggingface.co/TechWolf/JobBERT-v2"
        )

    @staticmethod
    def encode_batch(jobbert_model, texts, branch: str = "anchor"):
        """Encode using the 'anchor' job-branch of the JobBERT model."""
        features = jobbert_model.tokenize(texts)
        features = batch_to_device(features, jobbert_model.device)
        features["text_keys"] = [branch]
        with torch.no_grad():
            out_features = jobbert_model.forward(features)
        return out_features["sentence_embedding"]

    @staticmethod
    def encode(jobbert_model, texts, batch_size: int = 128, branch: str = "anchor"):
        device = jobbert_model.device

        # Sort texts by length and keep track of original indices
        sorted_indices = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        sorted_texts = [texts[i] for i in sorted_indices]

        embeddings = []

        # Encode in batches
        for i in tqdm(range(0, len(sorted_texts), batch_size)):
            batch = sorted_texts[i : i + batch_size]
            embeddings.append(JobBERTModel.encode_batch(jobbert_model, batch, branch))

        # Concatenate embeddings and reorder to original indices
        sorted_embeddings = torch.cat(embeddings, dim=0)
        inverse_positions = [0] * len(sorted_indices)
        for position, original_idx in enumerate(sorted_indices):
            inverse_positions[original_idx] = position
        original_order = torch.tensor(inverse_positions, dtype=torch.long, device=device)
        return sorted_embeddings.index_select(0, original_order)

    @torch.no_grad()
    def _compute_rankings(
        self,
        queries: list[str],
        targets: list[str],
        query_input_type: ModelInputType,
        target_input_type: ModelInputType,
    ) -> torch.Tensor:
        """Compute ranking scores using cosine similarity."""
        query_embeddings = JobBERTModel.encode(
            self.model,
            queries,
            branch=self.input_type_to_branch.get(query_input_type, self.default_branch),
        )
        target_embeddings = JobBERTModel.encode(
            self.model,
            targets,
            branch=self.input_type_to_branch.get(target_input_type, self.default_branch),
        )

        # Normalize for cosine similarity
        query_norm = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
        target_norm = torch.nn.functional.normalize(target_embeddings, p=2, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.mm(query_norm, target_norm.t())
        return similarity_matrix

    @torch.no_grad()
    def _compute_classification(
        self,
        texts: list[str],
        targets: list[str],
        input_type: ModelInputType,
        target_input_type: ModelInputType | None = None,
    ) -> torch.Tensor:
        """Compute classification scores by ranking texts against target labels.

        Args:
            texts: List of input texts to classify
            targets: List of target class labels (as text)
            input_type: Type of input (e.g., JOB_TITLE)
            target_input_type: Type of target (e.g., SKILL_NAME). If None, uses input_type.

        Returns
        -------
            Tensor of shape (n_texts, n_classes) with similarity scores
        """
        if target_input_type is None:
            target_input_type = input_type

        # Use ranking mechanism to compute similarity between texts and class labels
        return self._compute_rankings(
            queries=texts,
            targets=targets,
            query_input_type=input_type,
            target_input_type=target_input_type,
        )

    @property
    def classification_label_space(self) -> list[str] | None:
        """JobBERT models do not have classification heads."""
        return None

    @property
    def citation(self) -> str | None:
        """JobBERT model citations."""
        return """
@misc{jobbert_v3_2025,
      title={Multilingual JobBERT for Cross-Lingual Job Title Matching},
      author={Jens-Joris Decorte and Matthias De Lange and Jeroen Van Hautte},
      year={2025},
      eprint={2507.21609},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.21609},
}
@article{jobbert_v2_2025,
  abstract     = {{Labor market analysis relies on extracting insights from job advertisements, which provide valuable yet unstructured information on job titles and corresponding skill requirements. While state-of-the-art methods for skill extraction achieve strong performance, they depend on large language models (LLMs), which are computationally expensive and slow. In this paper, we propose ConTeXT-match, a novel contrastive learning approach with token-level attention that is well-suited for the extreme multi-label classification task of skill classification. ConTeXT-match significantly improves skill extraction efficiency and performance, achieving state-of-the-art results with a lightweight bi-encoder model. To support robust evaluation, we introduce Skill-XL a new benchmark with exhaustive, sentence-level skill annotations that explicitly address the redundancy in the large label space. Finally, we present JobBERT V2, an improved job title normalization model that leverages extracted skills to produce high-quality job title representations. Experiments demonstrate that our models are efficient, accurate, and scalable, making them ideal for large-scale, real-time labor market analysis.}},
  author       = {{Decorte, Jens-Joris and Van Hautte, Jeroen and Develder, Chris and Demeester, Thomas}},
  issn         = {{2169-3536}},
  journal      = {{IEEE ACCESS}},
  keywords     = {{Taxonomy,Contrastive learning,Training,Annotations,Benchmark testing,Training data,Large language models,Computational efficiency,Accuracy,Terminology,Labor market analysis,text encoders,skill extraction,job title normalization}},
  language     = {{eng}},
  pages        = {{133596--133608}},
  title        = {{Efficient text encoders for labor market analysis}},
  url          = {{http://doi.org/10.1109/ACCESS.2025.3589147}},
  volume       = {{13}},
  year         = {{2025}},
}
@inproceedings{jobbert_v1_2021,
    author       = {{Decorte, Jens-Joris and Van Hautte, Jeroen and Demeester, Thomas and Develder, Chris}},
    booktitle    = {{FEAST, ECML-PKDD 2021 Workshop, Proceedings}},
    language     = {{eng}},
    location     = {{Online}},
    pages        = {{9}},
    title        = {{JobBERT : understanding job titles through skills}},
    url          = {{https://feast-ecmlpkdd.github.io/papers/FEAST2021_paper_6.pdf}},
    year         = {{2021}},
}
"""
