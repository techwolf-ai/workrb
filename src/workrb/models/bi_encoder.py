"""BiEncoder model wrapper for WorkRB, along with some instances of the BiEncoder model."""

from typing import Any

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device
from torch.nn.utils.rnn import pad_sequence
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
        """Return the model name."""
        return f"BiEncoder-{self.base_model_name.split('/')[-1]}"

    @property
    def description(self) -> str:
        """Return the model description."""
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
        """Return the model name."""
        return self.base_model_name.split("/")[-1]

    @property
    def description(self) -> str:
        """Return the model description."""
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


@register_model()
class ConTeXTMatchModel(ModelInterface):
    """Token-level attention bi-encoder for skill extraction using ConTeXT-Match.

    Unlike standard bi-encoders that produce a single embedding per text, ConTeXT-Match
    retains per-token embeddings for query texts (shape: (B, L, D)) and computes
    attention-weighted similarity against mean-pooled target embeddings (shape: (B, D)).
    This allows the model to focus on the most relevant tokens in a query when matching
    against each target.

    Memory considerations:
        The scoring step produces intermediate tensors of shape (num_queries, num_targets, seq_len),
        which can cause OOM with large query sets. To mitigate this, scoring is batched over queries
        using ``scoring_batch_size`` — targets are encoded once and reused across all query chunks.

    Batch sizes:
        This model has two distinct batch size controls:

        - ``scoring_batch_size`` (constructor param, default=32): Controls how many queries are
          scored against all targets at once in ``_compute_rankings``. Lower values reduce peak
          GPU memory during the attention-weighted similarity computation. Must be >= 1.

        - ``encode_batch_size`` (param in ``encode()``, default=128): Controls how many texts are
          tokenized and encoded through the transformer at once. This affects memory during the
          forward pass of the underlying SentenceTransformer model. Typically less of a bottleneck
          than the scoring step.
    """

    _NEAR_ZERO_THRESHOLD = 1e-9

    def __init__(
        self,
        model_name: str = "TechWolf/ConTeXT-Skill-Extraction-base",
        temperature: float = 1.0,
        scoring_batch_size: int = 32,
        **kwargs,
    ):
        """Initialize the ConTeXT-Match model.

        Args:
            model_name: HuggingFace model identifier for the ConTeXT-Match model.
            temperature: Temperature for the attention softmax in scoring. Higher values
                produce softer attention weights across tokens; lower values concentrate
                attention on the most relevant tokens.
            scoring_batch_size: Number of queries to score against all targets at once.
                Controls peak GPU memory during ``_compute_rankings``. Lower values use
                less memory but may be slower. Must be >= 1.
            **kwargs: Additional keyword arguments (unused, for compatibility).
        """
        self.base_model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.temperature = temperature
        self.scoring_batch_size = max(1, scoring_batch_size)
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.model.eval()

    @staticmethod
    def _context_match_score(
        token_embeddings: torch.Tensor, target_embeddings: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        """Compute attention-weighted similarity between token and target embeddings.

        For each (query, target) pair, computes dot-product attention weights over the
        query's tokens, then returns the weighted sum of per-token cosine similarities.

        Args:
            token_embeddings: Query token embeddings of shape (B1, L, D), where B1 is the
                number of queries, L is the (padded) sequence length, and D is the embedding
                dimension. Padding tokens should have near-zero embeddings.
            target_embeddings: Mean-pooled target embeddings of shape (B2, D).
            temperature: Softmax temperature for attention weights. Controls how sharply
                the model attends to specific tokens.

        Returns
        -------
            Similarity matrix of shape (B1, B2), where entry (i, j) is the attention-weighted
            cosine similarity between query i and target j.
        """
        # token_embeddings: (B1, L, D), target_embeddings: (B2, D)
        dot_scores = (token_embeddings @ target_embeddings.T).transpose(1, 2)  # (B1, B2, L)
        dot_scores[dot_scores.abs() < ConTeXTMatchModel._NEAR_ZERO_THRESHOLD] = float("-inf")
        weights = torch.softmax(dot_scores / temperature, dim=2)  # (B1, B2, L)

        norm_tokens = torch.nn.functional.normalize(token_embeddings, p=2, dim=2)  # (B1, L, D)
        norm_targets = torch.nn.functional.normalize(target_embeddings, p=2, dim=1)  # (B2, D)
        sim_scores = (norm_tokens @ norm_targets.T).transpose(
            1, 2
        )  # (B1,L,B2) -> transposed to(B1, B2, L)

        return (weights * sim_scores).sum(dim=2)  # (B1, B2)

    @property
    def name(self) -> str:
        """Return the model name."""
        return self.base_model_name.split("/")[-1]

    @property
    def description(self) -> str:
        """Return the model description."""
        return "ConTeXT-Skill-Extraction-base from Techwolf: https://huggingface.co/TechWolf/ConTeXT-Skill-Extraction-base"

    @staticmethod
    def encode_batch(contextmatch_model, texts, mean: bool = False) -> torch.Tensor:
        """Encode a single batch of texts through the ConTeXT-Match model.

        Args:
            contextmatch_model: The underlying SentenceTransformer model instance.
            texts: List of texts to encode in this batch.
            mean: If False (default), returns per-token embeddings for use as queries.
                If True, returns mean-pooled sentence embeddings for use as targets.

        Returns
        -------
            Tensor of token embeddings (variable-length list) or sentence embeddings.
        """
        args: dict[str, Any] = {
            "normalize_embeddings": False,
            "convert_to_tensor": True,
        }
        if not mean:
            args["output_value"] = "token_embeddings"
        return contextmatch_model.encode(texts, **args)

    @staticmethod
    def encode(
        contextmatch_model, texts, encode_batch_size: int = 128, mean: bool = False
    ) -> torch.Tensor:
        """Encode texts through the ConTeXT-Match model in batches.

        Processes texts in chunks of ``encode_batch_size`` through the transformer,
        then pads variable-length token embeddings to a uniform sequence length.

        Args:
            contextmatch_model: The underlying SentenceTransformer model instance.
            texts: List of texts to encode.
            encode_batch_size: Number of texts to pass through the transformer at once.
                Controls GPU memory usage during the encoding forward pass.
            mean: If False (default), returns padded per-token embeddings (B, L, D).
                If True, returns mean-pooled sentence embeddings (B, D).

        Returns
        -------
            Tensor of shape (B, L, D) for token embeddings or (B, D) for mean-pooled.
        """
        # For token embeddings, process in batches and handle variable lengths
        all_token_embeddings = []
        for i in tqdm(range(0, len(texts), encode_batch_size)):
            batch = texts[i : i + encode_batch_size]
            batch_token_embs = ConTeXTMatchModel.encode_batch(contextmatch_model, batch, mean=mean)
            all_token_embeddings.extend(batch_token_embs)

        token_embeddings = pad_sequence(all_token_embeddings, batch_first=True)
        return token_embeddings

    @torch.no_grad()
    def _compute_rankings(
        self,
        queries: list[str],
        targets: list[str],
        query_input_type: ModelInputType,
        target_input_type: ModelInputType,
    ) -> torch.Tensor:
        """Compute ranking scores using attention-weighted similarity.

        Two-phase approach:
            1. **Encode**: All queries and targets are encoded through the transformer.
               Queries produce per-token embeddings (B1, L, D); targets are mean-pooled (B2, D).
               This phase is batched by ``encode_batch_size`` in ``encode()``.
            2. **Score**: The similarity between queries and targets is computed via
               ``_context_match_score``. Because this creates intermediate tensors of shape
               (chunk_size, num_targets, seq_len), queries are processed in chunks of
               ``scoring_batch_size`` to keep GPU memory bounded. Targets are encoded once
               and reused across all query chunks.

        Returns
        -------
            Similarity matrix of shape (num_queries, num_targets).
        """
        query_token_embeddings = ConTeXTMatchModel.encode(self.model, queries)
        target_token_embeddings_mean = ConTeXTMatchModel.encode(self.model, targets, mean=True)

        # Batch over queries to avoid OOM from the (chunk_size, num_targets, seq_len)
        # intermediate tensor in _context_match_score
        chunks = []
        for i in range(0, len(queries), self.scoring_batch_size):
            query_chunk = query_token_embeddings[i : i + self.scoring_batch_size]
            chunk_scores = self._context_match_score(
                query_chunk, target_token_embeddings_mean, temperature=self.temperature
            )
            chunks.append(chunk_scores)
        return torch.cat(chunks, dim=0)

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
        """ConTeXT-Match models do not have classification heads."""
        return None

    @property
    def citation(self) -> str | None:
        """ConTeXT-Match model citations."""
        return """
@ARTICLE{contextmatch_2025,
  author={Decorte, Jens-Joris and van Hautte, Jeroen and Develder, Chris and Demeester, Thomas},
  journal={IEEE Access},
  title={Efficient Text Encoders for Labor Market Analysis},
  year={2025},
  volume={13},
  number={},
  pages={133596-133608},
  keywords={Taxonomy;Contrastive learning;Training;Annotations;Benchmark testing;Training data;Large language models;Computational efficiency;Accuracy;Terminology;Labor market analysis;text encoders;skill extraction;job title normalization},
  doi={10.1109/ACCESS.2025.3589147}}
"""
