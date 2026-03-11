"""
Load paper experiment results and generate a LaTeX summary table.

Reads results saved by ``run_paper_results.py`` and prints a LaTeX table
comparing all models across task groups for a given metric.

Usage:
    python examples/generate_paper_table.py                    # default metric (map)
    python examples/generate_paper_table.py --metric mrr       # different metric
    python examples/generate_paper_table.py --aggregation-level task  # per-task columns
    python examples/generate_paper_table.py --list             # show discovered results
    python examples/generate_paper_table.py --output table.tex # write to file
"""

import argparse
import logging
import sys
from pathlib import Path

import workrb
from workrb.metrics.reporting import format_results_latex
from workrb.results import BenchmarkResults

# Model folder names produced by run_paper_results.py (model.name values).
MULTILINGUAL_MODELS = [
    "BM25-lower",
    "RandomRanking",
    "JobBERT-v3",
    "BiEncoder-Qwen3-Embedding-0.6B",
]

MONOLINGUAL_MODELS = [
    "ConTeXT-Skill-Extraction-base",
    "skillmatch-mpnet-curriculum-retriever",
    "JobBERT-v2",
]

# Display-friendly names for models and task groups.
SHORT_NAMES: dict[str, str] = {
    # Task groups (also controls column order)
    "rank_job2skill": "J2S",
    "rank_skill2job": "S2J",
    "rank_skill_extraction": "SkExt",
    "rank_skill_normalization": "SkNorm",
    "rank_semantic_similarity": "Sim",
    "rank_job_normalization": "JNorm",
    "rank_candidate_ranking": "CandR",
    # Models
    "BM25-lower": "BM25",
    "RandomRanking": "Random",
    "JobBERT-v3": "JobBERT-v3",
    "BiEncoder-Qwen3-Embedding-0.6B": "Qwen3-0.6B",
    "ConTeXT-Skill-Extraction-base": "ConTeXTMatch",
    "skillmatch-mpnet-curriculum-retriever": "CurriculumMatch",
    "JobBERT-v2": "JobBERT-v2",
}

logger = logging.getLogger(__name__)


def discover_results(results_dir: str) -> list[tuple[str, str, Path]]:
    """Find all results.json files under results_dir.

    Returns list of (group, model_folder_name, path) tuples.
    """
    base = Path(results_dir)
    found = []
    for group_dir in ("multilingual", "monolingual_en"):
        group_path = base / group_dir
        if not group_path.is_dir():
            continue
        for model_dir in sorted(group_path.iterdir()):
            results_file = model_dir / "results.json"
            if results_file.is_file():
                found.append((group_dir, model_dir.name, results_file))
    return found


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX table from paper results.")
    parser.add_argument("--metric", default="map", help="Target metric (default: map)")
    # Default: resolve relative to the repo root (parent of examples/)
    _default_results_dir = str(
        Path(__file__).resolve().parent.parent / ".." / "results" / "run_paper_results"
    )
    parser.add_argument(
        "--results-dir",
        default=_default_results_dir,
        help=f"Root results directory (default: {_default_results_dir})",
    )
    parser.add_argument("--output", default=None, help="Write LaTeX to file instead of stdout")
    parser.add_argument(
        "--aggregation-level",
        default="task_group",
        choices=["task_group", "task"],
        help="Column granularity (default: task_group)",
    )
    parser.add_argument("--list", action="store_true", help="List discovered results and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    # Discover result files
    found = discover_results(args.results_dir)
    if not found:
        print(f"No results found under {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    if args.list:
        print(f"Discovered {len(found)} result(s) in {args.results_dir}:\n")
        for group, model, path in found:
            print(f"  [{group}] {model}  ->  {path}")
        return

    # Load results in the expected order: multilingual first, then monolingual
    ordered_models = MULTILINGUAL_MODELS + MONOLINGUAL_MODELS
    path_by_model: dict[str, Path] = {model: path for _, model, path in found}

    results_list: list[BenchmarkResults] = []
    loaded_names: list[str] = []

    for model_name in ordered_models:
        if model_name not in path_by_model:
            print(f"Warning: results not found for '{model_name}', skipping.", file=sys.stderr)
            continue
        results_list.append(workrb.load_results(str(path_by_model[model_name])))
        loaded_names.append(model_name)

    # Also load any unexpected models found on disk but not in the predefined lists
    for _, model_name, path in found:
        if model_name not in ordered_models:
            print(f"Note: loading extra model '{model_name}'", file=sys.stderr)
            results_list.append(workrb.load_results(str(path)))
            loaded_names.append(model_name)

    if not results_list:
        print("No results loaded.", file=sys.stderr)
        sys.exit(1)

    # Build model_groups indices based on multilingual/monolingual split
    multi_indices = [i for i, n in enumerate(loaded_names) if n in MULTILINGUAL_MODELS]
    mono_indices = [i for i, n in enumerate(loaded_names) if n in MONOLINGUAL_MODELS]
    extra_indices = [
        i
        for i, n in enumerate(loaded_names)
        if n not in MULTILINGUAL_MODELS and n not in MONOLINGUAL_MODELS
    ]
    model_groups = [g for g in [multi_indices, mono_indices, extra_indices] if g]

    latex = format_results_latex(
        results_list=results_list,
        target_metric=args.metric,
        aggregation_level=args.aggregation_level,
        caption=f"WorkRB Benchmark Results ({args.metric.upper()})",
        label=f"tab:workrb_{args.metric}",
        model_groups=model_groups,
        short_names=SHORT_NAMES,
        highlight_best=True,
    )

    if args.output:
        Path(args.output).write_text(latex + "\n")
        print(f"LaTeX table written to {args.output}", file=sys.stderr)
    else:
        print(latex)


if __name__ == "__main__":
    main()
