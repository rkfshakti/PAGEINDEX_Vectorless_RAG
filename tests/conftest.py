"""Shared pytest fixtures."""

import json
from pathlib import Path

import pytest

# Minimal PageIndex tree that exercises the retrieval logic.
SAMPLE_TREE = {
    "id": "root",
    "title": "Sample Report",
    "summary": "An overview of the sample report covering chapters 1 to 3.",
    "text": "",
    "children": [
        {
            "id": "ch1",
            "title": "Introduction",
            "summary": "Background and motivation.",
            "text": "This is the introduction section discussing background and motivation for the work.",
            "children": [],
        },
        {
            "id": "ch2",
            "title": "Methodology",
            "summary": "Approach and experimental setup.",
            "text": "We used a mixed-methods approach combining qualitative and quantitative techniques.",
            "children": [
                {
                    "id": "ch2_1",
                    "title": "Data Collection",
                    "summary": "How data was gathered.",
                    "text": "Data was collected via structured interviews and automated sensors.",
                    "children": [],
                }
            ],
        },
        {
            "id": "ch3",
            "title": "Results and Conclusions",
            "summary": "Key findings and takeaways.",
            "text": "The key findings show a 40% improvement in accuracy compared to baseline methods.",
            "children": [],
        },
    ],
}


@pytest.fixture()
def sample_tree() -> dict:
    return json.loads(json.dumps(SAMPLE_TREE))


@pytest.fixture()
def tmp_index(tmp_path: Path, sample_tree: dict) -> Path:
    """Write the sample tree to a temp results directory."""
    results = tmp_path / "results"
    results.mkdir()
    index_file = results / "sample_index.json"
    index_file.write_text(json.dumps(sample_tree), encoding="utf-8")
    return tmp_path
