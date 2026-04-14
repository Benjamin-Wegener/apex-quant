# APEX Calibration Dataset (Reconstruction)

A reconstructed calibration dataset matching the composition described in the [APEX Technical Report](https://github.com/mudler/apex-quant/blob/main/paper/APEX_Technical_Report.md), Appendix B.

> **Note:** The original APEX calibration dataset is not publicly available. This repository provides a faithful reconstruction using publicly available sources and synthetic data.

## Overview

APEX I-variants use a custom calibration dataset with the following composition. The dataset contains **no Wikipedia text** to avoid biasing quantization toward encyclopedic prose.

## Dataset Composition

| Domain | Content Type | Approx. Share |
|---|---|---|
| **Multi-turn Chat** | Conversational exchanges in English and Spanish, covering general knowledge, advice, and Q&A | ~30% |
| **Code** | Python, JavaScript, and systems programming examples with inline comments and documentation | ~25% |
| **Reasoning** | Step-by-step mathematical and logical problem solving, chain-of-thought traces | ~25% |
| **Tool-calling** | Structured interactions with JSON function calls, API specifications, and tool use patterns | ~20% |

### Actual Distribution (v1.0)

| Domain | Tokens | Share | Target | Samples |
|---|---|---|---|---|
| Chat | 14,628 | 30.1% | 30% | 77 |
| Code | 11,899 | 24.5% | 25% | 18 |
| Reasoning | 12,294 | 25.3% | 25% | 60 |
| Tool-calling | 9,811 | 20.2% | 20% | 39 |
| **Total** | **48,632** | — | **50,000** | **194** |

## Quick Start

### Prerequisites

```bash
pip install datasets tokenizers
```

### Build the Dataset

```bash
python build_calibration_data.py
```

This will:
1. Download data from public HuggingFace datasets (OpenAssistant, CodeSearchNet)
2. Generate synthetic data for domains without public sources (Reasoning, Tool-calling)
3. Deduplicate all entries
4. Truncate to ~50,000 tokens following the target distribution
5. Save as `calibration_data.jsonl` and `calibration_stats.json`

### Use with llama.cpp

```bash
# Compute importance matrix using llama.cpp
./llama-imatrix \
  --model your-model-f16.gguf \
  --file calibration_data.jsonl \
  --output imatrix.dat \
  --chunk-size 512
```

## Data Sources

| Domain | Source | License |
|---|---|---|
| Multi-turn Chat | [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) | Apache 2.0 |
| Code | [CodeSearchNet](https://huggingface.co/datasets/code_search_net) | Apache 2.0 |
| Reasoning | Synthetic (math/logic problems) | MIT |
| Tool-calling | Synthetic (function calling patterns) | MIT |

### Fallback Behavior

If HuggingFace datasets are unavailable, the script generates synthetic data for all domains. The synthetic data covers:
- **Chat:** General knowledge Q&A, programming advice, multilingual (EN/ES)
- **Code:** Python, JavaScript, C, and Rust examples with documentation
- **Reasoning:** Step-by-step math problems, proofs, combinatorics, probability
- **Tool-calling:** Weather, flights, currency, reminders, GitHub, Yelp, email, calculator

## Output Format

### calibration_data.jsonl

Each line is a JSON object:

```json
{
  "text": "User: ...\nAssistant: ...",
  "source": "oasst1|code_search_net|synthetic",
  "domain": "chat|code|reasoning|tool_call",
  "lang": "en|es|python|javascript|c|rust",
  "token_count": 123
}
```

### calibration_stats.json

```json
{
  "chat": {
    "before_dedup": 200,
    "after_dedup": 200,
    "tokens": 14628,
    "samples": 77
  },
  "total": {
    "tokens": 48632,
    "samples": 194
  }
}
```

## Project Structure

```
├── README.md                    # This file
├── LICENSE                      # MIT License
├── build_calibration_data.py    # Dataset builder script
├── calibration_data.jsonl       # Generated calibration dataset
├── calibration_stats.json       # Dataset statistics
├── requirements.txt             # Python dependencies
├── .python-version              # Python version
└── .gitignore                   # Git ignore rules
```

## Exclusions

As per the original APEX specification:
- ❌ No Wikipedia text (avoids bias toward encyclopedic prose)
- ❌ No highly specialized academic text
- ❌ No non-English/Spanish text (except code comments)

## License

This project is licensed under the [MIT License](LICENSE).

The generated calibration dataset (`calibration_data.jsonl`) inherits the licenses of its source data:
- OpenAssistant/oasst1 samples: Apache 2.0
- CodeSearchNet samples: Apache 2.0
- Synthetic samples: MIT

## Acknowledgments

- Based on the [APEX Technical Report](https://github.com/mudler/apex-quant/blob/main/paper/APEX_Technical_Report.md)
- Uses [llama.cpp](https://github.com/ggml-org/llama.cpp) for importance matrix computation
- Data sourced from [HuggingFace Datasets](https://huggingface.co/datasets)

## Contributing

Feel free to improve the synthetic data quality or add additional public data sources. Pull requests welcome!
