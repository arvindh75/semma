<div align="center">

# SEMMA: A Semantic Aware Knowledge Graph Foundation Model #

[![ULTRA arxiv](http://img.shields.io/badge/arxiv-2310.04562-yellow.svg)](https://www.arxiv.org/abs/2505.20422)
![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)

</div>

#

Our implementation is based on the official [ULTRA codebase](https://github.com/DeepGraphLearning/ULTRA) which we extended to integrate the proposed dual-stream architecture, including the LLM-based relation enrichment, G<sub>r</sub><sup>TEXT</sup> construction, and the structural-textual fusion module. 

This repository is based on PyTorch 2.1 and PyTorch-Geometric 2.4

## Setup

You may install the dependencies via either conda or pip. SEMMA is implemented with Python 3.9, PyTorch 2.1, and PyG 2.4 (CUDA 11.8 or later when running on GPUs).

1.  **Create and activate a conda environment (recommended):**
    ```bash
    conda create -n semma python=3.9
    conda activate semma
    ```

2.  **Run the setup script:**
    This script will install all dependencies from `requirements.txt` and download the `fb_mid2name.tsv` file.
    ```bash
    bash setup.sh
    ```
    Make sure `setup.sh` is executable: `chmod +x setup.sh`.
    The `fb_mid2name.tsv` file is used for mapping Freebase MIDs to names.

## How to Use

The primary scripts for running experiments are located in the `script/` directory, with example usage provided below:

### Pretraining

To pretrain a model, you can use `script/pretrain.py` with a corresponding configuration file. For example:
```bash
python script/pretrain.py -c config/transductive/pretrain_3g.yaml --gpus [0]
```
Refer to  `config/` for various pretraining setups.

### Inference

For running inference (evaluation) on pretrained models:

*   **Transductive Setting:**
    Use `script/run.py` or `script/run_many.py` with a transductive config file.
    Example:
    ```bash
    python script/run.py -c config/transductive/inference-fb.yaml --dataset FB15k237_10 --epochs 0 --bpe null --ckpt <path_to_your_checkpoint.pth> --gpus [0]
    ```

*   **Inductive Setting:**
    Similarly, use `script/run.py` or `script/run_many.py` with an inductive config file.
    Example:
    ```bash
    python script/run.py -c config/inductive/inference.yaml --dataset Metafam --version 1 --ckpt <path_to_your_checkpoint.pth> --gpus [0]
    ```

The configuration files for different experiments (e.g., transductive, inductive, specific datasets) are located in the `config/` directory.

## Checkpoints

We provide pretrained model checkpoints in the `ckpts/` directory. Notably, `semma.pth` is the checkpoint for our proposed SEMMA model. You can use these checkpoints directly for inference or fine-tuning.

## Relation Descriptions (OpenRouter Integration)

The `openrouter/` directory contains scripts and resources for generating and utilizing LLM-based relation descriptions.
*   `prompt.py` and `prompt_async.py`: Scripts to query LLMs for relation descriptions.
*   `descriptions/`: Directory containing generated descriptions from different LLMs used in our study.
*   `relations/`: Contains the list of relations of all the datasets invovled in this study.

If you intend to use the scripts in `openrouter/` to query OpenAI (or other services via OpenRouter) for generating new relation descriptions, you will need to set up API access.
**Create a `.env` file** in the root of the project with your API key:
```env
OPENAI_API_KEY="your_openai_api_key_here"
# or other relevant keys if using different providers through OpenRouter
```

## Harder Setting Dataset (takeaway3/)

The `takeaway3/` directory is dedicated to experiments under a "harder" evaluation setting, involving unseen relations in test queries.
*   It contains the dataset used for this harder setting (within the `mtdea/` subdirectory).
*   `gen_split-1.py` and `gen_split-2.py`: Scripts for generating the specific data splits required for this setting.

## Configuration Flags (`flags.yaml`)

The `flags.yaml` file controls various aspects of the SEMMA model and experimental runs. Here's a breakdown of the key flags:

*   `run`: Specifies the model to run. Can be `ultra` (baseline) or `semma` (our proposed model).
    *   If `semma` is chosen, the subsequent flags related to the SEMMA architecture are used.
*   `LLM`: The Large Language Model used for relation enrichment.
    *   Options: `gpt4o`, `qwen3-32b`, `deepseekv3`.
*   `rg2_embedding`: Defines how the textual relation embeddings (G<sub>r</sub><sup>TEXT</sup>) are constructed.
    *   Options:
        *   `combined`: Takes the avg of the embeddings obtained from `no llm`, `llm name` and `llm description`
        *   `combined-sum`: Takes the sum of the embeddings obtained from `no llm`, `llm name` and `llm description`
        *   `no llm`: Excludes LLM-generated features.
        *   `llm name`: Uses only the relation name embedding.
        *   `llm description`: Uses only the relation description embedding.
*   `model_embed`: The embedding model used to encode relation names/descriptions.
    *   Options: `sentbert` (Sentence-BERT), `jinaai` (Jina AI embeddings).
*   `topx`: A float (0 to 1) indicating the top x% of all relation pairs (based on textual similarity) for which to consider adding an edge in G<sub>r</sub><sup>TEXT</sup>. `0` might imply using a threshold.
*   `threshold`: A float (e.g., 0.8). The cosine similarity threshold for constructing G<sub>r</sub><sup>TEXT</sup>
*   `embedding_combiner`: Method used to combine structural and textual embeddings in the fusion module.
    *   Options: `mlp` (Multi-Layer Perceptron), `concat` (concatenation), `attention`.
*   `eval_on_valid`: Boolean (`True`/`False`). If `True`, evaluation is also performed on the validation set during training or a inference run.
*   `use_cos_sim_weights`: Boolean (`True`/`False`). If `True`, the 5th type edges (textual similarity edges) are weighted by their cosine similarity scores.
*   `gpus`: Specifies the GPU ID(s) to use for training/inference (e.g., `0`, `[0, 1]`).
*   `harder_setting`: Boolean (`True`/`False`). If `True`, the model is configured for the "harder" evaluation setting, using data from `takeaway3/` which might involve new relations not seen during pretraining.

Adjust these flags in `flags.yaml` to configure your experiments according to your needs. 

## Todos
- [ ] Add SEMMA Hybrid code

## Citation

```bibtex
@misc{arun2025semmasemanticawareknowledge,
      title={SEMMA: A Semantic Aware Knowledge Graph Foundation Model}, 
      author={Arvindh Arun and Sumit Kumar and Mojtaba Nayyeri and Bo Xiong and Ponnurangam Kumaraguru and Antonio Vergari and Steffen Staab},
      year={2025},
      eprint={2505.20422},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.20422}, 
}
```
