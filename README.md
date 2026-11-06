# BERT-Log

Code for **BERT-Log: Anomaly Detection for System Logs Based on Pre-trained Language Model**, Song Chen and Hai Liao, *Applied Artificial Intelligence* 36(1), 2145642 (2022). [Paper](https://doi.org/10.1080/08839514.2022.2145642).

BERT-Log fine-tunes `bert-base-uncased` for binary log-sequence classification on HDFS and BGL.

## Code tree

```text
bert-log/
├── Models/
│   ├── bertlog/
│   │   ├── run_HDFS.py       # HDFS training and evaluation
│   │   ├── run_BGL.py        # BGL training and evaluation
│   │   └── BGL_Load_node.py  # BGL time-window/node grouping
│   ├── loglizer-master/      # Traditional baselines and data utilities
│   ├── electra/              # Alternative language-model experiments
│   ├── ernie/
│   ├── roberta/
│   ├── spanBert/
│   ├── T5/
│   └── UniLM/
├── utils/logutil.py          # Training logger
├── requirements.txt         # Suggested BERT-Log environment
├── citation.bib
└── README.md
```

## Setup

The original training scripts require an NVIDIA GPU and CUDA-enabled PyTorch. The following is a suggested legacy-compatible environment; it has not been training-tested here.

```bash
git clone https://github.com/iliaohai/bert-log.git
cd bert-log
conda create -n bert-log python=3.9 -y
conda activate bert-log
conda install pytorch==1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
python -m pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

The last command must print `True`. The first training run downloads `bert-base-uncased` through Hugging Face; ensure network access or populate the Hugging Face cache beforehand.

## Data preparation

Download the full [HDFS_v1 and BGL datasets from Loghub](https://github.com/logpai/loghub). Raw logs must be parsed and grouped into sequence CSVs before training. The processed CSVs used by the original experiments are **not included** in this repository.

Place your prepared files as follows:

```text
bert-log/
├── Dataset/
│   ├── HDFS/data_instances2.csv
│   └── BGL/BGL_sequence.csv
└── OutPut/BERTLog/             # Created before training
```

| Dataset | Required CSV columns | Sequence construction |
|---|---|---|
| HDFS | `EventSequence`, `Label` | Group parsed events by block ID; attach block anomaly labels |
| BGL | `sequence`, `label` | Group parsed events by time window and node; any anomalous event makes the sequence anomalous |

Sequence cells must contain stringified lists such as `['E1', 'E2']`; labels must be integers (`0` normal, `1` anomalous). Preserve the original event representation when reproducing an existing experiment; changing event IDs to template text changes the input.

For BGL, `Models/bertlog/BGL_Load_node.py` accepts a parsed CSV with `Time`, `Label`, `EventId`, and `Node`. `Time` uses `%Y-%m-%d-%H.%M.%S.%f`; raw normal labels are `-`. Edit its `para` paths, then run:

```bash
python Models/bertlog/BGL_Load_node.py
```

It defaults to a **0.5-hour window** and **0.2-hour step**, writes `BGL_sequence.csv`, and excludes node-window groups containing fewer than two events. Move the output to `Dataset/BGL/`. The exact HDFS preprocessing workflow producing `data_instances2.csv` is not supplied.

## Train and evaluate

Run these commands in Bash from the repository root so that relative data paths and `utils` imports resolve:

```bash
mkdir -p OutPut/BERTLog
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
cd Models/bertlog
python run_HDFS.py
# Or:
python run_BGL.py
```

| Setting | HDFS | BGL |
|---|---|---|
| Training / evaluation split | 1% / 99% | 1% / 99% |
| Split seed | 32 | Unspecified |
| Batch size | 16 | 24 |
| Epochs | 3 | 2 |
| AdamW learning rate | 2e-5 | 2e-5 |

Change `test_size`, `batch_size`, or `epochs` in the corresponding script. Training drops incomplete final batches; provide enough data for at least one full training batch. HDFS inputs must fit BERT's 512-token limit because token truncation is not enabled; BGL uses a 510-character cutoff before tokenization.

Training and per-epoch evaluation metrics are written to `OutPut/BERTLog/*.log`. F1 uses binary averaging; precision and recall use macro averaging. Model checkpoints are not saved by the original scripts.

**Validation:** entry points, paths, and defaults were checked against the source. Full training and paper metrics were not reproduced because the processed data, cached weights, and CUDA runtime were unavailable.

## Citation

```bibtex
@article{chen2022bertlog,
  author  = {Chen, Song and Liao, Hai},
  title   = {{BERT-Log}: Anomaly Detection for System Logs Based on Pre-trained Language Model},
  journal = {Applied Artificial Intelligence},
  volume  = {36},
  number  = {1},
  pages   = {2145642},
  year    = {2022},
  doi     = {10.1080/08839514.2022.2145642}
}
```
