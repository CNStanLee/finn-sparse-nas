# finn-sparse-nas

A hardware-aware Neural Architecture Search (NAS) and post-hoc pruning framework for FINN-deployed quantised neural networks (QNNs), developed for two FPGA-relevant case studies:

- a quantised MLP for jet substructure classification (JSC), inspired by the LogicNets example family
- a quantised CNV model for CIFAR-10 image classification, adapted from the Brevitas / FINN BNN-PYNQ examples

The repository contains:
- recreated reference baselines for both model families
- a shared task-agnostic evolutionary NAS pipeline
- finalist reevaluation with full FINN builds
- post-hoc unstructured and structured pruning utilities

## Requirements

The host-side code was developed for **Python 3.12**.
Using pip or conda, install the required Python packages with:
```bash
pip install -r requirements.txt
```
In addition, FINN must be available through a Docker image. Update the paths and settings in:
```
configs/finn.yaml
```
so that they point to your local FINN installation and Docker environment. This repository specifically used the following version of FINN: [Xilinx/finn@9b1f45e](https://github.com/Xilinx/finn/commit/9b1f45e9)


## Dataset setup

Download the processed Jet Substructure Classification (JSC) dataset with:
```bash
mkdir -p data && \
wget https://cernbox.cern.ch/s/jvFd5MoWhGs1l5v/download -O data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z
```
The CIFAR-10 dataset is downloaded automatically by the PyTorch dataset wrapper when required.

## Running the recreated baselines

JSC MLP baseline (--arch param can be jsc-s, jsc-m or jsc-l):
```python
python -m baselines.finn_jsc.main --cfg configs/baseline_mlp.yaml --finn-cfg configs/finn.yaml --arch jsc-s
```
CIFAR-10 CNV baseline (--arch can be cnv_1w1a, cnv_1w2a or cnv_2w2a):
```python
python -m baselines.finn_cifar10.main --cfg configs/baseline_cnv.yaml --finn-cfg configs/finn.yaml --arch cnv_1w1a
```

## Running NAS experiments

JSC MLP NAS:
```python
python -m nas.nas_main --cfg configs/nas_mlp.yaml --finn-cfg configs/finn.yaml
```
CIFAR-10 CNV NAS:
```python
python -m nas.nas_main --cfg configs/nas_cnv.yaml --finn-cfg configs/finn.yaml
```
Each NAS run creates a results directory under:
```
results/nas/<RUN_ID>
```

## Finalist reevaluation and pruning

After each NAS run, reevaluate the top-ranked finalists by passing the corresponding run directory to `nas.finalise_topk`.  
Use `--pruning-mode baseline`, `--pruning-mode unstructured`, or `--pruning-mode structured`.

JSC MLP finalists:
```python
python -m nas.finalise_topk --cfg configs/nas_mlp.yaml --finn-cfg configs/finn.yaml --run-dir results/nas/<RUN_ID> --top-k 5 --pruning-mode baseline
```

CIFAR-10 CNV finalists:
```python
python -m nas.finalise_topk --cfg configs/nas_cnv.yaml --finn-cfg configs/finn.yaml --run-dir results/nas/<RUN_ID> --top-k 5 --pruning-mode baseline
```

## Repository Layout

```
.
├── baselines/
│   ├── finn_cifar10/           # recreated FINN baseline for CIFAR-10 CNV presets
│   └── finn_jsc/               # recreated FINN baseline for LogicNets-style JSC MLP presets
├── configs
│   ├── baseline_cnv.yaml       # recreated CNV baseline presets
│   ├── baseline_mlp.yaml       # recreated JSC baseline presets
│   ├── finn.yaml               # shared Docker / FINN runtime settings
│   ├── nas_cnv.yaml            # NAS and finalist settings for the CIFAR-10 CNV task
│   └── nas_mlp.yaml            # NAS and finalist settings for the JSC MLP task
├── datasets/                   # dataset wrappers and task-specific dataset utilities
├── finn_integration/           # Docker launch helpers and FINN report parsing
├── models/                     # Brevitas model definitions for both case studies
├── nas/
│   ├── nas_main.py             # main evolutionary NAS loop
│   ├── finalise_topk.py        # finalist retraining, pruning, and final FINN builds
│   ├── ea_ops.py               # evolutionary operators
│   ├── task_factory.py         # task-agnostic model/data/training interface
│   ├── train.py                # shared NAS training utilities
│   ├── build_finn.py           # candidate export and FINN build entry point
│   ├── structured_pruning.py
│   ├── unstructured_pruning.py
│   └── utils.py
├── results/                    # generated NAS runs and finalist outputs
├── README.md
└── requirements.txt
```