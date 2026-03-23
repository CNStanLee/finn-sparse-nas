# finn-sparse-nas

## Prerequisites

Using pip or conda, install the packages included in requirements.txt for running on host (Python=3.12).
```bash
pip install -r requirements.txt
```
Install the FINN docker image and update the paths in configs/finn.yaml to point to it.

## Download the JSC dataset

```bash
mkdir -p data && \
wget https://cernbox.cern.ch/s/jvFd5MoWhGs1l5v/download -O data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z
```

## Running

The NAS JSC implementation:

```python
python -m nas.nas_main --cfg configs/nas_mlp.yaml --finn-cfg configs/finn.yaml
```

The FINN-JSC baseline (define the --arch param to be either one of jsc-s, jsc-m or jsc-l):

```python
python -m baselines.finn_jsc.main --cfg configs/baseline_mlp.yaml --finn-cfg configs/finn.yaml --arch jsc-s
```

The FINN CIFAR-10 baseline (--arch can be cnv_1w1a, cnv_1w2a or cnv_2w2a):

```python
python -m baselines.finn_cifar10.main --cfg configs/baseline_cnv.yaml --finn-cfg configs/finn.yaml --arch cnv_1w1a
```

## Layout

```
.
├── baselines
│   ├── finn_cifar10                            # faithful FINN baseline for Brevitas CIFAR-10 CNV presets
│   │   ├── build_finn.py
│   │   ├── folding_cfgs
│   │   │   ├── cnv_1w1a.json
│   │   │   ├── cnv_1w2a.json
│   │   │   ├── cnv_2w2a.json
│   │   │   └── cnv_specialize_layers.json
│   │   ├── main.py
│   │   ├── run.sh
│   │   └── train.py
│   └── finn_jsc                                # faithful FINN baseline for the LogicNets JSC MLP presets
│       ├── build_finn.py
│       ├── folding_cfgs
│       │   ├── folding_jsc-l_latency.json
│       │   ├── folding_jsc-l_resource.json
│       │   ├── folding_jsc-s_latency.json
│       │   └── folding_jsc-s_resource.json
│       ├── main.py
│       ├── run.sh
│       └── train.py
├── configs
│   ├── baseline_cnv.yaml                       # config for CIFAR-10 CNV baselines and preset exports
│   ├── baseline_mlp.yaml                       # config for JSC MLP baselines and preset exports
│   ├── finn.yaml                               # Docker / FINN runtime settings shared across builds
│   ├── nas_cnv.yaml                            # NAS search space and training settings for CNV experiments
│   └── nas_mlp.yaml                            # NAS search space and training settings for MLP experiments
├── datasets
│   ├── cifar10
│   │   └── dataset.py                          # CIFAR-10 dataset wrapper with train / val / test splits
│   └── jsc_logicnets
│       ├── dataset.py                          # Jet Substructure dataset wrapper adapted from LogicNets
│       └── yaml_IP_OP_config.yaml              # original LogicNets feature / label dataset description
├── finn_integration
│   ├── finn_client.py                          # helper for launching FINN builds inside Docker
│   └── report_parser.py                        # parses FINN build artifacts into compact JSON summaries
├── models
│   ├── brevitas_cnv.py                         # Brevitas-based quantised CNV model adapted for this project
│   └── brevitas_mlp.py                         # Brevitas-based quantised dense MLP for Jet Substructure
├── nas
│   ├── build_finn.py                           # candidate-based QONNX export and FINN build entry point
│   ├── ea_ops.py                               # evolutionary search operators: sampling, mutation, crossover
│   ├── finalise_topk.py                        # trains top-K candidates fully, applies pruning, and runs final FINN builds
│   ├── nas_main.py                             # main evolutionary NAS loop with caching and candidate evaluation
│   ├── run.sh
│   ├── structured_pruning.py                   # structured neuron pruning utilities and structured sweep helpers
│   ├── train.py                                # shared training / evaluation utilities for NAS candidates
│   ├── unstructured_pruning.py                 # unstructured weight pruning utilities
│   └── utils.py                                # shared helpers: hashing, JSON export, directories, small utilities
├── README.md
└── requirements.txt
```