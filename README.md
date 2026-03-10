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

The NAS implementation:

```python
python -m nas.nas_main --cfg configs/nas.yaml --finn-cfg configs/finn.yaml
```
The FINN-JSC baseline (define the --arch param to be either one of jsc-s, jsc-m or jsc-l):

```python
python -m baselines.finn_jsc.main --cfg configs/baseline.yaml --finn-cfg configs/finn.yaml --arch jsc-s
```

## Layout

```
.
├── baselines                                   # direct comparison to JSC Logicnets
│   └── finn_jsc                                # but using FINN instead
│       ├── folding_cfgs
│       │   ├── folding_jsc-l_latency.json
│       │   ├── folding_jsc-l_resource.json
│       │   ├── folding_jsc-s_latency.json
│       │   └── folding_jsc-s_resource.json
│       ├── build_finn.py
│       ├── main.py
│       ├── run.sh
│       └── train.py
├── configs
│   ├── baseline.yaml
│   ├── finn.yaml
│   └── nas.yaml
├── datasets
│   └── jsc_logicnets
│       ├── dataset.py                          # dataset generation (from LogicNets repo)
│       └── yaml_IP_OP_config.yaml
├── finn_integration
│   ├── finn_client.py                          # run-docker.sh helper
│   └── report_parser.py                        # parses FINN build reports
├── models
│   └── brevitas_mlp.py                         # dense MLP using Brevitas
├── nas
│   ├── build_finn.py                           # FINN build inside Docker container
│   ├── ea_ops.py                               # EA operators and helpers
│   ├── finalise_topk.py                        # final validation for top K architectures
│   ├── nas_main.py                             # EA driver (NAS search loop)
│   ├── run.sh
│   ├── train.py                                # Pytorch training 
│   └── utils.py                                # other small helpers (hash, caching)
├── README.md
└── requirements.txt
```