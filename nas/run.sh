#!/usr/bin/env bash
set -e

python -m nas.nas_main --cfg configs/nas_mlp.yaml --finn-cfg configs/finn.yaml

python -m nas.finalise_topk --cfg configs/nas_mlp.yaml --finn-cfg configs/finn.yaml --run-dir results/nas/<RUN_ID> --top-k 5 --pruning-mode baseline
python -m nas.finalise_topk --cfg configs/nas_mlp.yaml --finn-cfg configs/finn.yaml --run-dir results/nas/<RUN_ID> --top-k 5 --pruning-mode unstructured
python -m nas.finalise_topk --cfg configs/nas_mlp.yaml --finn-cfg configs/finn.yaml --run-dir results/nas/<RUN_ID> --top-k 5 --pruning-mode structured