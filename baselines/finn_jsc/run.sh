#!/usr/bin/env bash
set -e

python -m baselines.finn_jsc.main --cfg configs/baseline_mlp.yaml --finn-cfg configs/finn.yaml --arch jsc-s
python -m baselines.finn_jsc.main --cfg configs/baseline_mlp.yaml --finn-cfg configs/finn.yaml --arch jsc-m
python -m baselines.finn_jsc.main --cfg configs/baseline_mlp.yaml --finn-cfg configs/finn.yaml --arch jsc-l
