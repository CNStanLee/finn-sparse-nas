#!/usr/bin/env bash
set -e

python -m baselines.finn_jsc.main --cfg configs/baseline.yaml --finn-cfg configs/finn.yaml --arch jsc-s
python -m baselines.finn_jsc.main --cfg configs/baseline.yaml --finn-cfg configs/finn.yaml --arch jsc-m
python -m baselines.finn_jsc.main --cfg configs/baseline.yaml --finn-cfg configs/finn.yaml --arch jsc-l
