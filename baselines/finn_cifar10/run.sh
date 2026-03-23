#!/usr/bin/env bash
set -e

python -m baselines.finn_cifar10.main --cfg configs/baseline_cnv.yaml --finn-cfg configs/finn.yaml --arch cnv_1w1a
python -m baselines.finn_cifar10.main --cfg configs/baseline_cnv.yaml --finn-cfg configs/finn.yaml --arch cnv_1w2a
python -m baselines.finn_cifar10.main --cfg configs/baseline_cnv.yaml --finn-cfg configs/finn.yaml --arch cnv_2w2a
