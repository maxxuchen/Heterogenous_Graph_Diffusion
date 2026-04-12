#!/usr/bin/env bash
# Run HGD and DDM baseline on MUTAG

set -e
CONFIG=configs/MUTAG.yaml

echo "=== Training HGD ==="
python train_hgd.py --config $CONFIG --output_dir logs/hgd

echo "=== Training DDM (baseline) ==="
python train_ddm.py --config $CONFIG --output_dir logs/ddm
