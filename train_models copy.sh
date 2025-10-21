#!/bin/bash
echo "Training default.yaml"
python3 -m engine.main --config default.yaml --train
echo "Training linear_lr.yaml"
python3 -m engine.main --config linear_lr.yaml --train
echo "Training mse.yaml"
python3 -m engine.main --config mse.yaml --train
echo "Training deeper_head.yaml"
python3 -m engine.main --config deeper_head.yaml --train
echo "Process completed successfully!"