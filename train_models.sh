#!/bin/bash

echo "Training step_lr.yaml"
python3 -m engine.main --config step_lr.yaml --train
echo "Process completed successfully!"

echo "Training wider_head.yaml"
python3 -m engine.main --config wider_head.yaml --train
echo "Process completed successfully!"

echo "Training deeper_head.yaml"
python3 -m engine.main --config deeper_head.yaml --train
echo "Process completed successfully!"

echo "Training yolo_m.yaml"
python3 -m engine.main --config yolo_m.yaml --train
echo "Process completed successfully!"



