#!/bin/bash

echo "Training ver3_default_yolo_large.yaml"
python3 -m engine.main --train --config ver3_default_yolo_large.yaml 
sleep 3

echo "Training ver3_default.yaml"
python3 -m engine.main --train --config ver3_default.yaml 
sleep 3

echo "Training ver3_default_smaller_lr.yaml"
python3 -m engine.main --train --config ver3_default_smaller_lr.yaml 
sleep 3

echo "All models trained successfully!"
