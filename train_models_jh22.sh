#!/bin/bash
echo "Training ver3_less_aug_more_rot.yaml"
python3 -m engine.main --train --config ver3_less_aug_more_rot.yaml
sleep 5

echo "All models trained successfully!"