#!/bin/bash
echo "Training ver3.yaml"
python3 -m engine.main --train --config ver3.yaml
sleep 5
echo "Training ver4.yaml"
python3 -m engine.main --train --config ver4.yaml

echo "All models trained successfully!"