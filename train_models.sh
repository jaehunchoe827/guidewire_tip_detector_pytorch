#!/bin/bash
sleep 5
echo "Training ver2_default.yaml"
python3 -m engine.main --train --config ver2_default.yaml 
sleep 5
echo "Training ver2_unfreeze_at_0.yaml"
python3 -m engine.main --train --config ver2_unfreeze_at_0.yaml 
sleep 5
echo "Training ver3_default.yaml"
python3 -m engine.main --train --config ver3_default.yaml 
sleep 5
echo "All models trained successfully!"
