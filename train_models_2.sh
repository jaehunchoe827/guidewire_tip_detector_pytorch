#!/bin/bash
sleep 5
echo "Training ver3_unfreeze_at_2.yaml"
python3 -m engine.main --train --config ver3_unfreeze_at_2.yaml 
sleep 5
echo "Training ver3_unfreeze_at_3.yaml"
python3 -m engine.main --train --config ver3_unfreeze_at_3.yaml 
sleep 5
echo "All models trained successfully!"
