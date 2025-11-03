#!/bin/bash
echo "Training ver3_longer_epoch.yaml"
python3 -m engine.main --train --config ver3_longer_epoch.yaml 
sleep 5
echo "Training ver3_larger_batch.yaml"
python3 -m engine.main --train --config ver3_larger_batch.yaml 
sleep 5
echo "Training ver3_less_aug.yaml"
python3 -m engine.main --train --config ver3_less_aug.yaml 
sleep 5
echo "Training ver3_no_edge.yaml"
python3 -m engine.main --train --config ver3_no_edge.yaml 
sleep 5
echo "Training ver3_mae.yaml"
python3 -m engine.main --train --config ver3_mae.yaml 

echo "All models trained successfully!"
