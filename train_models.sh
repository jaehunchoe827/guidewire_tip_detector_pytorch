#!/bin/bash
sleep 5
echo "Training ver3_unfreeze_at_1_wider.yaml"
python3 -m engine.main --train --config ver3_unfreeze_at_1_wider.yaml 
sleep 5
echo "Training ver3_unfreeze_at_1_mse.yaml"
python3 -m engine.main --train --config ver3_unfreeze_at_1_mse.yaml 
sleep 5
echo "Training ver3_unfreeze_at_1_low_wd.yaml"
python3 -m engine.main --train --config ver3_unfreeze_at_1_low_wd.yaml 
sleep 5
echo "All models trained successfully!"
