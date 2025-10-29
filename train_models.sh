#!/bin/bash
echo "Training ver3_unfreeze_at_2_yolo_m.yaml"
python3 -m engine.main --train --config ver3_unfreeze_at_2_yolo_m.yaml 
sleep 5
echo "Training ver3_unfreeze_at_2_high_wd.yaml"
python3 -m engine.main --train --config ver3_unfreeze_at_2_high_wd.yaml 
sleep 5
echo "Training ver3_unfreeze_at_2_lower_lr.yaml"
python3 -m engine.main --train --config ver3_unfreeze_at_2_lower_lr.yaml 
echo "All models trained successfully!"
