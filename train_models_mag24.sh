#!/bin/bash
echo "Training ver3_less_aug_longer_epoch_larger_batch.yaml"
python3 -m engine.main --train --config ver3_less_aug_longer_epoch_larger_batch.yaml 
sleep 5
echo "Training ver3_less_aug_longer_epoch_larger_batch_unfreeze_3.yaml"
python3 -m engine.main --train --config ver3_less_aug_longer_epoch_larger_batch_unfreeze_3.yaml 

echo "All models trained successfully!"
