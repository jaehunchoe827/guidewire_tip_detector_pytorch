#!/bin/bash
echo "Training ver3_sigma_1.yaml"
python3 -m engine.main --train --config ver3_sigma_1.yaml
sleep 5
echo "Training ver3_sigma_2.yaml"
python3 -m engine.main --train --config ver3_sigma_2.yaml
sleep 5
echo "Training ver3_sigma_4.yaml"
python3 -m engine.main --train --config ver3_sigma_4.yaml
sleep 5
echo "Training ver3_sigma_8.yaml"
python3 -m engine.main --train --config ver3_sigma_8.yaml
sleep 5
echo "Training ver3_sigma_16.yaml"
python3 -m engine.main --train --config ver3_sigma_16.yaml
sleep 5
echo "All models trained successfully!"
