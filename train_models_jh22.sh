#!/bin/bash
echo "Training ver5_amplifier_1.yaml"
python3 -m engine.main --train --config ver5_amplifier_1.yaml
sleep 5
echo "Training ver5_amplifier_10.yaml"
python3 -m engine.main --train --config ver5_amplifier_10.yaml
sleep 5
echo "Training ver5_amplifier_100.yaml"
python3 -m engine.main --train --config ver5_amplifier_100.yaml
sleep 5
echo "Training ver5_amplifier_1000.yaml"
python3 -m engine.main --train --config ver5_amplifier_1000.yaml
sleep 5

echo "All models trained successfully!"