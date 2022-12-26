#!/bin/bash
CONFIGS=("roberta_large_lr01" "roberta_large_lr02" "roberta_large_lr03" "roberta_large_lr04" "roberta_large_lr06" "roberta_large_lr07" "roberta_large_lr08" "roberta_large_lr09" )

for (( i=0; i<8; i++ ))
do
    python3 train.py --config ${CONFIGS[$i]}
done