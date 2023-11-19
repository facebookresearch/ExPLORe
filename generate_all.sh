#!/bin/bash
mkdir -p sbatch
./generate_antmaze.sh $1 $2 $3
./generate_adroit.sh $1 $2 $3
./generate_cog.sh $1 $2 $3
