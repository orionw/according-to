#!/bin/bash

for f in $1/*.json
do
    echo $f
    ./bin/do_recalculation.sh $f $f
done