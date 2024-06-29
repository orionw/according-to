#!/bin/bash
if [ $# -ne 2 ]; then
    python src/recalculate.py -f $1 
else
    python src/recalculate.py -f $1 -o $2
fi