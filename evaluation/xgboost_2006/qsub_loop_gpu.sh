#!/bin/bash
for i in $(seq -f "%03g" 6 10)
do
  # echo "qsub $i.sub"
  qsub $i.sub
done

# https://stackoverflow.com/questions/8789729/how-to-zero-pad-a-sequence-of-integers-in-bash-so-that-all-have-the-same-width