#!/bin/sh

\rm -f simulation.log &&\
\rm -f /tmp/simulation.log &&\
cp -f regression-test-baseline/simulation.log.gz /tmp &&\
gunzip /tmp/simulation.log.gz &&\
python3 simple-simulation.py 31415 &&\
diff simulation.log /tmp/simulation.log
