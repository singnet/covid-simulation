#!/bin/sh

\rm -rf /tmp/regression-test/
mkdir /tmp/regression-test/
python3 simple-simulation.py 12334 &&\
mv scenario[12345678].csv /tmp/regression-test/ &&\
mv scenario[12345678].png /tmp/ &&\
diff /tmp/regression-test/ regression-test-baseline/
