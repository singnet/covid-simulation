#!/bin/sh

\rm -rf /tmp/regression-test/
mkdir /tmp/regression-test/
python3 simple-simulation.py 12234 &&\
mv scenario[12345678].* /tmp/regression-test/ &&\
diff /tmp/regression-test/ regression-test-baseline/
