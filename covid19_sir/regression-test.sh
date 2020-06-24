#!/bin/sh

\rm -rf /tmp/regression-test/
mkdir /tmp/regression-test/
#python3 regression-test.py 0.6 &&\
#python3 regression-test.py 0.5 &&\
#python3 regression-test.py 0.4 &&\
#python3 regression-test.py 0.3 &&\
python3 simple-simulation.py 12234 &&\
mv scenario[1234].* /tmp/regression-test/ &&\
#python3 wearable-simulation.py 12234 &&\
#mv wearable_scenario[123].* /tmp/regression-test/ &&\
diff /tmp/regression-test/ regression-test-baseline/
