#!/bin/sh

\rm -rf /tmp/regression-test/
mkdir /tmp/regression-test/
echo "1/6"
python3 regression-test.py 0.6
echo "2/6"
python3 regression-test.py 0.5
echo "3/6"
python3 regression-test.py 0.4
echo "4/6"
python3 regression-test.py 0.3
echo "5/6"
python3 simple-simulation.py 12234
mv scenario[1234].* /tmp/regression-test/
echo "6/6"
python3 wearable-simulation.py 12234
mv wearable_scenario[123].* /tmp/regression-test/

diff /tmp/regression-test/ regression-test-baseline/

