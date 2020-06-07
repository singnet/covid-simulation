#!/bin/sh

\rm -rf /tmp/regression-test/
mkdir /tmp/regression-test/
echo "1/4"
python3 regression-test.py 0.6
echo "2/4"
python3 regression-test.py 0.5
echo "3/4"
python3 regression-test.py 0.4
echo "4/4"
python3 regression-test.py 0.3

diff /tmp/regression-test/ regression-test-baseline/

