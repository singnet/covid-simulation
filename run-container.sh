#!/bin/bash

docker run -p 8887:8887 --name covid_simulation -v "$PWD":/opt/singnet/covid-simulation -ti covid-simulation /bin/bash -c "python3 -m pip install -r requirements.txt; bash"