#!/bin/bash

docker run -p 8887:8887 --name covid_simulation --mount "type=bind,src=$(pwd),dst=/opt/singnet/covid-simulation" -ti covid-simulation bash