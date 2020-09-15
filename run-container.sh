#!/bin/bash

docker run -p 8887:8887 --name covid_simulation_container --mount "type=bind,src=$(pwd),dst=/opt/singnet/covid-simulation" -v ~/Shared:/home/user/Shared -ti covid-simulation bash
