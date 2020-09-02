#!/bin/bash

docker build -t covid-simulation --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .
