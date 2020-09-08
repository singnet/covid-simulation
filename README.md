# COVID-19 simulations

This is the basic implementation of a simple MESA simulation model for COVID-19 based in 
Dr Petronio Silva's model described in [this article](https://towardsdatascience.com/agent-based-simulation-of-covid-19-health-and-economical-effects-6aa4ae0ff397) 
and implemented in [this repository](https://github.com/petroniocandido/COVID19_AgentBasedSimulation).

The model is under `covid19_sir/`. A Dockerfile is provided to ease the configuration of the required environment. To build the docker image:

```
$ ./build-image.sh
```

Once the image is built, start a container:

```
$ ./run-container.sh
```

Then inside the container:

```
$ cd covid19_sir/
$ python3 simple-simulation.py
```

The output is a bunch of `scenario*.png` and `scenario*.csv` files. A pair for each simulation scenario.

Scenarios can be edited/added/deleted in `simple-simulation.py`.

Optionally you can run the simulation from a Jupyter Notebook:

```
./start-notebook.sh
```

Then open the indicated URL in a browser on the host machine.

## Testing

To run unit tests on this repository, simply cd to `covid19_sir` (or the desired subdirectory) and run `python3 -m pytest`.