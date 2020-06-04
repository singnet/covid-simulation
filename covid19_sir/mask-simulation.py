import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from model import CovidModel, Location, SimulationParameters, set_parameters
from utils import SimpleLocation, BasicStatistics

################################################################################
# Common parameters amongst all scenarios

# COVID model

mask_user_rate = 0.0
mask_efficacy = 0.0
isolation_cheater_rate = 0.0
isolation_cheating_severity = 0.0
imune_rate = 0.005
initial_infection_rate = 0.05
hospitalization_capacity = 0.001
latency_period_mean = 2.0
latency_period_stdev = 1.0
incubation_period_mean = 14.0
incubation_period_stdev = 4.0
disease_period_mean = 25
disease_period_stdev = 5
daily_interaction_count = 40
contagion_probability = 0.9
asymptomatic_isolation_rate = 0.2
symptomatic_isolation_rate = 0.6

# Simulation

population_size = 10000
simulation_cycles = 120 # days

################################################################################
# Simulation

epochs = 50

ICR = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ICS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Baseline

set_parameters(SimulationParameters(
    mask_user_rate = 0.0,
    mask_efficacy = 0.0,
    isolation_cheater_rate = 0.0,
    isolation_cheating_severity = 0.0,
    imune_rate = imune_rate,
    initial_infection_rate = initial_infection_rate,
    hospitalization_capacity = hospitalization_capacity,
    latency_period_mean = latency_period_mean,
    latency_period_stdev = latency_period_stdev,
    incubation_period_mean = incubation_period_mean,
    incubation_period_stdev = incubation_period_stdev,
    disease_period_mean = disease_period_mean,
    disease_period_stdev = disease_period_stdev,
    asymptomatic_isolation_rate = asymptomatic_isolation_rate,
    symptomatic_isolation_rate = symptomatic_isolation_rate,
    daily_interaction_count = daily_interaction_count,
    contagion_probability = contagion_probability
))

sum = 0.0
for k in range(epochs):
    model = CovidModel()
    location = SimpleLocation(0, model, population_size)
    statistics = BasicStatistics(model)
    model.add_location(location)
    model.add_listener(statistics)
    for i in range(simulation_cycles):
        model.step()
    sum += statistics.death[-1]
baseline_deaths = sum / epochs
print("baseline_deaths: " + str(baseline_deaths))

# Parameter variation

ME = float(sys.argv[1])
difference = [None] * len(ICR)
for i in range(len(ICR)):
    difference[i] = [None] * len(ICS)
    for j in range(len(ICS)):
        sum = 0.0
        #print(str(i) + " " + str(j))
        for k in range(epochs):
            set_parameters(SimulationParameters(
                mask_user_rate = 0.4,
                mask_efficacy = ME,
                me_attenuation = 1.0,
                isolation_cheater_rate = ICR[i],
                isolation_cheating_severity = ICS[j],
                imune_rate = imune_rate,
                initial_infection_rate = initial_infection_rate,
                hospitalization_capacity = hospitalization_capacity,
                latency_period_mean = latency_period_mean,
                latency_period_stdev = latency_period_stdev,
                incubation_period_mean = incubation_period_mean,
                incubation_period_stdev = incubation_period_stdev,
                disease_period_mean = disease_period_mean,
                disease_period_stdev = disease_period_stdev,
                asymptomatic_isolation_rate = asymptomatic_isolation_rate,
                symptomatic_isolation_rate = symptomatic_isolation_rate,
                daily_interaction_count = daily_interaction_count,
                contagion_probability = contagion_probability
            ))
            model = CovidModel()
            location = SimpleLocation(0, model, population_size)
            statistics = BasicStatistics(model)
            model.add_location(location)
            model.add_listener(statistics)
            for g in range(simulation_cycles):
                model.step()
            sum += statistics.death[-1]
        print("deaths: " + str(sum / epochs))
        difference[i][j] = (sum / epochs) - baseline_deaths
cs = plt.contourf(ICR, ICS, difference, 
                  levels=[-0.015, -0.010, -0.005, 0.0, 0.005, 0.010, 0.015],
                  colors=['#336699', '#4183C4', '#53A7FB', '#7C7900', '#B2AC00', '#FFF700'], 
                  extend='both')
cs.cmap.set_over('#FFF700')
cs.cmap.set_under('#336699')
plt.title('Change in number of deaths (pp)')
plt.xlabel('Isolation cheating rate (%)')
plt.ylabel('Isolation cheating severity (%)')
plt.colorbar(cs)
cs.changed()
plt.savefig(str(ME) + ".png")
