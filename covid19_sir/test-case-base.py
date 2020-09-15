import sys
import copy
import numpy as np
from model.base import CovidModel, SimulationParameters, set_parameters, normal_ci, logger
from utils import BasicStatistics, RemovePolicy, Propaganda, setup_city_layout
from model.utils import SocialPolicy
from model.debugutils import DebugUtils

basic_parameters = SimulationParameters(
    mask_user_rate=0.0,
    mask_efficacy=0.0,
    imune_rate=0.01,
    initial_infection_rate=0.01,
    hospitalization_capacity=0.001,
    latency_period_shape=3,
    latency_period_scale=1,
    incubation_period_shape=6,
    incubation_period_scale=1,
    mild_period_duration_shape=14,
    mild_period_duration_scale=1,
    hospitalization_period_duration_shape=12,
    hospitalization_period_duration_scale=1,
    symptomatic_isolation_rate=0.0,
    asymptomatic_contagion_probability=0.1,
    risk_tolerance_mean=0.7,
    risk_tolerance_stdev=0.2,
    herding_behavior_mean=0.7,
    herding_behavior_stdev=0.2,
    allowed_restaurant_capacity=1.0,  # valid values: {1.0, 0.50, 0.25}
    spreading_rate=normal_ci(2.41, 3.90, 20),
    social_policies=[
        SocialPolicy.LOCKDOWN_OFFICE,
        SocialPolicy.LOCKDOWN_FACTORY,
        SocialPolicy.LOCKDOWN_RETAIL,
        SocialPolicy.LOCKDOWN_ELEMENTARY_SCHOOL,
        SocialPolicy.LOCKDOWN_MIDDLE_SCHOOL,
        SocialPolicy.LOCKDOWN_HIGH_SCHOOL,
        SocialPolicy.SOCIAL_DISTANCING
    ]
)

# Simulation

population_size = 1000
simulation_cycles = 90  # days
multiple_runs = 5

single_var = ('risk_tolerance_mean', [0.1, 0.5, 0.9])

infections_in_restaurants = {}

var_name, var_values = single_var
for value in var_values:

    infections_in_restaurants[value] = []

    for k in range(multiple_runs):
        params = copy.deepcopy(basic_parameters)
        params.params[var_name] = value
        set_parameters(params)
        model = CovidModel()
        setup_city_layout(model, population_size)
        model.add_listener(Propaganda(model, 30))
        model.add_listener(RemovePolicy(model, SocialPolicy.LOCKDOWN_ELEMENTARY_SCHOOL, 30))
        model.add_listener(RemovePolicy(model, SocialPolicy.LOCKDOWN_MIDDLE_SCHOOL, 60))
        model.add_listener(RemovePolicy(model, SocialPolicy.LOCKDOWN_HIGH_SCHOOL, 90))
        statistics = BasicStatistics(model)
        model.add_listener(statistics)
        debug = DebugUtils(model)
        logger().model = model
        for i in range(simulation_cycles):
            model.step()
        statistics.export_chart(f'scenario_{var_name}_{value}_{k}.png')
        statistics.export_csv(f'scenario_{var_name}_{value}_{k}.csv')
        debug.update_infection_status()

        infections_in_restaurants[value].append(debug.count_restaurant)

print(infections_in_restaurants)
