# Simulation parameters

* allowed_restaurant_capacity
* [asymptomatic_contagion_probability](#asymptomatic_contagion_probability)
* [contagion_probability](#contagion_probability)
* [extroversion_mean and extroversion_stdev](#extroversion_mean)
* [herding_behavior_mean and herding_behavior_stdev](#herding_behavior_mean)
* [hospitalization_capacity](#hospitalization_capacity)
* [hospitalization_period_duration_shape and hospitalization_period_duration_scale](#hospitalization_period_duration_scale)
* [imune_rate](#imune_rate)
* [incubation_period_shape and incubation_period_scale](#incubation_period_scale)
* [initial_infection_rate](#initial_infection_rate)
* [latency_period_shape and latency_period_scale](#latency_period_shape)
* [mask_efficacy](#mask_efficacy)
* mask_user_rate
* mild_period_duration_scale
* mild_period_duration_shape
* min_behaviors_to_copy
* risk_tolerance_mean
* risk_tolerance_stdev
* social_policies
* spreading_rate
* symptomatic_isolation_rate
* typical_restaurant_event_size
* weareable_adoption_rate

## asymptomatic_contagion_probability

To check whether an infected human is contagious, either of these conditions must be true:

* its infection latency period is ended (i.e. the human is already symptomatic)
* a _flip coin_ test with `asymptomatic_contagion_probability` must pass.

__Valid values__: [0,1]

__Default__: 0

__Where it's used__: `Human.is_contagious()` in `human.py`

## contagion_probability

When a location is spreading infection amongst humans inside it, a _flip coin_ test with `contagion_probability` is performed every time a contagious human encounters a susceptible one.

`contagion_probability` may be different from one location to another. This can be set up in location's constructor (`setup_city_layout()` in `utils.py`).

__Valid values__: [0,1]

__Default__: 0

__Where it's used__: `Location.check_spreading()` in `location.py`

## extroversion_mean

__extroversion__ is one of Human's personal properties. `extroversion_mean` and `extroversion_stdev` are used to sample individual __extroversion__ for each human using a normal distribution.

__Valid values__: [0,1]

__Default__: 0.5 (`extroversion_mean`) and 0.3 (`extroversion_stdev`)

__Where it's used__: `extroversion_mean` and `extroversion_stdev` are used in `Human.initialize_individual_properties()` in `human.py`. __extroversion__ is not being used in the current code


## herding_behavior_mean

__herding_behavior__ is one of Human's personal properties. `herding_behavior_mean` and `herding_behavior_stdev` are used to sample individual __herding_behavior__ for each human using a normal distribution.

When a human is taking a decision on a given `Dilemma`, a _flip coin_ test with probability equals to __herding_behavior__ is perfomed to determine if the human will decide according to a herding behavior or not.

__Valid values__: [0,1]

__Default__: 0.4 (`herding_behavior_mean`) and 0.3 (`herding_behavior_stdev`)

__Where it's used__: `herding_behavior_mean` and `herding_behavior_stdev` are used in `Human.initialize_individual_properties()` in `human.py`. The human's property __herding_behavior__ is used in `Human._standard_decision()` in `human.py`

## hospitalization_capacity

The % of the total population which could be hospitalized at any given moment.

__Valid values__: [0,1]

__Default__: 0.05

__Where it's used__: `CovidModel.reached_hospitalization_limit()` in `base.py`

## hospitalization_period_duration_shape

`Human.hospitalization_duration` is the number of days which a human remains hospitalized when its infection reaches severity `MODERATE`. `hospitalization_period_duration_shape` and `hospitalization_period_duration_scale` are used to sample `Human.hospitalization_duration` using a gamma distribution when a human is infected.

__Valid values__: (1, 2, 3, ...)

__Default__: 14 (`hospitalization_period_duration_shape`) and 1 (`hospitalization_period_duration_scale`)

__Where it's used__: `Human.disease_evolution()` in `human.py` (`Human.hospitalization_duration` is used in the same function)

## imune_rate

`Human.immune` is a boolean attribute set when humans are instantiated. A _flip coin_ test with probability `imune_rate` is performed to decide whether each created human is imune to COVID-19 or not.

__Valid values__: [0,1]

__Default__: 0.05

__Where it's used__: `Human.parameter_changed()` in `human.py`. `Human.immune` (the human attribute) is used in `Human.infect()` in `human.py`.

## incubation_period_shape

`Human.infection_incubation` is the number of days an infected human remains asymptomatic after being infected. `incubation_period_shape` and `incubation_period_scale` are used to sample `Human.infection_incubation` using a gamma distribution when the human gets infected.

__Valid values__: (1, 2, 3, ...)

__Default__: 7 (`incubation_period_shape`) and 2 (`incubation_period_scale`)

__Where it's used__: `Human.infect()` in `human.py`. `Human.infection_incubation` (the human attribute) is used in `Human.is_symptomatic()` and `Human.disease_evolution()` in `human.py`.

## initial_infection_rate

The expected % of total pupulation which is infected before the simulation starts. When each human is created, a _flip coin_ test with probability equals to `initial_infection_rate` is used to decide whether that human will start the simulation infected or not.

__Valid values__: [0,1]

__Default__: 0.05

__Where it's used__: `Human.factory()` in `human.py`.

## latency_period_shape

`Human.infection_latency` is the number of days an infected human remains non-contagious after being infected. `latency_period_shape` and `latency_period_scale` are used to sample `Human.infection_latency` using a gamma distribution when the human gets infected.

__Valid values__: (1, 2, 3, ...)

__Default__: 4 (`latency_period_shape`) and 1 (`latency_period_scale`)

__Where it's used__: `Human.infect()` in `human.py`. `Human.infection_latency` (the human attribute) is used in `Human.is_contagious()` in `human.py`.

## mask_efficacy

When a location is spreading infection amongst humans inside it, every time a contagious human which is weariung a mask encounters a susceptible human, a _flip coin_ test with probability equals to `mask_efficacy` is performed to check is the mask prevented the susceptible human from being infected.

__Valid values__: [0,1]

__Default__: 0

__Where it's used__: `Location.check_spreading()` in `location.py`.


