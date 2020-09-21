# Simulation parameters

* allowed_restaurant_capacity
* [asymptomatic_contagion_probability](#asymptomatic_contagion_probability)
* [contagion_probability](#contagion_probability)
* [extroversion_mean and extroversion_stdev](#extroversion_mean)
* [herding_behavior_mean and herding_behavior_stdev](#herding_behavior_mean)
* hospitalization_capacity
* hospitalization_period_duration_scale
* hospitalization_period_duration_shape
* imune_rate
* incubation_period_scale
* incubation_period_shape
* initial_infection_rate
* isolation_cheater_rate
* isolation_cheating_severity
* latency_period_scale
* latency_period_shape
* mask_efficacy
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

__Default__: 0.5 (extroversion_mean) and 0.3 (extroversion_stdev)

__Where it's used__: `extroversion_mean` and `extroversion_stdev` are used in `Human.initialize_individual_properties()` in `human.py`. __extroversion__ is not being used in the current code


## herding_behavior_mean

__herding_behavior__ is one of Human's personal properties. `herding_behavior_mean` and `herding_behavior_stdev` are used to sample individual __herding_behavior__ for each human using a normal distribution.

When a human is making a decision on a given `Dilemma`, a _flip coin_ test with probability equals to __herding_behavior__ is perfomed to determine if the human will decide according to a herding behavior or not.

__Default__: 0.4 (herding_behavior_mean) and 0.3 (herding_behavior_stdev)

__Where it's used__: `herding_behavior_mean` and `herding_behavior_stdev` are used in `Human.initialize_individual_properties()` in `human.py`. The human's property __herding_behavior__ is used in `Human._standard_decision()` in `human.py`

