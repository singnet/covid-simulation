# Simulation parameters

* allowed_restaurant_capacity
* [asymptomatic_contagion_probability](#asymptomatic_contagion_probability)
* contagion_probability
* extroversion_mean
* extroversion_stdev
* herding_behavior_mean
* herding_behavior_stdev
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

To check whether an infected person is contagious, either of these conditions must be true:

* its infection latency period is ended (i.e. the person is already symptomatic)
* a _flip coin_ test with `asymptomatic_contagion_probability` must pass.

__Valid values__: [0,1]

__Where it's used__: `Human.is_contagious()` in `human.py`

