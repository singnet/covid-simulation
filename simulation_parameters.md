# Simulation parameters

* [allowed_restaurant_capacity](#allowed_restaurant_capacity)
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
* [mask_user_rate](#mask_user_rate)
* [mild_period_duration_shape and mild_period_duration_scale](#mild_period_duration_shape)
* [min_behaviors_to_copy](#min_behaviors_to_copy)
* [risk_tolerance_mean and risk_tolerance_stdev](#risk_tolerance_mean)
* [social_policies](#social_policies)
* [symptomatic_isolation_rate](#symptomatic_isolation_rate)
* [typical_restaurant_event_size](#typical_restaurant_event_size)
* [weareable_adoption_rate](#weareable_adoption_rate)

## allowed_restaurant_capacity

`allowed_restaurant_capacity` is used in two different contexts:

* To determine the availability of restaurants when a human decided to invite friends.
* In a function to determine if infection will spread amongst humans from different events in a given restaurant. When a restaurant is spreading infection amongst the humans inside it, humans participating in an event will spread infection amongst themselves and a _flip coin_ test using `allowed_restaurant_capacity` is performed to check if some spreading will happen betwwen humans in different events (i.e. different tables in the Restaurant).

## asymptomatic_contagion_probability

To check whether an infected human is contagious, either of these conditions must be true:

__Valid values__: {1.0, 0.50, 0.25}

__Default__: 1.0

__Where it's used__: `Restaurant.spread_infection()` and `District.get_available_restaurant()` in `human.py`

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

When a location is spreading infection amongst humans inside it, every time a contagious human which is wearing a mask encounters a susceptible human, a _flip coin_ test with probability equals to `mask_efficacy` is performed to check if the mask prevented the susceptible human from being infected.

__Valid values__: [0,1]

__Default__: 0

__Where it's used__: `Location.check_spreading()` in `location.py`.

## mask_user_rate

When a location is spreading infection amongst humans inside it, every time a contagious human encounters a susceptible one, a _flip coin_ test with probability equals to mask_user_rate is performed to determine if the contagious human is wearing a mask or not.

__Valid values__: [0,1]

__Default__: 0

__Where it's used__: `Location.check_spreading()` in `location.py`.

## mild_period_duration_shape

`Human.mild_duration` is the number of days (after infection) which should be passed before performing a test to check whether the disease severity for that particular human will have severity > LOW. `mild_period_duration_shape` and `mild_period_duration_scale` are used to sample `Human.mild_duration` using a gamma distribution.

__Valid values__: [0,1]

__Default__: 14 (`mild_period_duration_shape`) and 1 (`mild_period_duration_scale`)

__Where it's used__: `Human.infect()` in `human.py`. `Human.mild_duration` is used in `Human.disease_evolution()` in `human.py`

## min_behaviors_to_copy

When a human is taking a decision on a given `Dilemma`, `min_behaviors_to_copy` is used to compute what would be a decision based in herding behavior.

__Valid values__: (1, 2, 3, ...)

__Default__: 3

__Where it's used__: `Human.personal_decision()` in `human.py`.

## risk_tolerance_mean

__risk_tolerance__ is one of Human's personal properties. `risk_tolerance_mean` and `risk_tolerance_stdev` are used to sample individual __risk_tolerance__ for each human using a normal distribution.

__risk_tolerance__ affects Human's decisions in a couple of different places:

* When deciding how to answer to a `Dilemma` 
    * when deciding on `Dilemma.GO_TO_WORK_ON_LOCKDOWN`, it's used as probability in a `flip coin` test to check if personal decision is `YES` or `NO`.
    * when deciding on `Dilemma.INVITE_FRIENDS_TO_RESTAURANT` and `Dilemma.ACCEPT_FRIEND_INVITATION_TO_RESTAURANT`, it's used as one of the parameters in a function to compute the personal decision.
* it's also used after the human decided to invite friends to a restaurant to compute the size of the event (the number of other humans that will be invited) and the type of reataurant they will go to.

__Valid values__: [0,1]

__Default__: 0.4 (`risk_tolerance_mean`) and 0.3 (`risk_tolerance_stdev`)

__Where it's used__: `herding_behavior_mean` and `herding_behavior_stdev` are used in `Human.initialize_individual_properties()` in `human.py`. The human's property __herding_behavior__ is used in `Human._standard_decision()` in `human.py`

## social_policies

A list of social policies which are in place. Different `social_policies` are used in different points during the simulation:

* `SocialPolicy.SOCIAL_DISTANCING` is used when a human is deciding on a `Dilemma.INVITE_FRIENDS_TO_RESTAURANT` or a `Dilemma.ACCEPT_FRIEND_INVITATION_TO_RESTAURANT`. It's one of the parameters in a function to compute the answer.
* `DocialPolicy.LOCKDOWN_*` are used when the human is deciding what to in your main activity (go to schools, go to the office, etc).

__VALID VALUES__: Any subset of {`SocialPolicy.SOCIAL_DISTANCING`, `SocialPolicy.LOCKDOWN_ALL`, `SocialPolicy.LOCKDOWN_OFFICE`, `SocialPolicy.LOCKDOWN_FACTORY`, `SocialPolicy.LOCKDOWN_RETAIL`, `SocialPolicy.LOCKDOWN_ELEMENTARY_SCHOOL`, `SocialPolicy.LOCKDOWN_MIDDLE_SCHOOL`, `SocialPolicy.LOCKDOWN_HIGH_SCHOOL`}

__DEFAULT__: {}

__Where it's used__: `Human.personal_decision()` and `Human.main_activity_isolated()` in `human.py`

## symptomatic_isolation_rate

When a symptomatic infected (infection severity <= LOW) human is deciding what to do in its main activity (go to school, go to the office, etc), a _flip coin_ test with probability equals to `symptomatic_isolation_rate` is performed to decide whether the human will go to its main activity or not.

__VALID VALUES__: [0,1]

__DEFAULT__: 0

__Where it's used__: `Human.main_activity_isolated()` in `human.py`

## typical_restaurant_event_size

When a human decided to invite friends to a restaurant, `typical_restaurant_event_size` is used as one of the parameters in a function to determine the number of other humans that will be invited.

__VALID VALUES__: (1, 2, 3, ...)

__DEFAULT__: 6

__Where it's used__: `Human.invite_friends_to_restaurant()` in `human.py`

## weareable_adoption_rate

Humans have an attribute `Human.early_symptom_detection` which is the number of days before the end of the latency period which the symptoms of infection will appear in that specific human. In other words, it's a bias towards down in the number of days before the symptoms appear in that human.

`weareable_adoption_rate` is used when each human is created, when a _flip coin_ test with probability equals to `weareable_adoption_rate` is performed to determine whether that human is a user of a wearable which would increase `Human.early_symptom_detection` by 1.

__VALID VALUES__: [0,1]

__DEFAULT__: 0

__Where it's used__: `Human.parameter_changed()` in `human.py`
