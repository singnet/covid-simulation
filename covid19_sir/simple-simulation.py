import sys
import copy
import numpy as np
from model.base import CovidModel, SimulationParameters, set_parameters, beta_range, logger
from utils import BasicStatistics, RemovePolicy, Propaganda, setup_city_layout
from model.utils import SocialPolicy
from model.debugutils import DebugUtils

seed = 31415
if len(sys.argv) > 1:
    seed = int(sys.argv[1])
np.random.seed(seed)

################################################################################
# Common parameters amongst all scenarios

# COVID model

common_parameters = SimulationParameters(
    mask_user_rate=0.0,
    mask_efficacy=0.0,
    imune_rate=0.01,
    initial_infection_rate=0.01,
    hospitalization_capacity=0.5,
    latency_period_shape=2,
    latency_period_scale=1,
    incubation_period_shape=4,
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
    spreading_rate=beta_range(2.41, 3.90)  # normal_ci(2.41, 3.90, 20)
)

# Simulation

population_size = 1000
simulation_cycles = 90  # days

################################################################################
# Scenarios

scenario = {}

# ------------------------------------------------------------------------------

sc = 1  # Do nothing
print(f"Setting up scenario {sc}")
scenario[sc] = {}
scenario[sc]['parameters'] = copy.deepcopy(common_parameters)
set_parameters(scenario[sc]['parameters'])
scenario[sc]['model'] = CovidModel()
np.random.seed(seed)
setup_city_layout(scenario[sc]['model'], population_size)

# ------------------------------------------------------------------------------

sc = 2  # Isolate symptomatic
print(f"Setting up scenario {sc}")
scenario[sc] = {}
scenario[sc]['parameters'] = copy.deepcopy(common_parameters)
scenario[sc]['parameters'].params['symptomatic_isolation_rate'] = 1.0
set_parameters(scenario[sc]['parameters'])
scenario[sc]['model'] = CovidModel()
np.random.seed(seed)
setup_city_layout(scenario[sc]['model'], population_size)

# ------------------------------------------------------------------------------

sc = 3  # complete lockdown
print(f"Setting up scenario {sc}")
scenario[sc] = {}
scenario[sc]['parameters'] = copy.deepcopy(common_parameters)
scenario[sc]['parameters'].params['social_policies'] = [
    SocialPolicy.LOCKDOWN_ALL
]
set_parameters(scenario[sc]['parameters'])
scenario[sc]['model'] = CovidModel()
np.random.seed(seed)
setup_city_layout(scenario[sc]['model'], population_size)

# ------------------------------------------------------------------------------

sc = 4  # Start with partial lockdown then gradually unlock schools
# on simulation day 30, 60 and 90
print(f"Setting up scenario {sc}")

scenario[sc] = {}
sc4_parameters = copy.deepcopy(common_parameters)
sc4_parameters.params['social_policies'] = [
    SocialPolicy.LOCKDOWN_OFFICE,
    SocialPolicy.LOCKDOWN_FACTORY,
    SocialPolicy.LOCKDOWN_RETAIL,
    SocialPolicy.LOCKDOWN_ELEMENTARY_SCHOOL,
    SocialPolicy.LOCKDOWN_MIDDLE_SCHOOL,
    SocialPolicy.LOCKDOWN_HIGH_SCHOOL,
    SocialPolicy.SOCIAL_DISTANCING
]
set_parameters(sc4_parameters)
sc4_model = CovidModel()
np.random.seed(seed)
setup_city_layout(sc4_model, population_size)
sc4_model.add_listener(RemovePolicy(sc4_model, SocialPolicy.LOCKDOWN_ELEMENTARY_SCHOOL, 30))
sc4_model.add_listener(RemovePolicy(sc4_model, SocialPolicy.LOCKDOWN_MIDDLE_SCHOOL, 60))
sc4_model.add_listener(RemovePolicy(sc4_model, SocialPolicy.LOCKDOWN_HIGH_SCHOOL, 90))
scenario[sc]['parameters'] = sc4_parameters
scenario[sc]['model'] = sc4_model

# ------------------------------------------------------------------------------

sc = 5  # Like scenario 4 but simulate the start of a public campaing in day 30
# to reinforce the importance of social distancing and consequently reduce
# the overall risk tolerance of the population
print(f"Setting up scenario {sc}")

scenario[sc] = {}
sc5_parameters = copy.deepcopy(common_parameters)
sc5_parameters.params['social_policies'] = [
    SocialPolicy.LOCKDOWN_OFFICE,
    SocialPolicy.LOCKDOWN_FACTORY,
    SocialPolicy.LOCKDOWN_RETAIL,
    SocialPolicy.LOCKDOWN_ELEMENTARY_SCHOOL,
    SocialPolicy.LOCKDOWN_MIDDLE_SCHOOL,
    SocialPolicy.LOCKDOWN_HIGH_SCHOOL,
    SocialPolicy.SOCIAL_DISTANCING
]
set_parameters(sc5_parameters)
sc5_model = CovidModel()
np.random.seed(seed)
setup_city_layout(sc5_model, population_size)
sc5_model.add_listener(Propaganda(sc5_model, 30))
sc5_model.add_listener(RemovePolicy(sc5_model, SocialPolicy.LOCKDOWN_ELEMENTARY_SCHOOL, 30))
sc5_model.add_listener(RemovePolicy(sc5_model, SocialPolicy.LOCKDOWN_MIDDLE_SCHOOL, 60))
sc5_model.add_listener(RemovePolicy(sc5_model, SocialPolicy.LOCKDOWN_HIGH_SCHOOL, 90))
scenario[sc]['parameters'] = sc5_parameters
scenario[sc]['model'] = sc5_model

# ------------------------------------------------------------------------------

sc = 6  # Like scenario 5 but start the campaing in day 1
print(f"Setting up scenario {sc}")

scenario[sc] = {}
sc6_parameters = copy.deepcopy(common_parameters)
sc6_parameters.params['social_policies'] = [
    SocialPolicy.LOCKDOWN_OFFICE,
    SocialPolicy.LOCKDOWN_FACTORY,
    SocialPolicy.LOCKDOWN_RETAIL,
    SocialPolicy.LOCKDOWN_ELEMENTARY_SCHOOL,
    SocialPolicy.LOCKDOWN_MIDDLE_SCHOOL,
    SocialPolicy.LOCKDOWN_HIGH_SCHOOL,
    SocialPolicy.SOCIAL_DISTANCING
]
set_parameters(sc6_parameters)
sc6_model = CovidModel()
np.random.seed(seed)
setup_city_layout(sc6_model, population_size)
sc6_model.add_listener(Propaganda(sc6_model, 1))
sc6_model.add_listener(RemovePolicy(sc6_model, SocialPolicy.LOCKDOWN_ELEMENTARY_SCHOOL, 30))
sc6_model.add_listener(RemovePolicy(sc6_model, SocialPolicy.LOCKDOWN_MIDDLE_SCHOOL, 60))
sc6_model.add_listener(RemovePolicy(sc6_model, SocialPolicy.LOCKDOWN_HIGH_SCHOOL, 90))
scenario[sc]['parameters'] = sc6_parameters
scenario[sc]['model'] = sc6_model

################################################################################
# Simulation of all scenarios

for sc in scenario:
    #for sc in [1, 2]:
    # print("--------------------------------------------------------------------------------")
    print(f"Running scenario {sc}")
    set_parameters(scenario[sc]['parameters'])
    model = scenario[sc]['model']
    debug = DebugUtils(model)
    logger().model = model
    model.reset_randomizer(seed)
    statistics = BasicStatistics(model)
    model.add_listener(statistics)
    for i in range(simulation_cycles):
        model.step()
    statistics.export_chart("scenario" + str(sc) + ".png")
    statistics.export_csv("scenario" + str(sc) + ".csv")
    #debug.print_infection_status()
