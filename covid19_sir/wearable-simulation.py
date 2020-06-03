from model import CovidModel, PeopleGroup, SimulationParameters, set_parameters
from utils import SimpleGroup, BasicStatistics

################################################################################
# Common parameters amongst all scenarios

# COVID model

mask_user_rate = 0.0
mask_efficacy = 0.0
imune_rate = 0.01
initial_infection_rate = 0.01
hospitalization_capacity = 0.02
latency_period_mean = 3.0
latency_period_stdev = 1.0
incubation_period_mean = 7.0
incubation_period_stdev = 4.0
disease_period_mean = 20
disease_period_stdev = 5

# Population group

daily_interaction_count = 4
contagion_probability = 0.2
asymptomatic_isolation_rate = 0.0
symptomatic_isolation_rate = 0.0

# Simulation

population_size = 1000
simulation_cycles = 90 # days

################################################################################
# Scenarios

scenario = {}

# ------------------------------------------------------------------------------

sc = 1 # Do nothing
scenario[sc] = {}
scenario[sc]['parameters'] = SimulationParameters(
    mask_user_rate = mask_user_rate,
    mask_efficacy = mask_efficacy,
    imune_rate = imune_rate,
    initial_infection_rate = initial_infection_rate,
    hospitalization_capacity = hospitalization_capacity,
    latency_period_mean = latency_period_mean,
    latency_period_stdev = latency_period_stdev,
    incubation_period_mean = incubation_period_mean,
    incubation_period_stdev = incubation_period_stdev,
    disease_period_mean = disease_period_mean,
    disease_period_stdev = disease_period_stdev
)
set_parameters(scenario[sc]['parameters'])
scenario[sc]['model'] = CovidModel()
scenario[sc]['group'] = SimpleGroup(0, scenario[sc]['model'], population_size,
    daily_interaction_count = daily_interaction_count,
    contagion_probability = contagion_probability,
    asymptomatic_isolation_rate = asymptomatic_isolation_rate,
    symptomatic_isolation_rate = symptomatic_isolation_rate
)

# ------------------------------------------------------------------------------

sc = 2 # Restrict the mobility only for infected people - no weareables
scenario[sc] = {}
scenario[sc]['parameters'] = SimulationParameters(
    weareable_adoption_rate = 0.0,
    mask_user_rate = mask_user_rate,
    mask_efficacy = mask_efficacy,
    imune_rate = imune_rate,
    initial_infection_rate = initial_infection_rate,
    hospitalization_capacity = hospitalization_capacity,
    latency_period_mean = latency_period_mean,
    latency_period_stdev = latency_period_stdev,
    incubation_period_mean = incubation_period_mean,
    incubation_period_stdev = incubation_period_stdev,
    disease_period_mean = disease_period_mean,
    disease_period_stdev = disease_period_stdev
)
set_parameters(scenario[sc]['parameters'])
scenario[sc]['model'] = CovidModel()
scenario[sc]['group'] = SimpleGroup(0, scenario[sc]['model'], population_size,
    symptomatic_isolation_rate = 0.9,
    asymptomatic_isolation_rate = asymptomatic_isolation_rate,
    daily_interaction_count = daily_interaction_count,
    contagion_probability = contagion_probability
)

# ------------------------------------------------------------------------------

sc = 3 # restrict the mobility for infected people  - some people use wearebles 
       # which allows earlier detection of symptoms
scenario[sc] = {}
scenario[sc]['parameters'] = SimulationParameters(
    weareable_adoption_rate = 0.3,
    mask_user_rate = mask_user_rate,
    mask_efficacy = mask_efficacy,
    imune_rate = imune_rate,
    initial_infection_rate = initial_infection_rate,
    hospitalization_capacity = hospitalization_capacity,
    latency_period_mean = latency_period_mean,
    latency_period_stdev = latency_period_stdev,
    incubation_period_mean = incubation_period_mean,
    incubation_period_stdev = incubation_period_stdev,
    disease_period_mean = disease_period_mean,
    disease_period_stdev = disease_period_stdev
)
set_parameters(scenario[sc]['parameters'])
scenario[sc]['model'] = CovidModel()
scenario[sc]['group'] = SimpleGroup(0, scenario[sc]['model'], population_size,
    symptomatic_isolation_rate = 0.9,
    asymptomatic_isolation_rate = 0.8,
    daily_interaction_count = daily_interaction_count,
    contagion_probability = contagion_probability
)

################################################################################
# Simulation of all scenarios

for sc in scenario:
    set_parameters(scenario[sc]['parameters'])
    model = scenario[sc]['model']
    group = scenario[sc]['group']
    statistics = BasicStatistics(model)
    model.add_group(group)
    model.add_listener(statistics)
    for i in range(simulation_cycles):
        model.step()
    statistics.export_chart("wearable_scenario" + str(sc) + ".png")
    statistics.export_csv("wearable_scenario" + str(sc) + ".csv")
