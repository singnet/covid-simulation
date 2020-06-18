import sys
import math
import numpy as np
from model.base import CovidModel, SimulationParameters, set_parameters, get_parameters, change_parameters
from model.human import Human, Adult, K12Student
from model.location import District, HomogeneousBuilding, BuildingUnit
from model.instantiation import FamilyFactory
from utils import BasicStatistics

if len(sys.argv) > 1:
    seed = int(sys.argv[1])
    np.random.seed(seed)

################################################################################
# Common parameters amongst all scenarios

# COVID model

mask_user_rate = 0.0
mask_efficacy = 0.0
imune_rate = 0.01
initial_infection_rate = 0.01
hospitalization_capacity = 0.05
latency_period_mean = 3.0
latency_period_stdev = 1.0
incubation_period_mean = 3.0
incubation_period_stdev = 1.0
disease_period_mean = 20
disease_period_stdev = 5
asymptomatic_isolation_rate = 0.0
symptomatic_isolation_rate = 0.0
asymptomatic_contagion_probability = 0.1

# Simulation

population_size = 1000
simulation_cycles = 90 # days

def build_district(name, model, building_capacity, unit_capacity,
                   occupacy_rate, spreading_rate, contagion_probability):

    district = District(name, model)
    building_count = math.ceil(
        math.ceil(population_size / unit_capacity) * (1 / occupacy_rate) 
        / building_capacity)
    for i in range(building_count):
        building = HomogeneousBuilding(building_capacity, model)
        for j in range(building_capacity):
            unit = BuildingUnit(unit_capacity, model, 
                                spreading_rate=spreading_rate, 
                                contagion_probability=contagion_probability)
            building.locations.append(unit)
        district.locations.append(building)
    return district

def setup_city_layout(model):

    work_building_capacity = 20
    office_capacity = 10
    work_building_accupacy_rate = 0.5
    appartment_building_capacity = 20
    appartment_capacity = 5
    appartment_building_accupacy_rate = 0.5
    school_capacity = 50
    classroom_capacity = 30
    school_accupacy_rate = 0.5

    home_district = build_district("Home", model, 
                                   appartment_building_capacity, 
                                   appartment_capacity,
                                   appartment_building_accupacy_rate, 
                                   0.9, 0.9)
    work_district = build_district("Work", model, 
                                   work_building_capacity, 
                                   office_capacity, 
                                   work_building_accupacy_rate, 
                                   0.5, 0.6)
    school_district = build_district("School", model, 
                                     school_capacity, 
                                     classroom_capacity, 
                                     school_accupacy_rate, 
                                     0.5, 0.9)
    #print(home_district)
    #print(work_district)
    #print(school_district)

    family_factory = FamilyFactory(model)
    family_factory.factory(population_size)
    model.global_count.total_population = family_factory.human_count

    #print(family_factory)

    for family in family_factory.families:
        adults = [human for human in family if isinstance(human, Adult)]
        students = [human for human in family if isinstance(human, K12Student)]
        home_district.allocate(family, True, True, True)
        work_district.allocate(adults)
        school_district.allocate(students, True)
        for human in family:
            human.home_district = home_district
            home_district.get_buildings(human)[0].get_unit(human).humans.append(human)
        for adult in adults:
            adult.work_district = work_district
        for student in students:
            student.school_district = school_district

    #print(home_district)
    #print(work_district)
    #print(school_district)

    #exit()

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
    disease_period_stdev = disease_period_stdev,
    asymptomatic_isolation_rate = asymptomatic_isolation_rate,
    symptomatic_isolation_rate = symptomatic_isolation_rate,
    asymptomatic_contagion_probability = asymptomatic_contagion_probability
)
set_parameters(scenario[sc]['parameters'])
scenario[sc]['model'] = CovidModel()
if len(sys.argv) > 1:
    seed = int(sys.argv[1])
    np.random.seed(seed)
setup_city_layout(scenario[sc]['model'])

# ------------------------------------------------------------------------------

sc = 2 # Restrict the mobility only for infected people
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
    disease_period_stdev = disease_period_stdev,
    symptomatic_isolation_rate = 0.9,
    asymptomatic_isolation_rate = asymptomatic_isolation_rate,
    asymptomatic_contagion_probability = asymptomatic_contagion_probability
)
set_parameters(scenario[sc]['parameters'])
scenario[sc]['model'] = CovidModel()
if len(sys.argv) > 1:
    seed = int(sys.argv[1])
    np.random.seed(seed)
setup_city_layout(scenario[sc]['model'])

# ------------------------------------------------------------------------------

sc = 3 # restrict the mobility for everybody
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
    disease_period_stdev = disease_period_stdev,
    symptomatic_isolation_rate = 0.9,
    asymptomatic_isolation_rate = 0.8,
    asymptomatic_contagion_probability = asymptomatic_contagion_probability
)
set_parameters(scenario[sc]['parameters'])
scenario[sc]['model'] = CovidModel()
if len(sys.argv) > 1:
    seed = int(sys.argv[1])
    np.random.seed(seed)
setup_city_layout(scenario[sc]['model'])

# ------------------------------------------------------------------------------

sc = 4 # Restrict the mobility after 10% of the population being infected
       # and release the restrictions when more then 95% is safe

       # This scenarios illustrates the use of listeners to change simulation
       # parameters during the simulation based in some dynamic criterion

class IsolationRule():
    def __init__(self, model, perc1, perc2):
        self.perc1 = perc1
        self.perc2 = perc2
        self.model = model
        self.state = 0
    def start_cycle(self, model):
        pass
    def end_cycle(self, model):
        if self.state == 0:
            if (self.model.global_count.infected_count / model.global_count.total_population) >= self.perc1:
                self.state = 1
                change_parameters(symptomatic_isolation_rate = 0.9,
                                  asymptomatic_isolation_rate = 0.8)
        elif self.state == 1:
            if (self.model.global_count.infected_count / model.global_count.total_population) <= (1.0 - self.perc2):
                self.state = 2
                change_parameters(symptomatic_isolation_rate = 0.0,
                                  asymptomatic_isolation_rate = 0.0)

scenario[sc] = {}
sc4_parameters = SimulationParameters(
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
    disease_period_stdev = disease_period_stdev,
    asymptomatic_contagion_probability = asymptomatic_contagion_probability,
    symptomatic_isolation_rate = 0.0,
    asymptomatic_isolation_rate = 0.0
)
set_parameters(sc4_parameters)
sc4_model = CovidModel()
if len(sys.argv) > 1:
    seed = int(sys.argv[1])
    np.random.seed(seed)
setup_city_layout(sc4_model)
sc4_model.add_listener(IsolationRule(sc4_model, 0.1, 0.95))
scenario[sc]['parameters'] = sc4_parameters
scenario[sc]['model'] = sc4_model

################################################################################
# Simulation of all scenarios

for sc in scenario:
    set_parameters(scenario[sc]['parameters'])
    model = scenario[sc]['model']
    statistics = BasicStatistics(model)
    model.add_listener(statistics)
    for i in range(simulation_cycles):
        model.step()
    statistics.export_chart("scenario" + str(sc) + ".png")
    statistics.export_csv("scenario" + str(sc) + ".csv")
