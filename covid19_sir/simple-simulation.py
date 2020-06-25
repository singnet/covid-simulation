import sys
import math
import numpy as np
from model.base import CovidModel, SimulationParameters, SocialPolicy, TribeSelector, set_parameters, get_parameters, change_parameters
from model.human import Human, Elder, Adult, K12Student, Toddler, Infant
from model.location import District, HomogeneousBuilding, BuildingUnit, FunGatheringSpot
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
symptomatic_isolation_rate = 0.9
asymptomatic_contagion_probability = 0.1
risk_tolerance_mean = 0.8
risk_tolerance_stdev = 0.2
herding_behavior_mean = 0.7
herding_behavior_stdev = 0.2

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
        district.locations.append(FunGatheringSpot(10, model))
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
    work_building_occupacy_rate = 0.5
    appartment_building_capacity = 20
    appartment_capacity = 5
    appartment_building_occupacy_rate = 0.5
    school_capacity = 50
    classroom_capacity = 30
    school_occupacy_rate = 0.5

    # Build empty districts
    home_district = build_district("Home", model, 
                                   appartment_building_capacity, 
                                   appartment_capacity,
                                   appartment_building_occupacy_rate, 
                                   0.9, 0.9)
    work_district = build_district("Work", model, 
                                   work_building_capacity, 
                                   office_capacity, 
                                   work_building_occupacy_rate, 
                                   0.5, 0.6)
    school_district = build_district("School", model, 
                                     school_capacity, 
                                     classroom_capacity, 
                                     school_occupacy_rate, 
                                     0.5, 0.9)
    #print(home_district)
    #print(work_district)
    #print(school_district)

    # Build families

    family_factory = FamilyFactory(model)
    family_factory.factory(population_size)
    model.global_count.total_population = family_factory.human_count

    #print(family_factory)

    age_class_tribes = {
        Infant: [],
        Toddler: [],
        K12Student: [],
        Adult: [],
        Elder: []
    }

    # Allocate buildings to people

    for family in family_factory.families:
        adults = [human for human in family if isinstance(human, Adult)]
        students = [human for human in family if isinstance(human, K12Student)]
        home_district.allocate(family, True, True, True)
        work_district.allocate(adults)
        school_district.allocate(students, True)
        for human in family:
            age_class_tribes[type(human)].append(human)
            human.home_district = home_district
            home_district.get_buildings(human)[0].get_unit(human).humans.append(human)
        for adult in adults:
            adult.work_district = work_district
        for student in students:
            student.school_district = school_district

    # Set tribes

    for family in family_factory.families:
        for human in family:
            human.tribe[TribeSelector.AGE_CLASS] = age_class_tribes[type(human)]
            human.tribe[TribeSelector.FAMILY] = family
            if isinstance(human, Adult):
                human.tribe[TribeSelector.COWORKER] = work_district.get_buildings(human)[0].get_unit(human).allocation
            if isinstance(human, K12Student):
                human.tribe[TribeSelector.CLASSMATE] = school_district.get_buildings(human)[0].get_unit(human).allocation
        
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
    symptomatic_isolation_rate = 0.0,
    asymptomatic_contagion_probability = asymptomatic_contagion_probability,
    herding_behavior_mean = herding_behavior_mean,
    herding_behavior_stdev = herding_behavior_stdev,
    risk_tolerance_mean = risk_tolerance_mean,
    risk_tolerance_stdev = risk_tolerance_stdev
)
set_parameters(scenario[sc]['parameters'])
scenario[sc]['model'] = CovidModel()
if len(sys.argv) > 1:
    seed = int(sys.argv[1])
    np.random.seed(seed)
setup_city_layout(scenario[sc]['model'])

# ------------------------------------------------------------------------------

sc = 2 # complete lockdown
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
    asymptomatic_contagion_probability = asymptomatic_contagion_probability,
    herding_behavior_mean = herding_behavior_mean,
    herding_behavior_stdev = herding_behavior_stdev,
    risk_tolerance_mean = risk_tolerance_mean,
    risk_tolerance_stdev = risk_tolerance_stdev,
    symptomatic_isolation_rate = symptomatic_isolation_rate,
    social_policies = [
        SocialPolicy.LOCKDOWN_OFFICE,
        SocialPolicy.LOCKDOWN_FACTORY,
        SocialPolicy.LOCKDOWN_RETAIL,
        SocialPolicy.LOCKDOWN_ELEMENTARY_SCHOOL,
        SocialPolicy.LOCKDOWN_MIDDLE_SCHOOL,
        SocialPolicy.LOCKDOWN_HIGH_SCHOOL,
        SocialPolicy.SOCIAL_DISTANCING
    ]
)
set_parameters(scenario[sc]['parameters'])
scenario[sc]['model'] = CovidModel()
if len(sys.argv) > 1:
    seed = int(sys.argv[1])
    np.random.seed(seed)
setup_city_layout(scenario[sc]['model'])

# ------------------------------------------------------------------------------

sc = 3 # all schools open
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
    asymptomatic_contagion_probability = asymptomatic_contagion_probability,
    herding_behavior_mean = herding_behavior_mean,
    herding_behavior_stdev = herding_behavior_stdev,
    risk_tolerance_stdev = risk_tolerance_stdev,
    symptomatic_isolation_rate = symptomatic_isolation_rate,
    risk_tolerance_mean = risk_tolerance_mean,
    social_policies = [
        SocialPolicy.LOCKDOWN_OFFICE,
        SocialPolicy.LOCKDOWN_FACTORY,
        SocialPolicy.LOCKDOWN_RETAIL,
        #SocialPolicy.LOCKDOWN_ELEMENTARY_SCHOOL,
        #SocialPolicy.LOCKDOWN_MIDDLE_SCHOOL,
        #SocialPolicy.LOCKDOWN_HIGH_SCHOOL,
        SocialPolicy.SOCIAL_DISTANCING
    ]
)
set_parameters(scenario[sc]['parameters'])
scenario[sc]['model'] = CovidModel()
if len(sys.argv) > 1:
    seed = int(sys.argv[1])
    np.random.seed(seed)
setup_city_layout(scenario[sc]['model'])

# ------------------------------------------------------------------------------

sc = 4 # all schools open + risk tolerance low
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
    asymptomatic_contagion_probability = asymptomatic_contagion_probability,
    herding_behavior_stdev = herding_behavior_stdev,
    risk_tolerance_stdev = risk_tolerance_stdev,
    symptomatic_isolation_rate = symptomatic_isolation_rate,
    herding_behavior_mean = 0.2,
    risk_tolerance_mean = 0.2,
    social_policies = [
        SocialPolicy.LOCKDOWN_OFFICE,
        SocialPolicy.LOCKDOWN_FACTORY,
        SocialPolicy.LOCKDOWN_RETAIL,
        #SocialPolicy.LOCKDOWN_ELEMENTARY_SCHOOL,
        #SocialPolicy.LOCKDOWN_MIDDLE_SCHOOL,
        SocialPolicy.LOCKDOWN_HIGH_SCHOOL,
        SocialPolicy.SOCIAL_DISTANCING
    ]
)
set_parameters(scenario[sc]['parameters'])
scenario[sc]['model'] = CovidModel()
if len(sys.argv) > 1:
    seed = int(sys.argv[1])
    np.random.seed(seed)
setup_city_layout(scenario[sc]['model'])

################################################################################
# Simulation of all scenarios

for sc in scenario:
#for sc in [3]:
    #print("--------------------------------------------------------------------------------")
    set_parameters(scenario[sc]['parameters'])
    model = scenario[sc]['model']
    model.reset_randomizer(seed)
    statistics = BasicStatistics(model)
    model.add_listener(statistics)
    for i in range(simulation_cycles):
        model.step()
    statistics.export_chart("scenario" + str(sc) + ".png")
    statistics.export_csv("scenario" + str(sc) + ".csv")
