import sys
import math
import numpy as np
from model.base import (CovidModel, SimulationParameters, set_parameters, 
get_parameters, change_parameters, random_selection)
from model.utils import SocialPolicy, TribeSelector
from model.human import Human, Elder, Adult, K12Student, Toddler, Infant
from model.location import District, HomogeneousBuilding, BuildingUnit, FunGatheringSpot
from model.instantiation import FamilyFactory
from utils import BasicStatistics

seed = 31415
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
incubation_period_mean = 6.0
incubation_period_stdev = 2.0
disease_period_mean = 20
disease_period_stdev = 5
symptomatic_isolation_rate = 0.9
asymptomatic_contagion_probability = 0.1
risk_tolerance_mean = 0.7
risk_tolerance_stdev = 0.2
herding_behavior_mean = 0.7
herding_behavior_stdev = 0.2

# Simulation

population_size = 1000
simulation_cycles = 180 # days

class RemovePolicy():
    def __init__(self, model, policy, n):
        self.switch = n
        self.policy = policy
        self.model = model
        self.state = 0
    def start_cycle(self, model):
        pass
    def end_cycle(self, model):
        if self.state == 0:
            if self.model.global_count.day_count == self.switch:
                get_parameters().get('social_policies').remove(self.policy)

class Propaganda():
    def __init__(self, model, n):
        self.switch = n
        self.count = 0
        self.model = model
        self.state = 0
    def start_cycle(self, model):
        pass
    def tick(self):
        v = get_parameters().get('risk_tolerance_mean') - 0.1
        if v < 0.1:
            v = 0.1
        change_parameters(risk_tolerance_mean = v)
        model.reroll_human_properties()
    def end_cycle(self, model):
        if self.state == 0:
            if self.model.global_count.day_count == self.switch:
                self.state = 1
                self.count += 1
                self.tick()
        elif self.state == 1:
            if not (self.count % 3):
                self.count += 1
                self.tick

def build_district(name, model, building_capacity, unit_capacity,
                   occupacy_rate, spreading_rate, contagion_probability):

    district = District(name, model)
    building_count = math.ceil(
        math.ceil(population_size / unit_capacity) * (1 / occupacy_rate) 
        / building_capacity)
    for i in range(building_count):
        district.locations.append(FunGatheringSpot(10, model,
                                  spreading_rate=spreading_rate, 
                                  contagion_probability=contagion_probability))
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

    all_adults = []
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
            all_adults.append(adult)
        for student in students:
            student.school_district = school_district

    # Set tribes

    np.random.shuffle(all_adults)
    for family in family_factory.families:
        for human in family:
            human.tribe[TribeSelector.AGE_CLASS] = age_class_tribes[type(human)]
            human.tribe[TribeSelector.FAMILY] = family
            if isinstance(human, Adult):
                human.tribe[TribeSelector.COWORKER] = work_district.get_buildings(human)[0].get_unit(human).allocation
                friend_tribe_size = 20
                for h in all_adults:
                    if h not in human.tribe[TribeSelector.FAMILY] and \
                       len(human.tribe[TribeSelector.FRIEND]) < friend_tribe_size and \
                       len(h.tribe[TribeSelector.FRIEND]) < friend_tribe_size:
                        human.tribe[TribeSelector.FRIEND].append(h)
                        h.tribe[TribeSelector.FRIEND].append(human)
            if isinstance(human, K12Student):
                human.tribe[TribeSelector.CLASSMATE] = school_district.get_buildings(human)[0].get_unit(human).allocation
    for human in all_adults:
        human.tribe[TribeSelector.FRIEND].append(human)
        
    #print(home_district)
    #print(work_district)
    #print(school_district)

    #exit()

################################################################################
# Scenarios

scenario = {}

# ------------------------------------------------------------------------------

sc = 1 # Do nothing
#print(f"Setting up scenario {sc}")
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
np.random.seed(seed)
setup_city_layout(scenario[sc]['model'])

# ------------------------------------------------------------------------------

sc = 2 # complete lockdown
#print(f"Setting up scenario {sc}")
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
np.random.seed(seed)
setup_city_layout(scenario[sc]['model'])

# ------------------------------------------------------------------------------

sc = 3 
#print(f"Setting up scenario {sc}")

scenario[sc] = {}
sc3_parameters = SimulationParameters(
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
set_parameters(sc3_parameters)
sc3_model = CovidModel()
np.random.seed(seed)
setup_city_layout(sc3_model)
sc3_model.add_listener(RemovePolicy(sc3_model, SocialPolicy.LOCKDOWN_ELEMENTARY_SCHOOL, 30))
sc3_model.add_listener(RemovePolicy(sc3_model, SocialPolicy.LOCKDOWN_MIDDLE_SCHOOL, 60))
sc3_model.add_listener(RemovePolicy(sc3_model, SocialPolicy.LOCKDOWN_HIGH_SCHOOL, 90))
scenario[sc]['parameters'] = sc3_parameters
scenario[sc]['model'] = sc3_model


# ------------------------------------------------------------------------------

sc = 4 
#print(f"Setting up scenario {sc}")

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
set_parameters(sc4_parameters)
sc4_model = CovidModel()
np.random.seed(seed)
setup_city_layout(sc4_model)
sc4_model.add_listener(Propaganda(sc4_model, 30))
sc4_model.add_listener(RemovePolicy(sc4_model, SocialPolicy.LOCKDOWN_ELEMENTARY_SCHOOL, 30))
sc4_model.add_listener(RemovePolicy(sc4_model, SocialPolicy.LOCKDOWN_MIDDLE_SCHOOL, 60))
sc4_model.add_listener(RemovePolicy(sc4_model, SocialPolicy.LOCKDOWN_HIGH_SCHOOL, 90))
scenario[sc]['parameters'] = sc4_parameters
scenario[sc]['model'] = sc4_model

# ------------------------------------------------------------------------------

sc = 5 
#print(f"Setting up scenario {sc}")

scenario[sc] = {}
sc5_parameters = SimulationParameters(
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
set_parameters(sc5_parameters)
sc5_model = CovidModel()
np.random.seed(seed)
setup_city_layout(sc5_model)
sc5_model.add_listener(Propaganda(sc5_model, 1))
sc5_model.add_listener(RemovePolicy(sc5_model, SocialPolicy.LOCKDOWN_ELEMENTARY_SCHOOL, 30))
sc5_model.add_listener(RemovePolicy(sc5_model, SocialPolicy.LOCKDOWN_MIDDLE_SCHOOL, 60))
sc5_model.add_listener(RemovePolicy(sc5_model, SocialPolicy.LOCKDOWN_HIGH_SCHOOL, 90))
scenario[sc]['parameters'] = sc5_parameters
scenario[sc]['model'] = sc5_model


# ------------------------------------------------------------------------------

sc = 6 
#print(f"Setting up scenario {sc}")

scenario[sc] = {}
sc6_parameters = SimulationParameters(
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
set_parameters(sc6_parameters)
sc6_model = CovidModel()
np.random.seed(seed)
setup_city_layout(sc6_model)
sc6_model.add_listener(RemovePolicy(sc6_model, SocialPolicy.LOCKDOWN_RETAIL, 30))
sc6_model.add_listener(RemovePolicy(sc6_model, SocialPolicy.LOCKDOWN_FACTORY, 60))
sc6_model.add_listener(RemovePolicy(sc6_model, SocialPolicy.LOCKDOWN_OFFICE, 90))
sc6_model.add_listener(RemovePolicy(sc6_model, SocialPolicy.LOCKDOWN_ELEMENTARY_SCHOOL, 90))
sc6_model.add_listener(RemovePolicy(sc6_model, SocialPolicy.LOCKDOWN_MIDDLE_SCHOOL, 90))
sc6_model.add_listener(RemovePolicy(sc6_model, SocialPolicy.LOCKDOWN_HIGH_SCHOOL, 120))
scenario[sc]['parameters'] = sc6_parameters
scenario[sc]['model'] = sc6_model

# ------------------------------------------------------------------------------

sc = 7 
#print(f"Setting up scenario {sc}")

scenario[sc] = {}
sc7_parameters = SimulationParameters(
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
set_parameters(sc7_parameters)
sc7_model = CovidModel()
np.random.seed(seed)
setup_city_layout(sc7_model)
sc7_model.add_listener(Propaganda(sc7_model, 30))
sc7_model.add_listener(RemovePolicy(sc7_model, SocialPolicy.LOCKDOWN_RETAIL, 30))
sc7_model.add_listener(RemovePolicy(sc7_model, SocialPolicy.LOCKDOWN_FACTORY, 60))
sc7_model.add_listener(RemovePolicy(sc7_model, SocialPolicy.LOCKDOWN_OFFICE, 90))
sc7_model.add_listener(RemovePolicy(sc7_model, SocialPolicy.LOCKDOWN_ELEMENTARY_SCHOOL, 90))
sc7_model.add_listener(RemovePolicy(sc7_model, SocialPolicy.LOCKDOWN_MIDDLE_SCHOOL, 90))
sc7_model.add_listener(RemovePolicy(sc7_model, SocialPolicy.LOCKDOWN_HIGH_SCHOOL, 120))
scenario[sc]['parameters'] = sc7_parameters
scenario[sc]['model'] = sc7_model

# ------------------------------------------------------------------------------

sc = 8 
#print(f"Setting up scenario {sc}")

scenario[sc] = {}
sc8_parameters = SimulationParameters(
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
set_parameters(sc8_parameters)
sc8_model = CovidModel()
np.random.seed(seed)
setup_city_layout(sc8_model)
sc8_model.add_listener(Propaganda(sc8_model, 1))
sc8_model.add_listener(RemovePolicy(sc8_model, SocialPolicy.LOCKDOWN_RETAIL, 30))
sc8_model.add_listener(RemovePolicy(sc8_model, SocialPolicy.LOCKDOWN_FACTORY, 60))
sc8_model.add_listener(RemovePolicy(sc8_model, SocialPolicy.LOCKDOWN_OFFICE, 90))
sc8_model.add_listener(RemovePolicy(sc8_model, SocialPolicy.LOCKDOWN_ELEMENTARY_SCHOOL, 90))
sc8_model.add_listener(RemovePolicy(sc8_model, SocialPolicy.LOCKDOWN_MIDDLE_SCHOOL, 90))
sc8_model.add_listener(RemovePolicy(sc8_model, SocialPolicy.LOCKDOWN_HIGH_SCHOOL, 120))
scenario[sc]['parameters'] = sc8_parameters
scenario[sc]['model'] = sc8_model


################################################################################
# Simulation of all scenarios

for sc in scenario:
#for sc in [1, 2, 3, 4, 5, 6, 7, 8]:
    #print("--------------------------------------------------------------------------------")
    #print(f"Running scenario {sc}")
    set_parameters(scenario[sc]['parameters'])
    model = scenario[sc]['model']
    model.reset_randomizer(seed)
    statistics = BasicStatistics(model)
    model.add_listener(statistics)
    for i in range(simulation_cycles):
        model.step()
    statistics.export_chart("scenario" + str(sc) + ".png")
    statistics.export_csv("scenario" + str(sc) + ".csv")
