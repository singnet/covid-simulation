import math
import matplotlib.pyplot as plt
import pandas as pd
from model.base import CovidModel, get_parameters, change_parameters, flip_coin, normal_cap, logger
from model.human import Human, Elder, Adult, K12Student, Toddler, Infant
from model.location import Location, District, HomogeneousBuilding, BuildingUnit, Restaurant
from model.instantiation import FamilyFactory, HomophilyRelationshipFactory
from model.utils import SocialPolicy, TribeSelector, RestaurantType

from scipy.stats import sem, t
from scipy import mean
import random
import math
import numpy as np

from model.base import SimulationParameters, set_parameters, normal_ci


def confidence_interval(data, confidence=0.95):
    n = len(data)
    m = mean(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)

    start = m - h
    end = m + h
    return (start,end)
	

def multiple_runs(params,population_size,simulation_cycles,num_runs = 5,seeds=None,debug=False,
                  desired_stats = ["susceptible","infected","recovered","hospitalization", "icu", "death", "income"],
                 do_print = False):

    randomlist = random.sample(range(10000), num_runs) if seeds is None else seeds
    if (do_print):
        print("Save these seeds if you want to rerun a scenario")
        print(randomlist)

    all_runs = {}
    avg = {}
    last = {}
    peak ={}
    for stat in desired_stats:
        all_runs[stat]= {}
        avg[stat]=[]
        last[stat]=[]
        peak[stat]=[]
    for s in randomlist:
        set_parameters(params)
        model = CovidModel(debug=debug)
        logger().model = model   
        if debug:
            model.debug_each_n_cycles = 20
        np.random.seed(s+1)
        random.seed(s+2)
        setup_city_layout(model, population_size)
        if (do_print):
            print("run with seed {0}:".format(str(s)))
        model.reset_randomizer(s)
        statistics = BasicStatistics(model)
        model.add_listener(statistics)
        for i in range(simulation_cycles):
            model.step()
        for stat in desired_stats:
            all_runs[stat][s]= getattr(statistics,stat)
            avg[stat].append(np.mean(all_runs[stat][s]))
            last[stat].append(all_runs[stat][s][-1])
            peak[stat].append(max(all_runs[stat][s]))
    if (do_print):
        for stat,x in avg.items():
            print("using average of time series:")
            print("stats on {}:".format(stat))
            print("data: {}".format(x))
            print ("min:")
            print(np.min(x))
            print("max:")
            print(np.max(x))
            print("std:")
            print(np.std(x))
            print("mean:")
            print(np.mean(x))
            print("median:")
            print(np.median(x))
            print ("95% confidence interval for the mean:")
            low,high = confidence_interval(x, confidence=0.95)
            print ("({0},{1})".format(low,high))
        for stat,x in last.items():
            print("using last of time series:")
            print("stats on {}:".format(stat))
            print("data: {}".format(x))
            print ("min:")
            print(np.min(x))
            print("max:")
            print(np.max(x))
            print("std:")
            print(np.std(x))
            print("mean:")
            print(np.mean(x))
            print("median:")
            print(np.median(x))
            print ("95% confidence interval for the mean:")
            low,high = confidence_interval(x, confidence=0.95)
            print ("({0},{1})".format(low,high))
        for stat,x in peak.items():
            print("using peak of time series:")
            print("stats on {}:".format(stat))
            print("data: {}".format(x))
            print ("min:")
            print(np.min(x))
            print("max:")
            print(np.max(x))
            print("std:")
            print(np.std(x))
            print("mean:")
            print(np.mean(x))
            print("median:")
            print(np.median(x))
            print ("95% confidence interval for the mean:")
            low,high = confidence_interval(x, confidence=0.95)
            print ("({0},{1})".format(low,high))
    return (avg.items,last.items,peak.items)



class BasicStatistics():
    def __init__(self, model):
        self.susceptible = []
        self.infected = []
        self.recovered = []
        self.hospitalization = []
        self.icu = []
        self.death = []
        self.income = [1.0]
        self.cycles_count = 0
        self.covid_model = model

    def start_cycle(self, model):
        self.cycles_count += 1
        pop = self.covid_model.global_count.total_population
        work_pop = self.covid_model.global_count.work_population
        #print(f"infected = {self.covid_model.global_count.infected_count} recovered = {self.covid_model.global_count.recovered_count}")
        self.susceptible.append(self.covid_model.global_count.susceptible_count / pop)
        self.infected.append(self.covid_model.global_count.infected_count / pop)
        self.recovered.append(self.covid_model.global_count.recovered_count / pop)
        self.hospitalization.append((self.covid_model.global_count.total_hospitalized) / pop)
        self.icu.append(self.covid_model.global_count.high_severity_count / pop)
        self.death.append(self.covid_model.global_count.death_count / pop)
        self.income.append(self.covid_model.global_count.total_income / work_pop)

    def end_cycle(self, model):
        pass

    def export_chart(self, fname):
        self.income.pop(1)
        df = pd.DataFrame(data={
            'Susceptible': self.susceptible,
            'Infected': self.infected,
            'Recovered': self.recovered,
            'Death': self.death,
            'Hospitalization': self.hospitalization,
            'Severe': self.icu,
            'Income': self.income
        })
        color = {
            'Susceptible' : 'lightblue',
            'Infected': 'gray',
            'Recovered': 'lightgreen',
            'Death': 'black',
            'Hospitalization': 'orange',
            'Severe': 'red',
            'Income': 'magenta'
        }
        fig, ax = plt.subplots()
        ax.set_title('Contagion Evolution')
        ax.set_xlim((0, self.cycles_count))
        ax.axhline(y=get_parameters().get('hospitalization_capacity'), c="black", ls='--', label='Critical limit')
        for col in df.columns.values:
            ax.plot(df.index.values, df[col].values, c=color[col], label=col)
        ax.set_xlabel("Days")
        ax.set_ylabel("% of Population")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper right')
        fig.savefig(fname)

    def export_csv(self, fname):
        df = pd.DataFrame(data={
            'Susceptible': self.susceptible,
            'Infected': self.infected,
            'Recovered': self.recovered,
            'Death': self.death,
            'Hospitalization': self.hospitalization,
            'Severe': self.icu,
            'Income': self.income
        })
        df.to_csv(fname)

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
        logger().debug(f'Global risk_tolerance change to {v}')
        change_parameters(risk_tolerance_mean = v)
        self.model.reroll_human_properties()
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

def build_district(name, model, population_size, building_capacity, unit_capacity,
                   occupacy_rate, contagion_probability):

    logger().info(f"Building district {name} contagion_probability = {contagion_probability}")
    district = District(name, model, '', name)
    building_count = math.ceil(
        math.ceil(population_size / unit_capacity) * (1 / occupacy_rate) 
        / building_capacity)
    for i in range(building_count):
        building = HomogeneousBuilding(building_capacity, model, name, str(i))
        for j in range(building_capacity):
            unit = BuildingUnit(unit_capacity, model, name, str(i) + '-' + str(j),
                                contagion_probability=contagion_probability)
            building.locations.append(unit)
        district.locations.append(building)
    return district

def setup_city_layout(model, population_size):

    work_building_capacity = 20
    office_capacity = 10
    work_building_occupacy_rate = 0.5
    appartment_building_capacity = 20
    appartment_capacity = 5
    appartment_building_occupacy_rate = 0.5
    school_capacity = 50
    classroom_capacity = 20
    school_occupacy_rate = 0.5

    # Build empty districts
    # https://docs.google.com/document/d/1imCNXOyoyecfD_sVNmKpmbWVB6xqP-FWlHELAyOg1Vs/edit
    home_district = build_district("Home", model, population_size,
                                   appartment_building_capacity, 
                                   appartment_capacity,
                                   appartment_building_occupacy_rate, 
                                   normal_ci(0.021, 0.12, 10))
    work_district = build_district("Work", model, population_size,
                                   work_building_capacity, 
                                   office_capacity, 
                                   work_building_occupacy_rate, 
                                   normal_ci(0.007, 0.06, 10))
    school_district = build_district("School", model, population_size,
                                     school_capacity, 
                                     classroom_capacity, 
                                     school_occupacy_rate, 
                                     normal_ci(0.014, 0.08, 10))

    home_district.debug = model.debug
    work_district.debug = model.debug
    school_district.debug = model.debug

    # Add Restaurants to work_district

    for i in range(10):
        if flip_coin(0.5):
            restaurant_type = RestaurantType.FAST_FOOD
            rtype = "FASTFOOD"
        else:
            restaurant_type = RestaurantType.FANCY
            rtype = "FANCY"
        restaurant = Restaurant(
            normal_cap(50, 20, 16, 100), 
            restaurant_type, 
            flip_coin(0.5), 
            model, 
            '',
            rtype + '-' + str(i))
        work_district.locations.append(restaurant)
    for i in range(2):
        bar = Restaurant(
            normal_cap(100, 20, 50, 200), 
            RestaurantType.BAR, 
            flip_coin(0.5), 
            model, 
            '', 
            'BAR-' + str(i))
        work_district.locations.append(bar)

    #print(home_district)
    #print(work_district)
    #print(school_district)

    # Build families

    family_factory = FamilyFactory(model)
    family_factory.factory(population_size)
    model.global_count.total_population = family_factory.human_count

    #print(family_factory)

    age_group_sets = {
        Infant: [],
        Toddler: [],
        K12Student: [],
        Adult: [],
        Elder: []
    }

    # Allocate buildings to people

    all_adults = []
    all_students = []
    for family in family_factory.families:
        adults = [human for human in family if isinstance(human, Adult)]
        students = [human for human in family if isinstance(human, K12Student)]
        home_district.allocate(family, True, True, True)
        work_district.allocate(adults)
        school_district.allocate(students, True)
        for human in family:
            age_group_sets[type(human)].append(human)
            human.home_district = home_district
            home_district.get_buildings(human)[0].get_unit(human).humans.append(human)
        for adult in adults:
            adult.work_district = work_district
            all_adults.append(adult)
        for student in students:
            student.school_district = school_district
            all_students.append(student)

    # Set tribes

    adult_rf = HomophilyRelationshipFactory(model, all_adults)
    student_rf = HomophilyRelationshipFactory(model, all_students)
    #exit()


    count = 0
    for family in family_factory.families:
        for human in family:
            count += 1
            human.tribe[TribeSelector.AGE_GROUP] = age_group_sets[type(human)]
            human.tribe[TribeSelector.FAMILY] = family
            if isinstance(human, Adult):
                human.unique_id = "Adult"+str(count)
                human.tribe[TribeSelector.COWORKER] = work_district.get_buildings(human)[0].get_unit(human).allocation
                t1 = adult_rf.build_tribe(human, human.tribe[TribeSelector.COWORKER], 1, office_capacity)
                t2 = adult_rf.build_tribe(human, human.tribe[TribeSelector.AGE_GROUP], 1, 20)
                human.tribe[TribeSelector.FRIEND] = t1
                for h in t2:
                    if h not in human.tribe[TribeSelector.FRIEND]:
                        human.tribe[TribeSelector.FRIEND].append(h)
            elif isinstance(human, K12Student):
                human.unique_id = "K12Student"+str(count)
                human.tribe[TribeSelector.CLASSMATE] = school_district.get_buildings(human)[0].get_unit(human).allocation
                t1 = student_rf.build_tribe(human, human.tribe[TribeSelector.CLASSMATE], 1, classroom_capacity)
                t2 = student_rf.build_tribe(human, human.tribe[TribeSelector.AGE_GROUP], 1, 20)
                human.tribe[TribeSelector.FRIEND] = t1
                for h in t2:
                    if h not in human.tribe[TribeSelector.FRIEND]:
                        human.tribe[TribeSelector.FRIEND].append(h)
            elif isinstance(human, Elder):
                human.unique_id = "Elder"+str(count)
            elif isinstance(human, Infant):
                human.unique_id = "Infant"+str(count)
            elif isinstance(human, Toddler):
                human.unique_id = "Toddler"+str(count)

        
    #print(home_district)
    #print(work_district)
    #print(school_district)

    #exit()
