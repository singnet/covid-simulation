import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from model.base import CovidModel, get_parameters, change_parameters, flip_coin, normal_cap, logger, ENABLE_WORKER_CLASS_SPECIAL_BUILDINGS
from model.human import Elder, Adult, K12Student, Toddler, Infant, Human
from model.location import District, HomogeneousBuilding, BuildingUnit, Restaurant, Hospital
from model.instantiation import FamilyFactory, HomophilyRelationshipFactory
from model.utils import TribeSelector, RestaurantType,SimulationState
import model.utils
import copy
from scipy.stats import sem, t
import random
import math
import numpy as np
from numpy import mean

from model.base import set_parameters, beta_range


def confidence_interval(data, confidence=0.95):
    n = len(data)
    m = np.mean(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)

    start = m - h
    end = m + h
    return start, m, end


def multiple_runs(params, population_size, simulation_cycles, num_runs=5, seeds=[], debug=False, desired_stats=None,
                  fname="scenario", listeners=[], do_print=False, home_grid_height=1, home_grid_width=1,
                  work_home_list=[[(0,0)]], school_home_list=[[(0,0)]], temperature = -1, zoomed_plot=True,
                  zoomed_plot_ylim=(-0.01, .12)):
    color = {
            'susceptible': 'lightblue',
            'infected': 'gray',
            'recovered': 'lightgreen',
            'death': 'black',
            'hospitalization': 'orange',
            'icu': 'red',
            'income': 'magenta',
            'clumpiness':'purple'
        }
    if desired_stats is None:
        desired_stats = ["susceptible", "infected", "recovered", "hospitalization", "icu", "death", "income","clumpiness"]
    randomlist = random.sample(range(10000), num_runs) if len(seeds) == 0  else seeds
    if do_print:
        print("Save these seeds if you want to rerun a scenario")
        print(randomlist)

    all_runs = {}
    avg = {}
    last = {}
    peak = {}
    average = {}
    lower = {}
    upper = {}
    for stat in desired_stats:
        all_runs[stat] = {}
        avg[stat] = []
        last[stat] = []
        peak[stat] = []
        average[stat]= []
        upper[stat] = []
        lower[stat]= []
    for s in randomlist:
        paramcopy = copy.deepcopy(params)
        set_parameters(paramcopy)
        model = CovidModel(debug=debug)
        l = []
        ls= copy.deepcopy(listeners)
        for listener in ls:
            funct = listener.pop(0)
            listener[:0]=[model]
            l.append(globals()[funct](*listener))
        for k in l:
            model.add_listener(k)
        #for m in model.ls:
            #print(m)
        if debug:
            model.debug_each_n_cycles = 20
        np.random.seed(s + 1)
        random.seed(s + 2)
        model.reset_randomizer(s)
        setup_homophilic_layout(model,population_size,home_grid_height,home_grid_width,work_home_list,school_home_list)
        #setup_grid_layout(model, population_size, home_grid_height, 
        #home_grid_width,work_height,work_width, school_height, school_width)
        if do_print:
            print("run with seed {0}:".format(str(s)))
        statistics = BasicStatistics(model)
        model.add_listener(statistics)
        network = Network(model)
        model.add_listener(network)
        for i in range(simulation_cycles):
            model.step()
        #print("clumpiness:")
        #print (getattr(network,"clumpiness"))
        for stat in desired_stats:
            if stat is "clumpiness":
                all_runs[stat][s]=copy.deepcopy(getattr(network,"clumpiness"))
            else:
                all_runs[stat][s] = getattr(statistics, stat)
            if stat is "income":
                all_runs[stat][s].pop(1)
            avg[stat].append(np.mean(all_runs[stat][s]))
            last[stat].append(all_runs[stat][s][-1])
            peak[stat].append(max(all_runs[stat][s]))
			
	
    fig, ax = plt.subplots(figsize=(8,5))
    ax.set_title('Contagion Evolution')
    ax.set_xlim((0, simulation_cycles))
    ax.set_ylim((-0.1,1.1))
    ax.axhline(y=get_parameters().get('icu_capacity'), c="black", ls='--', label='Critical limit')

    if zoomed_plot:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.set_title('Contagion Evolution')
        ax2.set_xlim((0, simulation_cycles))
        ax2.set_ylim(zoomed_plot_ylim)
        ax2.axhline(y=get_parameters().get('icu_capacity'), c="black", ls='--', label='Critical limit')

    for s in randomlist:
        adict = {stat:all_runs[stat][s] for stat in desired_stats} 
        #df = pd.DataFrame (data=adict)
        #df.to_csv(fname+"-"+str(s)+".csv")
  
    each_step = {}
    for stat in desired_stats:
        each_step[stat] = []
        for i in range(simulation_cycles):
            each_step[stat].append([all_runs [stat][s][i] for s in randomlist])
        for i in range(simulation_cycles):
            loweri, averagei, upperi = confidence_interval(each_step[stat][i], confidence=0.95)
            lower[stat].append(loweri)
            average[stat].append(averagei)
            upper[stat].append(upperi)
        #print (stat)
        #print (lower[stat])
        #print (upper[stat])
#Plotting:
        ax.plot(lower[stat], color=color[stat], linewidth=.3) #mean curve.
        ax.plot(average[stat], color=color[stat], linewidth=2, label=stat)
        ax.plot(upper[stat], color=color[stat], linewidth=.3)
        ax.fill_between(np.arange(simulation_cycles), lower[stat], upper[stat], color=color[stat], alpha=.1) #std curves.
        if zoomed_plot:
            ax2.plot(lower[stat], color=color[stat], linewidth=.3)  # mean curve.
            ax2.plot(average[stat], color=color[stat], linewidth=2, label=stat)
            ax2.plot(upper[stat], color=color[stat], linewidth=.3)
            ax2.fill_between(np.arange(simulation_cycles), lower[stat], upper[stat], color=color[stat],
                             alpha=.1)  # std curves.

			
    ax.set_xlabel("Days")
    ax.set_ylabel("Ratio of Population")
    handles, labels = ax.get_legend_handles_labels()
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    fig.show()
    fig.savefig(fname+".png")

    if zoomed_plot:
        ax2.set_xlabel("Days")
        ax2.set_ylabel("Ratio of Population")
        handles, labels = ax2.get_legend_handles_labels()
        # Shrink current axis by 20%
        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        ax2.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
        fig2.show()
        fig2.savefig(fname + ".png")

    if do_print:
        for stat, x in avg.items():
            print("using average of time series:")
            print("stats on {}:".format(stat))
            print("data: {}".format(x))
            print("min:")
            print(np.min(x))
            print("max:")
            print(np.max(x))
            print("std:")
            print(np.std(x))
            low, mean, high = confidence_interval(x, confidence=0.95)
            print("mean:")
            print(mean)
            print("median:")
            print(np.median(x))
            print("95% confidence interval for the mean:")
            print("({0},{1})".format(low, high))
        for stat, x in last.items():
            print("using last of time series:")
            print("stats on {}:".format(stat))
            print("data: {}".format(x))
            print("min:")
            print(np.min(x))
            print("max:")
            print(np.max(x))
            print("std:")
            print(np.std(x))
            low, mean, high = confidence_interval(x, confidence=0.95)
            print("mean:")
            print(mean)
            print("median:")
            print(np.median(x))
            print("95% confidence interval for the mean:")
            print("({0},{1})".format(low, high))
        for stat, x in peak.items():
            print("using peak of time series:")
            print("stats on {}:".format(stat))
            print("data: {}".format(x))
            print("min:")
            print(np.min(x))
            print("max:")
            print(np.max(x))
            print("std:")
            print(np.std(x))
            low, mean, high = confidence_interval(x, confidence=0.95)
            print("mean:")
            print(mean)
            print("median:")
            print(np.median(x))
            print("95% confidence interval for the mean:")
            print("({0},{1})".format(low, high))
    return avg.items, last.items, peak.items


class BasicStatistics:
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
        # print(f"infected = {self.covid_model.global_count.infected_count} "
        #       f"recovered = {self.covid_model.global_count.recovered_count}")
        self.susceptible.append(self.covid_model.global_count.susceptible_count / pop)
        self.infected.append(self.covid_model.global_count.infected_count / pop)
        self.recovered.append(self.covid_model.global_count.recovered_count / pop)
        self.hospitalization.append(self.covid_model.global_count.total_hospitalized / pop)
        self.icu.append(self.covid_model.global_count.high_severity_count / pop)
        self.death.append(self.covid_model.global_count.death_count / pop)
        self.income.append(self.covid_model.global_count.total_income / work_pop)
    
    def state_change(self, model):
        pass

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
            'Susceptible': 'lightblue',
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
        ax.axhline(y=get_parameters().get('icu_capacity'), c="black", ls='--', label='Critical limit')
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


class RemovePolicy:
    def __init__(self, model, policy, n):
        self.switch = n
        self.policy = policy
        self.model = model
        self.state = 0

    def start_cycle(self, model):
        pass

    def state_change(self,model):
        pass

    def end_cycle(self, model):
        if self.state == 0:
            if self.model.global_count.day_count == self.switch:
                if self.policy in get_parameters().get('social_policies'):
                    get_parameters().get('social_policies').remove(self.policy)
                self.state = 1

class AddPolicy:
    def __init__(self, model, policy, n):
        self.switch = n
        self.policy = policy
        self.model = model
        self.state = 0

    def start_cycle(self, model):
        pass

    def state_change(self,model):
        pass

    def end_cycle(self, model):
        if self.state == 0:
            if self.model.global_count.day_count == self.switch:
                get_parameters().get('social_policies').append(self.policy)
                self.state = 1


class AddPolicyInfectedRate:
    def __init__(self, model, policy, v):
        self.trigger = v
        self.policy = policy
        self.model = model
        self.state = 0

    def start_cycle(self, model):
        pass

    def state_change(self,model):
        pass

    def end_cycle(self, model):
        if self.state == 0:
            if self.model.global_count.infected_count / self.model.global_count.total_population >= self.trigger:
                get_parameters().get('social_policies').append(self.policy)
                self.state = 1
                
                
class AddPolicyInfectedRateWindow:
    # Adds a policy after the infection rate has been above a value for n cycles
    
    def __init__(self, model, policy, v, n):
        self.trigger = v
        self.policy = policy
        self.model = model
        self.state = 0
        self.recent_trigger_window =[]
        self.n = n
        
    def start_cycle(self, model):
        pass

    def state_change(self,model):
        pass

    def end_cycle(self, model):
        self.recent_trigger_window.append(self.model.global_count.infected_count/ self.model.global_count.total_population)
        while len( self.recent_trigger_window) > self.n:
            self.recent_trigger_window.pop(0)
        if self.state == 0 and len(self.recent_trigger_window) == self.n:
            add = True
            for rate in self.recent_trigger_window:
                if  rate < self.trigger:
                    add = False
            if add:
                get_parameters().get('social_policies').append(self.policy)
                self.state = 1
                
     
    
class RemovePolicyInfectedRateWindow:
    # Removes a policy after the infection rate has been below a value for n cycles
    
    def __init__(self, model, policy, v, n):
        self.trigger = v
        self.policy = policy
        self.model = model
        self.state = 0
        self.recent_trigger_window = []
        self.n=n

    def start_cycle(self, model):
        pass

    def state_change(self,model):
        pass

    def end_cycle(self, model):
        self.recent_trigger_window.append(self.model.global_count.infected_count/ self.model.global_count.total_population)
        while len( self.recent_trigger_window) > self.n:
            self.recent_trigger_window.pop(0)
        if self.state == 0 and len(self.recent_trigger_window) == self.n:
            remove = True
            for rate in self.recent_trigger_window:
                if  rate >= self.trigger:
                    remove = False
                
            if remove and self.policy in get_parameters().get('social_policies'):
                get_parameters().get('social_policies').remove(self.policy)
                self.state = 1

class RemovePolicyVaccinationTarget:
    # Removes a policy after the infection rate has been below a value for n cycles
    def __init__(self, model, policy, target_rate, work_class, age):
        assert bool(work_class is None) != bool(age is None) # Logical XOR. Exactly one is supposed to be None
        self.model = model
        self.policy = policy
        self.target_rate = target_rate
        self.work_class = work_class
        self.age = age
        self.state = 0

    def start_cycle(self, model):
        pass

    def state_change(self,model):
        pass

    def end_cycle(self, model):
        if self.state != 0:
            return
        total = 0
        vacinated = 0
        for human in [agent for agent in self.model.agents if isinstance(agent, Human)]:
            if (age is not None and human.age >= age) or \
            (work_class is not None and isinstance(human, Adult) and human.work_info.work_class == work_class):
                total += 1
                if human.vaccinated:
                    vaccinated += 1
        if total == 0 or (vaccinated / total) >= target_rate:
            if self.policy in get_parameters().get('social_policies'):
                get_parameters().get('social_policies').remove(self.policy)
                self.state = 1

class Propaganda:
    def __init__(self, model, n):
        self.switch = n
        self.count = 0
        self.model = model
        self.state = 0

    def start_cycle(self, model):
        pass

    def state_change(self,model):
        pass

    def tick(self):
        v = get_parameters().get('risk_tolerance_mean') - 0.1
        if v < 0.1:
            v = 0.1
        logger().debug(f'Global risk_tolerance change to {v}')
        change_parameters(risk_tolerance_mean = v)
        self.model.reroll_human_properties()

    def end_cycle(self, model):
        self.count += 1
        if self.state == 0:
            if self.model.global_count.day_count == self.switch:
                self.state = 1
                self.tick()
        elif self.state == 1:
            if not (self.count % 3):
                self.tick()

class Vaccination:
    def __init__(self, model, start_day, capacity_per_month, total_capacity, campaign_stages):
        self.model = model
        self.finished = False
        self.start_day = start_day
        self.capacity_per_day = capacity_per_month / 30
        self.total_capacity = total_capacity
        self.stages = campaign_stages
        self.vaccinated_count = 0
        self.current_stage = -1
        self.days_left_in_current_stage = 0
        self.allowed_work_classes = set()
        self.allowed_age = 1000 # infinity

    def start_cycle(self, model):
        pass

    def state_change(self,model):
        pass

    def tick(self):
        candidates = []
        for human in [agent for agent in self.model.agents if isinstance(agent, Human)]:
            if human.vaccinated:
                continue
            if human.age >= self.allowed_age or (isinstance(human, Adult) and human.work_info.work_class in self.allowed_work_classes):
                candidates.append(human)
        if len(candidates) == 0:
            return
        if self.capacity_per_day >= len(candidates):
            prob = 1
        else:
            prob = self.capacity_per_day / len(candidates)
        for human in candidates:
            if self.vaccinated_count < self.total_capacity and flip_coin(prob):
                human.vaccinate()
                self.vaccinated_count += 1

    def cleared_current_stage(self):
        if self.current_stage < 0 or self.days_left_in_current_stage == 0:
            return True
        else:
            self.days_left_in_current_stage -= 1
            return False

    def end_cycle(self, model):
        if self.model.global_count.day_count >= self.start_day and self.vaccinated_count < self.total_capacity and not self.finished:
            if self.cleared_current_stage():
                self.current_stage += 1
                if self.current_stage < len(self.stages):
                    num_days, work_classes, age = self.stages[self.current_stage]
                    assert work_classes is not None or age is not None
                    if work_classes is not None:
                        self.allowed_work_classes = self.allowed_work_classes.union(work_classes)
                    if age is not None:
                        self.allowed_age = age
                    self.days_left_in_current_stage = num_days
                else:
                    self.finished = True
            self.tick()


class Network:
    def __init__(self, model):
        #self.model = model      
        self.districts = [ agent for agent in model.agents if isinstance(agent,District)]
        #self.G = nx.Graph()
        self.G = nx.MultiGraph()
        self.clumpiness = [] 
        
    def start_cycle(self, model):
       self.state_change(model)

    def state_change(self,model):
        similarities = []
        for district in self.districts:
            for building in district.locations:
                for room in building.locations:
                    for human in room.humans:
                        if human.strid not in self.G.nodes:
                            self.G.add_node(human.strid)
                        if room.strid not in self.G.nodes:
                            self.G.add_node(room.strid)
                        weight = -1 * np.log(room.get_parameter('contagion_probability'))
                        self.G.add_edge(human.strid, room.strid,weight=weight)
                        #print (f"edge added betweem {human.strid} and {room.strid}")
                for human in building.humans:
                    if model.current_state == SimulationState.POST_WORK_ACTIVITY:
                        sim = model.hrf.similarity(model.hrf.feature_vector[human],model.hrf.unit_info_map[building.strid]["vector"])
                        similarities.append(sim)
                    if human.strid not in self.G.nodes:
                        self.G.add_node(human.strid)
                    if building.strid not in self.G.nodes:
                        self.G.add_node(building.strid)
                    weight = -1 * np.log(building.get_parameter('contagion_probability'))
                    self.G.add_edge(human.strid, building.strid,weight=weight)
                    #print (f"edge added between {human.strid} and {building.strid}")
        if len(similarities) > 0:
            avg_sim = np.mean(similarities)
            print(f"avg restaurant similarity {avg_sim}")
            similarities = []
    def end_cycle(self, model):
        #print ("self.G.nodes")
        #print (self.G.nodes)
        #print ("self.G.edges")
        #print (self.G.edges)
        self.clumpiness.append(self.compute_clumpiness2())
        self.G.clear()
        #self.G.remove_edges_from(self.G.edges())

    def compute_clumpiness1(self):
        avg_path = 0
        connected_component_subgraphs = [self.G.subgraph(c) for c in nx.connected_components(self.G)]
        for C in connected_component_subgraphs:
            avg_path += nx.average_shortest_path_length(C, weight="weight") * len(C.nodes)
        avg_path /= len(self.G.nodes)
        return avg_path

    def compute_clumpiness2(self):
        #Just sample 
        num_nodes = len(self.G.nodes)
        k = 100
        avg_len = 0
        disconnects = 0
        for i in range (k):
            nodes = random.sample(self.G.nodes, 2)
            try:
                shortest_path = nx.dijkstra_path(self.G,nodes[0],nodes[1], weight = "weight")
                shortest_path_len = len(shortest_path)
            except(nx.NetworkXNoPath):
                shortest_path_len = num_nodes
                disconnects += 1

            avg_len += shortest_path_len
        avg_len /= k*num_nodes
        disconnects /= k

        #print ("disconnects")
        #print (disconnects)

        return avg_len

        
        
def build_district(name, model, population_size, building_capacity, unit_capacity,
                   occupacy_rate, contagion_probability, home_district_list=[]):
    logger().info(f"Building district {name} contagion_probability = {contagion_probability}")
    district = District(name, model, '', name, home_district_list)
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
                                   beta_range(0.021, 0.12))  # normal_ci(0.021, 0.12, 10)
    work_district = build_district("Work", model, population_size,
                                   work_building_capacity,
                                   office_capacity,
                                   work_building_occupacy_rate,
                                   beta_range(0.007, 0.06))  # normal_ci(0.007, 0.06, 10)
    school_district = build_district("School", model, population_size,
                                     school_capacity,
                                     classroom_capacity,
                                     school_occupacy_rate,
                                     beta_range(0.014, 0.08))  # normal_ci(0.014, 0.08, 10)

    home_district.debug = model.debug
    work_district.debug = model.debug
    school_district.debug = model.debug

    hospital = Hospital(10, 0.01, model, 'HospitalBuilding', 'unique', beta_range(0.001, 0.06))
    hospital_container = District('HospitalContainer', model, 'HospitalContainer', 'unique', [])
    hospital_container.locations.append(hospital)

    # Add Restaurants to work_district

    for i in range(get_parameters().params['restaurant_count_per_work_district']):
        if flip_coin(0.5):
            restaurant_type = RestaurantType.FAST_FOOD
            rtype = "FASTFOOD"
        else:
            restaurant_type = RestaurantType.FANCY
            rtype = "FANCY"
        restaurant = Restaurant(
            normal_cap(
                get_parameters().params['restaurant_capacity_mean'], 
                get_parameters().params['restaurant_capacity_stdev'], 
                16, 
                200
            ),
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

    # print(home_district)
    # print(work_district)
    # print(school_district)

    # Build families

    family_factory = FamilyFactory(model)
    family_factory.factory(population_size)
    model.global_count.total_population = family_factory.human_count

    # print(family_factory)

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
    # exit()

    count = 0
    for family in family_factory.families:
        for human in family:
            human.hospital_district = hospital_container
            count += 1
            human.tribe[TribeSelector.AGE_GROUP] = age_group_sets[type(human)]
            human.tribe[TribeSelector.FAMILY] = family
            if isinstance(human, Adult):
                human.unique_id = "Adult" + str(count)
                human.tribe[TribeSelector.COWORKER] = work_district.get_buildings(human)[0].get_unit(human).allocation
                t1 = adult_rf.build_tribe(human, human.tribe[TribeSelector.COWORKER], 1, office_capacity)
                t2 = adult_rf.build_tribe(human, human.tribe[TribeSelector.AGE_GROUP], 1, 20)
                human.tribe[TribeSelector.FRIEND] = t1
                for h in t2:
                    if h not in human.tribe[TribeSelector.FRIEND]:
                        human.tribe[TribeSelector.FRIEND].append(h)
            elif isinstance(human, K12Student):
                human.unique_id = "K12Student" + str(count)
                human.tribe[TribeSelector.CLASSMATE] = school_district.get_buildings(human)[0].get_unit(
                    human).allocation
                t1 = student_rf.build_tribe(human, human.tribe[TribeSelector.CLASSMATE], 1, classroom_capacity)
                t2 = student_rf.build_tribe(human, human.tribe[TribeSelector.AGE_GROUP], 1, 20)
                human.tribe[TribeSelector.FRIEND] = t1
                for h in t2:
                    if h not in human.tribe[TribeSelector.FRIEND]:
                        human.tribe[TribeSelector.FRIEND].append(h)
            elif isinstance(human, Elder):
                human.unique_id = "Elder" + str(count)
            elif isinstance(human, Infant):
                human.unique_id = "Infant" + str(count)
            elif isinstance(human, Toddler):
                human.unique_id = "Toddler" + str(count)

def setup_grid_layout(model, population_size,
        home_grid_height, home_grid_width,work_height,work_width, school_height, school_width):
    #Makes a grid of homogeneous home districts, overlaid by school and work districts.
    #home_grid_height is the number of home districts high the grid is, and
    #home_grid_width is the nmber of home districts wide the grid is
    #school height and work height are how many home districts high a school
    #district and work are respectively, and the same for their length.
    #each begins in grid 0,0 and cover the orignal home district grid.
    #Persons assigned to the home districts are also assigned to the school
    #and work districts that cover them. The parameters determine the amount
    #of leakage across groups of people.  With parameters (10,10,1,1,1,1) you get 100
    #completely separated districts with no leakage.  With parameters (6,6,2,2,3,3) you
    #get a grid where every one is connected to everyone else, but there is a
    #degree of separation.  For example, a person in home district (0,0) can be infected
    #by a person in (5,5) but it would be bridged by three infections, slowing the
    #virus down.  Larger sizes for work and school districts enable faster spread. Fastest
    #spread occurs with parameters (1,1,1,1,1,1) or equivalently (10,10, 10,10,10,10)
    #or any of the same number
    #Since this is just a way to allocate human interactions, no label is needed and
    #the grid need not be saved, for interactions to occur, although this inforamtion
    #may be useful for visualizations.    
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

    home_districts = []
    work_districts=[]
    school_districts = []
    school_map ={}
    work_map = {}
    school_grid_height = math.ceil(home_grid_height/school_height)
    school_grid_width = math.ceil(home_grid_width/school_width)
    work_grid_height = math.ceil(home_grid_height/work_height)
    work_grid_width = math.ceil(home_grid_width/work_width)

    if ENABLE_WORKER_CLASS_SPECIAL_BUILDINGS:
        hospital = Hospital(10, 0.01, model, 'HospitalBuilding', 'unique', beta_range(0.001, 0.06))
        hospital_container = District('HospitalContainer', model, 'HospitalContainer', 'unique', [])
        hospital_container.locations.append(hospital)

    for hw in range(home_grid_width):
        for hh in range(home_grid_height):

            home_district = build_district(f"Home ({hh},{hw})", model, population_size,
                                   appartment_building_capacity,
                                   appartment_capacity,
                                   appartment_building_occupacy_rate,
                                   beta_range(0.021, 0.12))  # normal_ci(0.021, 0.12, 10)

            home_district.debug = model.debug

            home_districts.append(home_district)
            home_number = hw*home_grid_height + hh 
            assert home_number == len(home_districts) - 1

            sh = hh // school_height 
            sw = hw // school_width
            school_number = sw*school_grid_height+ sh
            school_map[home_number] = school_number

            wh = hh // work_height 
            ww = hw // work_width
            work_number = ww*work_grid_height+ wh
            work_map[home_number] = work_number


    for ww in range(work_grid_width):
        for wh in range(work_grid_height):
             
            work_district = build_district(f"Work ({wh},{ww})", model, population_size,
                                   work_building_capacity,
                                   office_capacity,
                                   work_building_occupacy_rate,
                                   beta_range(0.007, 0.06))  # normal_ci(0.007, 0.06, 10)
            # Add Restaurants to work_district

            for i in range(get_parameters().params['restaurant_count_per_work_district']):
                if flip_coin(0.5):
                    restaurant_type = RestaurantType.FAST_FOOD
                    rtype = "FASTFOOD"
                else:
                    restaurant_type = RestaurantType.FANCY
                    rtype = "FANCY"
                restaurant = Restaurant(
                    normal_cap(
                        get_parameters().params['restaurant_capacity_mean'], 
                        get_parameters().params['restaurant_capacity_stdev'], 
                        16, 
                        200
                    ),
                    restaurant_type,
                    flip_coin(0.5),
                    model,
                    '',
                    rtype + '-' + str(i)+ f"({wh},{ww})")
                work_district.locations.append(restaurant)
            for i in range(2):
                bar = Restaurant(
                    normal_cap(100, 20, 50, 200),
                    RestaurantType.BAR,
                    flip_coin(0.5),
                    model,
                    '',
                    'BAR-' + str(i)+ f"({wh},{ww})")
                work_district.locations.append(bar)
            work_district.debug = model.debug
            work_districts.append(work_district)


    for sw in range(school_grid_width):
        for sh in range(school_grid_height):
    
            school_district = build_district(f"School ({sh},{sw})", model, population_size,
                                     school_capacity,
                                     classroom_capacity,
                                     school_occupacy_rate,
                                     beta_range(0.014, 0.08))  # normal_ci(0.014, 0.08, 10)
            
            school_district.debug = model.debug
            school_districts.append(school_district)

    #print ("work_map")
    #print (work_map)
    #print ("school_map")
    #print (school_map)
    

    # Build families

    family_factory = FamilyFactory(model)
    family_factory.factory(population_size)
    model.global_count.total_population = family_factory.human_count

    # print(family_factory)

    age_group_sets = {
        Infant: [],
        Toddler: [],
        K12Student: [],
        Adult: [],
        Elder: []
    }

    # Allocate buildings to people
    #print ("home_districts")
    #print (home_districts)
    #print ("work_districts")
    #print (work_districts)
    #print ("school_districts")
    #print (school_districts)

    all_adults = []
    all_students = []
    for family in family_factory.families:
        adults = [human for human in family if isinstance(human, Adult)]
        students = [human for human in family if isinstance(human, K12Student)]
        home_district_num = np.random.randint (0,len(home_districts))
        #print("home_district_num")
        #print(home_district_num)
        home_district = home_districts[home_district_num]
        work_district = work_districts[work_map[home_district_num]]
        school_district = school_districts[school_map[home_district_num]]
    
        home_district.allocate(family, True, True, True)
        work_district.allocate(adults)
        school_district.allocate(students, True)
        for human in family:
            age_group_sets[type(human)].append(human)
            human.home_district = home_district
            home_district.get_buildings(human)[0].get_unit(human).humans.append(human)
            if ENABLE_WORKER_CLASS_SPECIAL_BUILDINGS:
                human.hospital_district = hospital_container
        for adult in adults:
            adult.work_district = work_district
            all_adults.append(adult)
        for student in students:
            student.school_district = school_district
            all_students.append(student)

    # Set tribes

    adult_rf = HomophilyRelationshipFactory(model, all_adults)
    student_rf = HomophilyRelationshipFactory(model, all_students)
    # exit()

    count = 0
    for family in family_factory.families:
        for human in family:
            work_district = human.work_district
            school_district = human.school_district
            count += 1
            human.tribe[TribeSelector.AGE_GROUP] = age_group_sets[type(human)]
            human.tribe[TribeSelector.FAMILY] = family
            if isinstance(human, Adult):
                human.unique_id = "Adult" + str(count)
                #print(work_district.get_buildings(human))
                #print(work_district.get_buildings(human))
                #print(workd_district.get_buildings(human)[0].get_unit(human))
                #print(workd_district.get_buildings(human)[0].get_unit(human))
                human.tribe[TribeSelector.COWORKER] = work_district.get_buildings(human)[0].get_unit(human).allocation
                t1 = adult_rf.build_tribe(human, human.tribe[TribeSelector.COWORKER], 1, office_capacity)
                t2 = adult_rf.build_tribe(human, human.tribe[TribeSelector.AGE_GROUP], 1, 20)
                human.tribe[TribeSelector.FRIEND] = t1
                for h in t2:
                    if h not in human.tribe[TribeSelector.FRIEND]:
                        human.tribe[TribeSelector.FRIEND].append(h)
            elif isinstance(human, K12Student):
                human.unique_id = "K12Student" + str(count)
                human.tribe[TribeSelector.CLASSMATE] = school_district.get_buildings(human)[0].get_unit(
                    human).allocation
                t1 = student_rf.build_tribe(human, human.tribe[TribeSelector.CLASSMATE], 1, classroom_capacity)
                t2 = student_rf.build_tribe(human, human.tribe[TribeSelector.AGE_GROUP], 1, 20)
                human.tribe[TribeSelector.FRIEND] = t1
                for h in t2:
                    if h not in human.tribe[TribeSelector.FRIEND]:
                        human.tribe[TribeSelector.FRIEND].append(h)
            elif isinstance(human, Elder):
                human.unique_id = "Elder" + str(count)
            elif isinstance(human, Infant):
                human.unique_id = "Infant" + str(count)
            elif isinstance(human, Toddler):
                human.unique_id = "Toddler" + str(count)



def setup_homophilic_layout(model, population_size,home_grid_height, home_grid_width,work_home_list=[],school_home_list=[],
         temperature=-1):
    # This is made to be implemented on a realistic map.  The input is meant to describe a realistic map.
    # Send a grid shape of home districts and two list of lists of grid tuples of the home district representing 
    # the school and work districts that the homes belong in.  Non-grids shapes can be projected onto a grid with 
    # smaller home district sizes for higher resolution.  An empty list assumes one work district.  Each list in 
    # the list of lists is one of the work or school districts. Represent the districts with a list of tuples
    #(x,y) where x is the place among the width and y is the place along the height.
    
     
    work_building_capacity = 70
    office_capacity =3 
    work_building_occupacy_rate = 1.0 
    appartment_building_capacity = 20
    appartment_capacity = 5
    appartment_building_occupacy_rate = 0.5
    school_capacity = 6
    classroom_capacity = 5
    school_occupacy_rate = 1.0
    num_favorite_restaurants =2 
    family_temperature =  get_parameters().params['temperature']
    home_room_temperature = get_parameters().params['temperature']
    school_room_temperature = get_parameters().params['temperature']
    work_room_temperature = get_parameters().params['temperature']
    home_temperature = -1
    school_temperature = get_parameters().params['temperature']
    work_temperature = get_parameters().params['temperature']
    restaurant_temperature = get_parameters().params['temperature']


    home_districts = []
    work_districts=[]
    school_districts = []
    home_district_in_position = {}
    agents_per_home_district = math.ceil(population_size/(home_grid_width*home_grid_height))
    agents_per_school_district = math.ceil((0.25*population_size) /len(school_home_list))
    agents_per_work_district = math.ceil((0.5*population_size)/len(work_home_list))

    if ENABLE_WORKER_CLASS_SPECIAL_BUILDINGS:
        hospital = Hospital(10, 0.01, model, 'HospitalBuilding', 'unique', beta_range(0.001, 0.06))
        hospital_container = District('HospitalContainer', model, 'HospitalContainer', 'unique', [])
        hospital_container.locations.append(hospital)

    for hw in range(home_grid_width):
        for hh in range(home_grid_height):

            home_district = build_district(f"Home ({hh},{hw})", model, agents_per_home_district,
                                   appartment_building_capacity,
                                   appartment_capacity,
                                   appartment_building_occupacy_rate,
                                   beta_range(0.021, 0.12))  # normal_ci(0.021, 0.12, 10)

            home_district.debug = model.debug

            home_districts.append(home_district)
            home_district_in_position[(hh,hw)] = home_district

    for w in range(len(work_home_list)):
        work_district = build_district(f"Work ({w})", model, agents_per_work_district,
                               work_building_capacity,
                               office_capacity,
                               work_building_occupacy_rate,
                               beta_range(0.007, 0.06),
                               work_home_list[w])  # normal_ci(0.007, 0.06, 10)
        # Add Restaurants to work_district

        for i in range(get_parameters().params['restaurant_count_per_work_district']):
            if flip_coin(0.5):
                restaurant_type = RestaurantType.FAST_FOOD
                rtype = "FASTFOOD"
            else:
                restaurant_type = RestaurantType.FANCY
                rtype = "FANCY"
            restaurant = Restaurant(
                normal_cap(
                    get_parameters().params['restaurant_capacity_mean'], 
                    get_parameters().params['restaurant_capacity_stdev'], 
                    16, 
                    200
                ),
                restaurant_type,
                flip_coin(0.5),
                model,
                '',
                rtype + '-' + str(i)+ f"({w})")
            work_district.locations.append(restaurant)
        for i in range(2):
            bar = Restaurant(
                normal_cap(100, 20, 50, 200),
                RestaurantType.BAR,
                flip_coin(0.5),
                model,
                '',
                'BAR-' + str(i)+ f"({w})")
            work_district.locations.append(bar)
        work_district.debug = model.debug
        work_districts.append(work_district)

    for s in range(len(school_home_list)):

        school_district = build_district(f"School ({s})", model, agents_per_school_district,
                                 school_capacity,
                                 classroom_capacity,
                                 school_occupacy_rate,
                                 beta_range(0.014, 0.08),
                                 school_home_list[s])  # normal_ci(0.014, 0.08, 10)
        
        school_district.debug = model.debug
        school_districts.append(school_district)


    # print(home_district)
    # print(work_district)
    # print(school_district)

    # Build families

    family_factory = FamilyFactory(model)
    family_factory.factory(population_size)
    model.global_count.total_population = family_factory.human_count
    #print ("family_factory.human_count")
    #print (family_factory.human_count)

    # print(family_factory)
    hrf = HomophilyRelationshipFactory(model,family_factory.human_count,get_parameters().params['num_communities'],
        get_parameters().params['num_features'],home_district_in_position)
    model.hrf = hrf
    hrf.assign_features_to_families(family_factory.families,family_temperature)
    hrf.map_home_districts_to_blobs(home_grid_height,home_grid_width)
    hrf.assign_features_to_homes(home_room_temperature)
    hrf.assign_features_to_schools(school_room_temperature)
    hrf.assign_features_to_offices(work_room_temperature)

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
    
        #home_district.allocate(family, True, True, True)
        #work_district.allocate(adults)
        #school_district.allocate(students, True)
        for human in family:
            age_group_sets[type(human)].append(human)
            #human.home_district = home_district
            #home_district.get_buildings(human)[0].get_unit(human).humans.append(human)
            if ENABLE_WORKER_CLASS_SPECIAL_BUILDINGS:
                human.hospital_district = hospital_container
        for adult in adults:
            #adult.work_district = work_district
            all_adults.append(adult)
        for student in students:
            #student.school_district = school_district
            all_students.append(student)

    hrf.allocate_homes(family_factory.families, home_temperature)
    hrf.allocate_school_districts(school_districts,school_temperature)
    hrf.allocate_work_districts(work_districts, work_temperature)
    hrf.allocate_favorite_restaurants(all_adults, restaurant_temperature, num_favorite_restaurants)


    # Set tribes

    adult_friend_similarity = []
    student_friend_similarity = []
    count = 0
    for family in family_factory.families:
        for human in family:
            count += 1
            human.tribe[TribeSelector.AGE_GROUP] = age_group_sets[type(human)]
            human.tribe[TribeSelector.FAMILY] = family
            if isinstance(human, Adult):
                human.unique_id = "Adult" + str(count)
                if human.work_district is not None:
                    human.tribe[TribeSelector.COWORKER] = human.work_district.get_buildings(human)[0].get_unit(human).allocation
                else:
                    print(f"Adult {human} was not assigned a work district")
                t1 = hrf.build_tribe(human, human.tribe[TribeSelector.COWORKER], 1, office_capacity,restaurant_temperature)
                t2 = hrf.build_tribe(human, human.tribe[TribeSelector.AGE_GROUP], 1, 20, restaurant_temperature)
                human.tribe[TribeSelector.FRIEND] = t1
                for h in t2:
                    if h not in human.tribe[TribeSelector.FRIEND]:
                        human.tribe[TribeSelector.FRIEND].append(h)
                for h in human.tribe[TribeSelector.FRIEND]:
                    sim = hrf.similarity(hrf.feature_vector[human],hrf.feature_vector[h])
                    adult_friend_similarity.append(sim)
            elif isinstance(human, K12Student):
                human.unique_id = "K12Student" + str(count)
                if human.school_district is not None:
                    if len(human.school_district.get_buildings(human)) > 0:
                        human.tribe[TribeSelector.CLASSMATE] = human.school_district.get_buildings(human)[0].get_unit(
                    human).allocation
                else:
                    print (f"student {human} wasnt assigned a school district")
                t1 = hrf.build_tribe(human, human.tribe[TribeSelector.CLASSMATE], 1, classroom_capacity, restaurant_temperature)
                t2 = hrf.build_tribe(human, human.tribe[TribeSelector.AGE_GROUP], 1, 20, restaurant_temperature)
                human.tribe[TribeSelector.FRIEND] = t1
                for h in t2:
                    if h not in human.tribe[TribeSelector.FRIEND]:
                        human.tribe[TribeSelector.FRIEND].append(h)
                for h in human.tribe[TribeSelector.FRIEND]:
                    sim = hrf.similarity(hrf.feature_vector[human],hrf.feature_vector[h])
                    student_friend_similarity.append(sim)
            elif isinstance(human, Elder):
                human.unique_id = "Elder" + str(count)
            elif isinstance(human, Infant):
                human.unique_id = "Infant" + str(count)
            elif isinstance(human, Toddler):
                human.unique_id = "Toddler" + str(count)
    avg_adult_sim = np.mean(adult_friend_similarity)
    avg_student_sim = np.mean(student_friend_similarity)
    print (f"Average friend similarity for adults: {avg_adult_sim} for kids: {avg_student_sim}")
    print ("home_districts")
    print (home_districts)
    print ("work_districts")
    print (work_districts)
    print ("school_districts")
    print (school_districts)

 
