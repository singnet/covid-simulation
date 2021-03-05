import matplotlib.pyplot as plt
import pandas as pd
from model.base import CovidModel, get_parameters, change_parameters, flip_coin, normal_cap, logger, ENABLE_WORKER_CLASS_SPECIAL_BUILDINGS
from model.human import Elder, Adult, K12Student, Toddler, Infant
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
                  zoomed_plot_ylim=(-0.01, .12),compute_hoprank = False):
    color = {
            'susceptible': 'lightblue',
            'infected': 'gray',
            'recovered': 'lightgreen',
            'death': 'black',
            'hospitalization': 'orange',
            'icu': 'red',
            'income': 'magenta',
            'clumpiness':'purple',
            'maxlen':'yellow'
        }
    if desired_stats is None:
        desired_stats = ["susceptible", "infected", "recovered", "hospitalization", "icu", "death", "income","clumpiness","maxlen"]
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
        network = Network(model,fname,compute_hoprank)
        model.add_listener(network)
        for i in range(simulation_cycles):
            model.step()
        #print("clumpiness:")
        #print (getattr(network,"clumpiness"))
        network.print_infections()
        for stat in desired_stats:
            if stat is "clumpiness":
                all_runs[stat][s]=copy.deepcopy(getattr(network,"clumpiness"))
            elif stat is "maxlen":
                all_runs[stat][s]=copy.deepcopy(getattr(network,"maxlen"))
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
	
    fig3, ax3 = plt.subplots(figsize=(8,5))
    ax3.set_title('Clumpiness')
    ax3.set_xlim((0, simulation_cycles))
    #ax3.set_ylim((-0.1,1.1))
    #ax3.axhline(y=get_parameters().get('icu_capacity'), c="black", ls='--', label='Critical limit')


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
    ax3.plot(lower["clumpiness"], color=color["clumpiness"], linewidth=.3)  # mean curve.
    ax3.plot(average["clumpiness"], color=color["clumpiness"], linewidth=2, label="clumpiness")
    ax3.plot(upper["clumpiness"], color=color["clumpiness"], linewidth=.3)
    ax3.fill_between(np.arange(simulation_cycles), lower["clumpiness"], upper["clumpiness"], color=color["clumpiness"],
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
    
    ax3.set_xlabel("Days")
    #ax3.set_ylabel("Ratio of Population")
    handles, labels = ax3.get_legend_handles_labels()
        # Shrink current axis by 20%
    box = ax3.get_position()
    ax3.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
    ax3.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    fig3.show()
    fig3.savefig(fname + ".png")


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
    def __init__(self, model, start_day, max_days_until_full_vaccination, work_class=None, age=None):
        self.model = model
        self.start_day = start_day
        self.end_day = start_day + max_days_until_full_vaccination - 1
        self.work_class = work_class

    def start_cycle(self, model):
        pass

    def state_change(self,model):
        pass

    def tick(self):
        candidates = []
        for agent in self.model.agents:
            if work_class is None or isinstance(agent, Adult) and agent.work_info.work_class == self.work_class:
                if age is None or isinstance(agent, Human) and agent.age >= age:
                    candidates.append(agent)
        if self.model.global_count.day_count >= self.end_day:
            prob = 1
        else:
            prob = (self.model.global_count.day_count - self.start_day + 1) / (self.end_day - self.start_day + 1)
        for human in candidates:
            if not human.vaccinated:
                if flip_coin(prob):
                    human.vaccinate()

    def end_cycle(self, model):
        if self.model.global_count.day_count >= self.start_day and self.model.global_count.day_count <= self.end_day:
            self.tick()


import networkx as nx


class Network:
    def __init__(self, model,fname, compute_hoprank = False):
        #self.model = model      
        self.districts = [ agent for agent in model.agents if isinstance(agent,District)]
        self.G = nx.Graph()
        #self.G = nx.MultiGraph()
        #self.G = nx.DiGraph()
        #self.old_clumpiness = []
        self.maxlen = []
        self.clumpiness = [] 
        self.location_hopranks = {}
        self.blob_hopranks = {}
        self.hoprank_cycle =get_parameters().params['hoprank_cycle'] 
        #for old clumpiness put num_nodes
        self.infinity =  get_parameters().params['infinity'] 
        #self.infinity= 9
        self.fname = fname
        self.compute_hoprank = compute_hoprank

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
                        #For new clumpiness:
                        #weight = math.sqrt(room.get_parameter('contagion_probability'))
                        self.G.add_edge(room.strid,human.strid,weight=weight)
                        #print (f"edge added betweem {human.strid} and {room.strid}")
                for human in building.humans:
                    if model.current_state == SimulationState.POST_WORK_ACTIVITY:
                        sim = model.hrf.similarity(model.hrf.feature_vector[human],model.hrf.unit_info_map[building.strid]["vector"])
                        similarities.append(sim)
                    if human.strid not in self.G.nodes:
                        self.G.add_node(human.strid)
                    if building.strid not in self.G.nodes:
                        self.G.add_node(building.strid)
                    #for new clumpiness:
                    #weight = math.sqrt(room.get_parameter('contagion_probability'))
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

        #To run new clumpiness uncomment the following:
        #print("computing old clumpiness...")
        #self.old_clumpiness.append(self.compute_clumpiness2())
        #print("computing new clumpiness...")
        #clumpiness,hoprank = self.compute_clumpiness3(compute_hoprank = self.compute_hoprank)
        #print("ending cycle...")
        self.actual_infections=model.actual_infections
        if self.compute_hoprank and  model.global_count.day_count % self.hoprank_cycle==0:
            hoprank=self.compute_maxprob_hoprank(model) 
            self.location_hopranks=self.compute_location_hopranks(model,hoprank)
            self.blob_hopranks=self.compute_blob_hopranks(model,hoprank)
            self.print_hopranks(model.global_count.day_count)

        #To run new clumpiness comment the following:
        clumpiness,maxlen = self.compute_clumpiness2()
        print (f"clumpiness {clumpiness}")
        self.clumpiness.append(clumpiness)
        print (f"maxlen {maxlen}")
        self.maxlen.append(maxlen)
                
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
        k = get_parameters().params["num_samples_clumpiness"] 
        avg_len = 0
        disconnects = 0
        shortest_paths = []
        avg_len_no_infinity = 0
        max_len_no_infinity = 0
        min_len_no_infinity=100000
        for i in range (k):
            nodes = random.sample(self.G.nodes, 2)
            try:
                shortest_path = nx.dijkstra_path(self.G,nodes[0],nodes[1], weight = "weight")
                shortest_path_len = len(shortest_path)
                if shortest_path_len > max_len_no_infinity:
                    max_len_no_infinity = shortest_path_len
                if shortest_path_len < min_len_no_infinity:
                    min_len_no_infinity = shortest_path_len
                avg_len_no_infinity += shortest_path_len
                shortest_paths.append(shortest_path)
            except(nx.NetworkXNoPath):
                shortest_path_len =self.infinity 
                disconnects += 1

            avg_len += shortest_path_len
        shortest_paths.sort(key=len)
        avg_len /= k
        if disconnects < k:
            avg_len_no_infinity /=(k-disconnects) 
        #pathlens = [len(p) for p in shortest_paths]
        #avg_len /= k*num_nodes
        disconnects /= k
        #print ("max_len_no_infinity")
        #print (max_len_no_infinity)
        #print ("avg_len_no_infinity")
        #print (avg_len_no_infinity)
        #print ("min_len_no_infinity")
        #print (min_len_no_infinity)
        print ("disconnects")
        print (disconnects)
        #print ("pathlens")
        #print (pathlens)

        return avg_len,max_len_no_infinity

    def print_infections(self):
        df = pd.DataFrame.from_dict(data = self.actual_infections)
        df.to_csv(f"{self.fname}-actual_infections.csv")


    def print_hopranks(self, day):
        pr = nx.pagerank(self.G)
        df = pd.DataFrame.from_dict(data = pr,orient='index')
        df["pred"] = day
        df.to_csv(f"{self.fname}-{day}-pagerank.csv")
        #print(f"pagerank: {test}")
        #print ('self.blob_hopranks')
        #print (self.blob_hopranks)
        #print ('self.location_hopranks')
        #print (self.location_hopranks)
        df = pd.DataFrame.from_dict(data = self.location_hopranks["Restaurant"],orient='index')
        df["pred"] = day
        df.to_csv(f"{self.fname}-{day}-restaurant_hopranks.csv")
        df = pd.DataFrame.from_dict(data = self.location_hopranks["School"],orient='index')
        df["pred"] = day
        df.to_csv(f"{self.fname}-{day}-school_hopranks.csv")
        df = pd.DataFrame.from_dict(data = self.location_hopranks["Office"], orient='index')
        df["pred"] = day
        df.to_csv(f"{self.fname}-{day}-office_hopranks.csv")
        df = pd.DataFrame.from_dict(data = self.location_hopranks["Home_District"], orient='index')
        df["pred"] = day
        df.to_csv(f"{self.fname}-{day}-home_district_hopranks.csv")
        df = pd.DataFrame.from_dict(data=self.blob_hopranks, orient='index')
        df["pred"] = day
        df.to_csv(f"{self.fname}-{day}-blob_hopranks.csv")
        self.print_infections()


    def compute_blob_hopranks(self,model,hoprank):
        blobranks = {}
        avg_blobranks = {}
        for node,hops in hoprank.items():
            if node in model.hrf.strid_to_human:
                blob = model.hrf.vector_to_blob[model.hrf.feature_vector[model.hrf.strid_to_human[node]]]
                if blob not in blobranks:
                    blobranks[blob] = []
                blobranks[blob].append(hops)
        for blob,hoplist in blobranks.items():
            avg_blobranks[blob] = [np.mean(hoplist)]
        return avg_blobranks

    def compute_location_hopranks(self,model,hoprank):
        ranks = {}
        ranks["School"] = {}
        ranks["Office"]= {}
        ranks["Home_District"] ={}
        ranks["Restaurant"] = {}

        home_districts = {}
        for node,hops in hoprank.items():
            if "Restaurant" in node:
                ranks["Restaurant"][node]=[hops]
            elif node in model.hrf.unit_info_map and len(model.hrf.unit_info_map[node]["unit"].allocation)> 1:
                if "Home" in node:
                    did = model.hrf.unit_info_map[node]["district"].strid 
                    if did not in home_districts:
                        home_districts[did]=[]
                    home_districts[did].append(hops)
                elif "School" in node:
                    ranks["School"][node]=[hops]
                elif "Work" in node:
                    ranks["Office"][node]=[hops]

        for did,hoplist in home_districts.items():
            ranks["Home_District"][did]= [np.mean(hoplist)]

        return ranks


    def probabilities_by_grouped_lengths(self,node,probs_by_lengths):
        # probs_by_lengths comes in [node1][node2][len] form, and we are changing it to [len]
        lengths_to_probs = {}
        if node in probs_by_lengths:
            for node2,len_dict in probs_by_lengths[node].items():
                for length,prob in len_dict.items():
                    if length not in lengths_to_probs:
                        lengths_to_probs[length] = 1.
                    lengths_to_probs[length] *= (1.-prob)
        for node2,node_dict in probs_by_lengths.items():
            if node in node_dict:
                for length,prob in node_dict[node].items():
                    if length not in lengths_to_probs:
                        lengths_to_probs[length] = 1.
                    lengths_to_probs[length] *= (1.-prob)
        lengths_to_probabilities = {k:1.-v for k,v in lengths_to_probs.items()}
       # if len(lengths_to_probabilities) > 0:
            #print(f"{node} hops to probs:  {lengths_to_probabilities}")
        return lengths_to_probabilities 
                

    def probabilities_by_individual_lengths(self,node1,node2):
        print(f"finding simple_paths from node {node1} to {node2}")
        try:
            #all_simple_paths = nx.shortest_simple_paths(self.G, node1,node2)
            all_simple_paths = nx.all_simple_paths(self.G,node1,node2,cutoff=self.infinity-1.)
            #test= nx.pagerank(self.G)
            #print(f"pagerank: {test}")
            pathlist = list(all_simple_paths)
        except(nx.NetworkXNoPath):
            pathlist = []
            print("no path found")        
        num_paths = len(pathlist)
        
        if num_paths > 0:
            maxlen = 0
            for path in pathlist:
                #print (path)
                length = len(path)
                if length > maxlen:
                    maxlen = length
            print(f"found {num_paths} paths, greatest length of {maxlen}  from {node1} to {node2}")

        # Since the packet can replicate it is ok to treat all routes as though non overlapping
        lengths_to_probabilities = {}
        temp = {}
        #for path in all_simple_paths:
        for path in pathlist:
            pathlength = len(path)
            if pathlength not in temp:
                temp[pathlength] = set()
            temp[pathlength].add(tuple(path))
        for length,pathset in temp.items():
            prob_all_paths_of_length = 1.
            for path in pathset:
                lastnode = None
                prob = 1.
                for node in path:
                    if lastnode is None:
                        lastnode = node
                    else:
                        factor = self.G[lastnode][node]['weight'] if lastnode in self.G and node in self.G[lastnode] else None
                        if factor is None:
                            factor = self.G[node][lastnode]['weight'] if node in self.G and lastnode in self.G[node] else None
                        if factor is not None:
                            prob*=factor
                        else:
                            print(f"error, nodes {node} and {lastnode} have no connection in G but are in path {path}")
                        lastnode = node
                prob_all_paths_of_length *= (1.-prob)
            lengths_to_probabilities[length] = 1.-prob_all_paths_of_length
       # if len(lengths_to_probabilities) > 0:
           # print(f"node {node1} to {node2} lengths to probabilities are {lengths_to_probabilities}")
        return lengths_to_probabilities
                
    def clumpiness_given_lengths(self,lengths_to_probabilities):
        #Probability of a packet traveling through paths of different sizes between two nodes
        cumulative_isnts = 1.
        clumpiness = 0.
        for length,prob in lengths_to_probabilities.items():
            clumpiness += length*cumulative_isnts*prob
            cumulative_isnts *=(1.-prob)
        clumpiness += cumulative_isnts * self.infinity
        return clumpiness


    def sample_from_infected_nodes(self,n,model):
        to_sample= []
        returnlist = []
        for blob_num in model.hrf.infected_blobs:
            to_sample.extend( model.hrf.blob_dict[blob_num])
        tups_to_sample = [model.hrf.vector_to_human[tuple(v)].strid for v in to_sample] 
        num_remaining = n-len(tups_to_sample)
        if num_remaining > 0:
            returnlist.extend(random.sample(list(set(self.G.nodes)-set(tups_to_sample)),num_remaining))
            n -= num_remaining
        returnlist.extend(random.sample(tups_to_sample,n))
        return returnlist


    def compute_maxprob_hoprank(self,model):
        #Just sample 
        number_to_hoprank= get_parameters().params["number_to_hoprank"]
        n=get_parameters().params["num_samples_clumpiness"]
        ratio_sample_from_infected=get_parameters().params["hoprank_infected_sample_ratio"]
        n_from_infected = math.ceil(n*ratio_sample_from_infected)
        hops = {}
        hoprank = {}
        clumpiness = {}
        to_be_hopranked = random.sample(self.G.nodes,number_to_hoprank)

        #nodes1=self.sample_from_infected_nodes(number_to_hoprank,model
                #) if sample_from_infected else random.sample(self.G.nodes,n)
        for node1 in to_be_hopranked:
            #for each agent choose a random n agents
            #nodes = random.sample(self.G.nodes,n+1)
            nodes=self.sample_from_infected_nodes(n_from_infected,model)
            #print (f"infected samples: {len(nodes)}")
            nodes.extend(random.sample(list(set(self.G.nodes) - set(nodes)-set(node1)),n-n_from_infected))
            #print (f"all samples:{len(nodes)}")
            avg_len = 0
            for node2 in nodes:
                if node1 < node2:
                    firstnode = node1
                    secondnode = node2
                else:
                    firstnode = node2
                    secondnode = node1
                #print("firstnode")
                #print(firstnode)
                #print("secondnode")
                #print (secondnode)
                if firstnode not in clumpiness:
                    clumpiness[firstnode]={}
                if secondnode not in clumpiness[firstnode]:
                    try:
                        shortest_path = nx.dijkstra_path(self.G,firstnode,secondnode, weight = "weight")
                        shortest_path_len = len(shortest_path)
                    except(nx.NetworkXNoPath):
                        shortest_path_len =self.infinity 
                    clumpiness[firstnode][secondnode] = shortest_path_len
        for node1,node2_dict in clumpiness.items():
            for node2,clump in node2_dict.items():
                if node1 not in hops:
                    hops[node1] = []
                if node2 not in hops:
                    hops[node2] = []
                hops[node1].append(clump)
                hops[node2].append(clump)
        for node,clumplist in hops.items():
            if len(clumplist) > n:
                hoprank[node] = np.mean(clumplist)
        return hoprank

 
    def compute_clumpiness3(self, compute_hoprank = True):
        #Just sample 
        n=1
        hoprank = {}
        probs_by_lens = {}
        clumpiness = {}
        clump = 0.
        count = 0
        for node1 in self.G.nodes:
            #for each agent choose a random n agents
            nodes = random.sample(self.G.nodes,n+1)
            if node1 in nodes:
                nodes.remove(node1)
            else:
                del(nodes[0])
            for node2 in nodes:
                if node1 < node2:
                    firstnode = node1
                    secondnode = node2
                else:
                    firstnode = node2
                    secondnode = node1
                #print("firstnode")
                #print(firstnode)
                #print("secondnode")
                #print (secondnode)
                if firstnode not in clumpiness:
                    clumpiness[firstnode]={}
                    probs_by_lens[firstnode] = {}
                if secondnode not in clumpiness[firstnode]:
                    pl= self.probabilities_by_individual_lengths(firstnode,secondnode)
                    clump += self.clumpiness_given_lengths (pl)
                    count += 1
                    if compute_hoprank:
                        probs_by_lens[firstnode][secondnode] = pl
        clump/=count

        if compute_hoprank:
            for node in self.G.nodes:
                pl = self.probabilities_by_grouped_lengths(node,probs_by_lens)
                hoprank[node] = self.clumpiness_given_lengths(pl)
            
        return (clump,hoprank)
                  
        
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
    #work_building_capacity = 20
    #office_capacity = 10
    #work_building_occupacy_rate = 0.5
    #appartment_building_capacity = 20
    #appartment_capacity = 5
    #appartment_building_occupacy_rate = 0.5
    #school_capacity = 50
    #classroom_capacity = 20
    #school_occupacy_rate = 0.5

    work_building_capacity = 3
    office_capacity = 20
    work_building_occupacy_rate = 0.5
    appartment_building_capacity = 20
    appartment_capacity = 5
    appartment_building_occupacy_rate = 0.5
    school_capacity = 1
    classroom_capacity = 30
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

    # Loose scenario
    work_building_capacity = 3
    office_capacity = 20
    work_building_occupacy_rate = 0.5
    appartment_building_capacity = 20
    appartment_capacity = 5
    appartment_building_occupacy_rate = 0.5
    school_capacity = 1
    classroom_capacity = 30
    school_occupacy_rate = 0.5

   
    #Tight scenario 
    #work_building_capacity = 70
    #office_capacity =3 
    #work_building_occupacy_rate = 1.0 
    #appartment_building_capacity = 20
    #appartment_capacity = 5
    #appartment_building_occupacy_rate = 0.5
    #school_capacity = 70
    #classroom_capacity = 3
    #school_occupacy_rate = 1.0

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
    agents_per_school_district = math.ceil((0.33 *population_size) /len(school_home_list))
    agents_per_work_district = math.ceil((0.65*population_size)/len(work_home_list))

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
    for i in range(1): #20
        blobnum = np.random.randint(get_parameters().params['num_communities'])
        hrf.infect_blob(blobnum)
        hrf.infected_blobs.append(blobnum)
