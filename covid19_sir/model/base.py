import uuid
import numpy as np
from enum import Enum, auto
from mesa import Agent, Model
from mesa.time import RandomActivation

def flip_coin(prob):
    if np.random.random() < prob:
        return True
    else:
        return False

def random_selection(v):
    return(v[np.random.random_integers(0, len(v) - 1)])

def build_roulette(w):
    r = []
    s = 0
    for v in w: 
        s += v
    acc = 0
    for v in w:
        r.append(acc + (v / s))
        acc += v / s

def roulette_selection(v, w):
    assert len(v) == len(w)
    r = np.random.random()
    for i in range(len(w)):
        if r <= w[i]: return v[i]
    return v[len(v) - 1]

def normal_cap(mean, stdev, lower_bound, upper_bound):
    r = np.random.normal(mean, stdev)
    if r < lower_bound: r = lower_bound
    if r > upper_bound: r = upper_bound
    return r

def unique_id():
    return uuid.uuid1()

def set_parameters(new_parameters):
    global parameters
    parameters = new_parameters

def get_parameters():
    global parameters
    return parameters

def change_parameters(**kwargs):
    global parameters
    for key in kwargs:
        parameters.params[key] = kwargs.get(key)
    
class SimulationStatus:
    def __init__(self):
        self.infected_count = 0
        self.non_infected_count = 0
        self.susceptible_count = 0
        self.immune_count = 0
        self.recovered_count = 0
        self.moderate_severity_count = 0
        self.high_severity_count = 0
        self.death_count = 0
        self.symptomatic_count = 0
        self.asymptomatic_count = 0
        self.total_hospitalized = 0
        self.total_population = 0
        self.work_population = 0
        self.total_income = 0.0
    
class InfectionStatus(Enum):
    SUSCEPTIBLE = auto()
    INFECTED = auto()
    RECOVERED = auto()

class DiseaseSeverity(Enum):
    ASYMPTOMATIC = auto()
    LOW = auto() # No hospitalization
    MODERATE = auto() # hospitalization
    HIGH = auto() # hospitalization in ICU
    DEATH = auto()

class SimulationState(Enum):
    COMMUTING_TO_MAIN_ACTIVITY = auto()
    COMMUTING_TO_HOME = auto()
    MAIN_ACTIVITY = auto()
    MORNING_AT_HOME = auto()
    EVENING_AT_HOME = auto()

class SimulationParameters:
    def __init__(self, **kwargs):
        self.params = {}
        self.params['mask_user_rate'] = kwargs.get("mask_user_rate", 0.0)
        self.params['mask_efficacy'] = kwargs.get("mask_efficacy", 0.0)
        self.params['isolation_cheater_rate'] = kwargs.get("isolation_cheater_rate", 0.0)
        self.params['isolation_cheating_severity'] = kwargs.get("isolation_cheating_severity", 0.0)
        self.params['imune_rate'] = kwargs.get("imune_rate", 0.05)
        self.params['initial_infection_rate'] = kwargs.get("initial_infection_rate", 0.05)
        self.params['hospitalization_capacity'] = kwargs.get("hospitalization_capacity", 0.05)
        self.params['latency_period_mean'] = kwargs.get("latency_period_mean", 4.0)
        self.params['latency_period_stdev'] = kwargs.get("latency_period_stdev", 1.0)
        self.params['incubation_period_mean'] = kwargs.get("incubation_period_mean", 7.0)
        self.params['incubation_period_stdev'] = kwargs.get("incubation_period_stdev", 2.0)
        self.params['disease_period_mean'] = kwargs.get("disease_period_mean", 20.0)
        self.params['disease_period_stdev'] = kwargs.get("disease_period_stdev", 5.0)
        self.params['me_attenuation'] = kwargs.get("me_attenuation", 1.0)
        self.params['weareable_adoption_rate'] = kwargs.get("weareable_adoption_rate", 0.0)
        self.params['contagion_probability'] = kwargs.get("contagion_probability", 0.9)
        self.params['spreading_rate'] = kwargs.get("spreading_rate", 0.0)
        self.params['asymptomatic_isolation_rate'] = kwargs.get("asymptomatic_isolation_rate", 0.0)
        self.params['symptomatic_isolation_rate'] = kwargs.get("symptomatic_isolation_rate", 0.0)
        self.params['asymptomatic_contagion_probability'] = kwargs.get("asymptomatic_contagion_probability", 0.1)

    def get(self, key):
        return self.params[key]

    def set(self, key, value):
        self.params[key] = value

parameters = None

class AgentBase(Agent):
    # MESA agent
    def __init__(self, unique_id, covid_model):
        super().__init__(unique_id, covid_model)
        self.id = unique_id
        self.covid_model = covid_model
        covid_model.schedule.add(self)

    def __repr__(self):
        return f'<{type(self).__name__} {self.id}>'

class CovidModel(Model):
    def __init__(self):
        self.all_agents = []
        self.global_count = SimulationStatus()
        self.schedule = RandomActivation(self)
        self.listeners = []
        self.current_state = SimulationState.MORNING_AT_HOME
        self.next_state = {
            SimulationState.MORNING_AT_HOME: SimulationState.COMMUTING_TO_MAIN_ACTIVITY,
            SimulationState.COMMUTING_TO_MAIN_ACTIVITY: SimulationState.MAIN_ACTIVITY,
            SimulationState.MAIN_ACTIVITY: SimulationState.COMMUTING_TO_HOME,
            SimulationState.COMMUTING_TO_HOME: SimulationState.EVENING_AT_HOME,
            SimulationState.EVENING_AT_HOME: SimulationState.MORNING_AT_HOME
        }
        

    def reached_hospitalization_limit(self):
        return (self.global_count.total_hospitalized / self.global_count.total_population) >= parameters.get('hospitalization_capacity')

    def add_listener(self, listener):
        self.listeners.append(listener)

    def step(self):
        assert self.current_state == SimulationState.MORNING_AT_HOME
        for listener in self.listeners:
            listener.start_cycle(self)

        self.global_count.total_income = 0.0
        flag = False
        while not flag:
            self.schedule.step()
            self.current_state = self.next_state[self.current_state]
            if self.current_state == SimulationState.MORNING_AT_HOME:
                flag = True
        
        for listener in self.listeners:
            listener.end_cycle(self)
