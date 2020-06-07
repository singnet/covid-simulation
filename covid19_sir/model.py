import math
import uuid
from enum import Enum, auto
from mesa import Agent, Model
from mesa.time import RandomActivation
import numpy as np

def flip_coin(prob):
    if np.random.random() < prob:
        return True
    else:
        return False

def human_unique_id():
    return uuid.uuid1()

def set_parameters(new_parameters):
    global parameters
    parameters = new_parameters

def get_parameters():
    global parameters
    return parameters

def change_parameters(**kwargs):
    global parameters
    #TODO Set only parameters passed in kwargs
    parameters.mask_user_rate = kwargs.get("mask_user_rate", parameters.mask_user_rate)
    parameters.mask_efficacy = kwargs.get("mask_efficacy", parameters.mask_efficacy)
    parameters.isolation_cheater_rate = kwargs.get("isolation_cheater_rate", parameters.isolation_cheater_rate)
    parameters.isolation_cheating_severity = kwargs.get("isolation_cheating_severity", parameters.isolation_cheating_severity)
    parameters.imune_rate = kwargs.get("imune_rate", parameters.imune_rate)
    parameters.initial_infection_rate = kwargs.get("initial_infection_rate", parameters.initial_infection_rate)
    parameters.hospitalization_capacity = kwargs.get("hospitalization_capacity", parameters.hospitalization_capacity)
    parameters.latency_period_mean = kwargs.get("latency_period_mean", parameters.latency_period_mean)
    parameters.latency_period_stdev = kwargs.get("latency_period_stdev", parameters.latency_period_stdev)
    parameters.incubation_period_mean = kwargs.get("incubation_period_mean", parameters.incubation_period_mean)
    parameters.incubation_period_stdev = kwargs.get("incubation_period_stdev", parameters.incubation_period_stdev)
    parameters.disease_period_mean = kwargs.get("disease_period_mean", parameters.disease_period_mean)
    parameters.disease_period_stdev = kwargs.get("disease_period_stdev", parameters.disease_period_stdev)
    parameters.me_attenuation = kwargs.get("me_attenuation", parameters.me_attenuation)
    parameters.weareable_adoption_rate = kwargs.get("weareable_adoption_rate", parameters.weareable_adoption_rate)
    parameters.daily_interaction_count = kwargs.get("daily_interaction_count", parameters.daily_interaction_count)
    parameters.contagion_probability = kwargs.get("contagion_probability", parameters.contagion_probability)
    parameters.asymptomatic_isolation_rate = kwargs.get("asymptomatic_isolation_rate", parameters.asymptomatic_isolation_rate)
    parameters.symptomatic_isolation_rate = kwargs.get("symptomatic_isolation_rate", parameters.symptomatic_isolation_rate)
    
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

class SimulationParameters(object):
    def __init__(self, **kwargs):
        self.mask_user_rate = kwargs.get("mask_user_rate", 0.0)
        self.mask_efficacy = kwargs.get("mask_efficacy", 0.0)
        self.isolation_cheater_rate = kwargs.get("isolation_cheater_rate", 0.0)
        self.isolation_cheating_severity = kwargs.get("isolation_cheating_severity", 0.0)
        self.imune_rate = kwargs.get("imune_rate", 0.05)
        self.initial_infection_rate = kwargs.get("initial_infection_rate", 0.05)
        self.hospitalization_capacity = kwargs.get("hospitalization_capacity", 0.05)
        self.latency_period_mean = kwargs.get("latency_period_mean", 4.0)
        self.latency_period_stdev = kwargs.get("latency_period_stdev", 1.0)
        self.incubation_period_mean = kwargs.get("incubation_period_mean", 7.0)
        self.incubation_period_stdev = kwargs.get("incubation_period_stdev", 2.0)
        self.disease_period_mean = kwargs.get("disease_period_mean", 20.0)
        self.disease_period_stdev = kwargs.get("disease_period_stdev", 5.0)
        self.me_attenuation = kwargs.get("me_attenuation", 1.0)
        self.weareable_adoption_rate = kwargs.get("weareable_adoption_rate", 0.0)
        self.daily_interaction_count = kwargs.get("daily_interaction_count", 5)
        self.contagion_probability = kwargs.get("contagion_probability", 0.9)
        self.asymptomatic_isolation_rate = kwargs.get("asymptomatic_isolation_rate", 0.0)
        self.symptomatic_isolation_rate = kwargs.get("symptomatic_isolation_rate", 0.0)

parameters = None

class AgentBase(Agent):
    # MESA agent
    def __init__(self, unique_id, covid_model):
        super().__init__(unique_id, covid_model)

class Human(AgentBase):

    @staticmethod
    def factory(covid_model, location):
        moderate_severity_probs = [0.001, 0.003, 0.012, 0.032, 0.049, 0.102, 0.166, 0.243, 0.273, 0.273]
        high_severity_probs = [0.05, 0.05, 0.05, 0.05, 0.063, 0.122, 0.274, 0.432, 0.709, 0.709]
        death_probs = [0.002, 0.00006, 0.0003, 0.0008, 0.0015, 0.006, 0.022, 0.051, 0.093, 0.093]
        age = int(np.random.beta(2, 5, 1) * 100)
        index = age // 10
        msp = moderate_severity_probs[index]
        hsp = high_severity_probs[index]
        mfd = flip_coin(death_probs[index])
        if age <= 1: return Infant(covid_model, location, age, msp, hsp, mfd)
        if age <= 4: return Toddler(covid_model, location, age, msp, hsp, mfd)
        if age <= 18: return K12Student(covid_model, location, age, msp, hsp, mfd)
        if age <= 64: return Adult(covid_model, location, age, msp, hsp, mfd)
        return Elder(covid_model, location, age, msp, hsp, mfd)

    def __init__(self, covid_model, location, age, msp, hsp, mfd):
        super().__init__(human_unique_id(), covid_model)
        self.covid_model = covid_model
        self.location = location
        self.age = age
        self.moderate_severity_prob = msp
        self.high_severity_prob = hsp
        self.death_mark = mfd
        self.infection_days_count = 0
        self.infection_latency = 0
        self.infection_incubation = 0
        self.infection_duration = 0
        self.infection_status = InfectionStatus.SUSCEPTIBLE
        self.hospitalized = False
        self.parameter_changed()

    def parameter_changed(self):
        self.mask_user = flip_coin(parameters.mask_user_rate)
        self.isolation_cheater = flip_coin(parameters.isolation_cheater_rate)
        self.immune = flip_coin(parameters.imune_rate)
        if flip_coin(parameters.weareable_adoption_rate):
            self.early_symptom_detection = 1 # number of days
        else:
            self.early_symptom_detection = 0
        
    def infect(self, index):
        if not self.immune:
            self.location.non_infected_people.pop(index)
            self.location.infected_people.append(self)
            self.location.infected_count += 1
            self.location.non_infected_count -= 1
            self.location.susceptible_count -= 1
            self.infection_status = InfectionStatus.INFECTED
            self.disease_severity = DiseaseSeverity.ASYMPTOMATIC
            self.location.asymptomatic_count += 1
            mean = parameters.latency_period_mean
            stdev = parameters.latency_period_stdev
            self.infection_latency = np.random.normal(mean, stdev) - self.early_symptom_detection
            if self.infection_latency < 1.0:
                self.infection_latency = 1.0
            mean = parameters.incubation_period_mean
            stdev = parameters.incubation_period_stdev
            self.infection_incubation = np.random.normal(mean, stdev)
            if self.infection_incubation <= self.infection_latency:
                self.infection_incubation = self.infection_latency + 1
            mean = parameters.disease_period_mean
            stdev = parameters.disease_period_stdev
            self.infection_duration = np.random.normal(mean, stdev)
            if self.infection_duration < (self.infection_incubation + 7):
                self.infection_duration = self.infection_incubation + 7

    def recover(self):
        self.location.recovered_count += 1
        if self.disease_severity == DiseaseSeverity.MODERATE:
            self.location.moderate_severity_count -= 1
        elif self.disease_severity == DiseaseSeverity.HIGH:
            self.location.high_severity_count -= 1
        self.location.infected_people.remove(self)
        self.location.infected_count -= 1
        self.location.non_infected_people.append(self)
        if self.hospitalized:
            self.covid_model.total_hospitalized -= 1
            self.hospitalized = False
        self.infection_status == InfectionStatus.RECOVERED
        self.disease_severity == DiseaseSeverity.ASYMPTOMATIC
        self.location.symptomatic_count -= 1
        self.location.asymptomatic_count += 1
        self.immune = True

    def die(self):
        self.location.symptomatic_count -= 1
        self.disease_severity = DiseaseSeverity.DEATH
        self.location.high_severity_count -= 1
        self.location.infected_count -= 1
        self.location.death_count += 1
        self.location.infected_people.remove(self)
        self.location.dead_people.append(self)
        if self.hospitalized:
            self.covid_model.total_hospitalized -= 1
            self.hospitalized = False

    def disease_evolution(self):
        if self.is_infected():
            self.infection_days_count += 1
            if self.disease_severity == DiseaseSeverity.ASYMPTOMATIC:
                if self.infection_days_count >= self.infection_incubation:
                    self.disease_severity = DiseaseSeverity.LOW
                    self.location.asymptomatic_count -= 1
                    self.location.symptomatic_count += 1
            elif self.disease_severity == DiseaseSeverity.LOW:
                if flip_coin(self.moderate_severity_prob):
                    self.disease_severity = DiseaseSeverity.MODERATE
                    self.location.moderate_severity_count += 1
                    if not self.covid_model.reached_hospitalization_limit():
                        self.covid_model.total_hospitalized += 1
                        self.hospitalized = True
            elif self.disease_severity == DiseaseSeverity.MODERATE:
                if flip_coin(self.high_severity_prob):
                    self.disease_severity = DiseaseSeverity.HIGH
                    self.location.moderate_severity_count -= 1
                    self.location.high_severity_count += 1
                    if not self.hospitalized or self.death_mark:
                        self.die()
            elif self.disease_severity == DiseaseSeverity.HIGH:
                if self.death_mark:
                    self.die()
            if self.disease_severity != DiseaseSeverity.DEATH:
                if self.infection_days_count > self.infection_duration:
                    self.recover()
        
    def is_infected(self):
        return self.infection_status == InfectionStatus.INFECTED

    def is_contagious(self):
        return self.is_infected() and self.infection_days_count >= self.infection_latency
    
    def is_symptomatic(self):
        return self.is_infected() and self.infection_days_count >= self.infection_incubation
    
    def is_worker(self):
        return self.age >= 15 and self.age <= 64

class Infant(Human):
    pass
    
class Toddler(Human):
    pass
    
class K12Student(Human):
    pass
    
class Adult(Human):
    pass
    
class Elder(Human):
    pass
    
class Location(AgentBase):
    def __init__(self, unique_id, covid_model, size, **kwargs):
        super().__init__(unique_id, covid_model)
        self.size = size
        self.covid_model = covid_model
        self.infected_people = []
        self.non_infected_people = []
        self.dead_people = []
        self.infected_count = 0;
        self.non_infected_count = 0;
        self.susceptible_count = 0;
        self.immune_count = 0;
        self.recovered_count = 0;
        self.moderate_severity_count = 0;
        self.high_severity_count = 0;
        self.death_count = 0;
        self.symptomatic_count = 0;
        self.asymptomatic_count = 0;
        count = 0
        for i in range(size):
            human = Human.factory(covid_model, self)
            self.non_infected_people.append(human)
            self.non_infected_count += 1
            if human.immune:
                self.immune_count += 1
            else:
                self.susceptible_count += 1
            if not flip_coin(parameters.initial_infection_rate):
                count += 1
            else:
                self.non_infected_people[count].infect(count)

    def step(self):
        self.disease_evolution()
        if self.susceptible_count < 1:
            return
        infections_count = 0.0
        ics = 1.0 - parameters.isolation_cheating_severity
        p = parameters.daily_interaction_count * parameters.contagion_probability
        me = 1.0 - pow(parameters.mask_efficacy, parameters.me_attenuation)
        for human in self.infected_people:
            if human.is_contagious():
                if human.is_symptomatic():
                    if human.isolation_cheater:
                        p *= (1.0 - (parameters.symptomatic_isolation_rate * ics))
                    else:
                        p *= (1.0 - parameters.symptomatic_isolation_rate)
                else:
                    if human.isolation_cheater:
                        p *= (1.0 - (parameters.asymptomatic_isolation_rate * ics))
                    else:
                        p *= (1.0 - parameters.asymptomatic_isolation_rate)
                if human.mask_user:
                    p *= me
            infections_count += p
        targets = (1 - (parameters.asymptomatic_isolation_rate * ics) ) * self.non_infected_count + (1 - (parameters.asymptomatic_isolation_rate * ics) ) * self.asymptomatic_count + (1 - (parameters.symptomatic_isolation_rate * ics) ) * self.symptomatic_count
        for i in range(int(math.ceil(infections_count))):
            if self.susceptible_count <= 0:
                break
            if targets > 0:
                selected_index = np.random.random_integers(0, targets - 1)
            else:
                selected_index = self.non_infected_count
            if selected_index < self.non_infected_count:
                selected = self.non_infected_people[selected_index]
                if not selected.immune:
                    selected.infect(selected_index)

    def disease_evolution(self):
        for human in self.infected_people:
            human.disease_evolution()

class CovidModel(Model):
    def __init__(self):
        self.schedule = RandomActivation(self)
        self.locations = []
        self.listeners = []
        self.total_population = 0
        self.total_hospitalized = 0

    def reached_hospitalization_limit(self):
        return (self.total_hospitalized / self.total_population) >= parameters.hospitalization_capacity

    def add_listener(self, listener):
        self.listeners.append(listener)

    def add_location(self, location):
        self.schedule.add(location)
        self.locations.append(location)
        self.total_population += location.size

    def step(self):
        for listener in self.listeners:
            listener.start_cycle(self)
        self.schedule.step()
        for listener in self.listeners:
            listener.end_cycle(self)
