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

def roulette_selection(v, w):
    assert len(v) == len(w)
    r = np.random.random()
    for i in range(len(w)):
        if r <= w[i]: return v[i]
    return v[len(v) - 1]

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
    for key in kwargs:
        parameters.params[key] = kwargs.get(key)
    
class SimulationStatus:
    def __init__(self):
        self.infected_people = []
        self.non_infected_people = []
        self.dead_people = []
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

class WorkClasses(Enum):
    OFFICE = auto()
    HOUSEBOND = auto()
    FACTORY = auto()
    RETAIL = auto()
    ESSENTIAL = auto()
    TRANSPORTATION = auto()

class WorkInfo:
    can_work_from_home = False
    meet_non_coworkers_at_work = False
    essential_worker = False
    fixed_work_location = False
    house_bound_worker = False
    earnings = 0.0

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
        self.params['daily_interaction_count'] = kwargs.get("daily_interaction_count", 5)
        self.params['contagion_probability'] = kwargs.get("contagion_probability", 0.9)
        self.params['asymptomatic_isolation_rate'] = kwargs.get("asymptomatic_isolation_rate", 0.0)
        self.params['symptomatic_isolation_rate'] = kwargs.get("symptomatic_isolation_rate", 0.0)

    def get(self, key):
        return self.params[key]

    def set(self, key, value):
        self.params[key] = value

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
        if self.is_worker(): self.setup_work_info()
        self.parameter_changed()

    def parameter_changed(self):
        self.mask_user = flip_coin(parameters.get('mask_user_rate'))
        self.isolation_cheater = flip_coin(parameters.get('isolation_cheater_rate'))
        self.immune = flip_coin(parameters.get('imune_rate'))
        if flip_coin(parameters.get('weareable_adoption_rate')):
            self.early_symptom_detection = 1 # number of days
        else:
            self.early_symptom_detection = 0
        
    def infect(self, index):
        if not self.immune:
            self.covid_model.global_count.non_infected_people.pop(index)
            self.covid_model.global_count.infected_people.append(self)
            self.covid_model.global_count.infected_count += 1
            self.covid_model.global_count.non_infected_count -= 1
            self.covid_model.global_count.susceptible_count -= 1
            self.infection_status = InfectionStatus.INFECTED
            self.disease_severity = DiseaseSeverity.ASYMPTOMATIC
            self.covid_model.global_count.asymptomatic_count += 1
            mean = parameters.get('latency_period_mean')
            stdev = parameters.get('latency_period_stdev')
            self.infection_latency = np.random.normal(mean, stdev) - self.early_symptom_detection
            if self.infection_latency < 1.0:
                self.infection_latency = 1.0
            mean = parameters.get('incubation_period_mean')
            stdev = parameters.get('incubation_period_stdev')
            self.infection_incubation = np.random.normal(mean, stdev)
            if self.infection_incubation <= self.infection_latency:
                self.infection_incubation = self.infection_latency + 1
            mean = parameters.get('disease_period_mean')
            stdev = parameters.get('disease_period_stdev')
            self.infection_duration = np.random.normal(mean, stdev)
            if self.infection_duration < (self.infection_incubation + 7):
                self.infection_duration = self.infection_incubation + 7

    def recover(self):
        self.covid_model.global_count.recovered_count += 1
        if self.disease_severity == DiseaseSeverity.MODERATE:
            self.covid_model.global_count.moderate_severity_count -= 1
        elif self.disease_severity == DiseaseSeverity.HIGH:
            self.covid_model.global_count.high_severity_count -= 1
        self.covid_model.global_count.infected_people.remove(self)
        self.covid_model.global_count.infected_count -= 1
        self.covid_model.global_count.non_infected_people.append(self)
        if self.hospitalized:
            self.covid_model.global_count.total_hospitalized -= 1
            self.hospitalized = False
        self.infection_status == InfectionStatus.RECOVERED
        self.disease_severity == DiseaseSeverity.ASYMPTOMATIC
        self.covid_model.global_count.symptomatic_count -= 1
        self.covid_model.global_count.asymptomatic_count += 1
        self.immune = True

    def die(self):
        self.covid_model.global_count.symptomatic_count -= 1
        self.disease_severity = DiseaseSeverity.DEATH
        self.covid_model.global_count.high_severity_count -= 1
        self.covid_model.global_count.infected_count -= 1
        self.covid_model.global_count.death_count += 1
        self.covid_model.global_count.infected_people.remove(self)
        self.covid_model.global_count.dead_people.append(self)
        if self.hospitalized:
            self.covid_model.global_count.total_hospitalized -= 1
            self.hospitalized = False

    def disease_evolution(self):
        if self.is_infected():
            self.infection_days_count += 1
            if self.disease_severity == DiseaseSeverity.ASYMPTOMATIC:
                if self.infection_days_count >= self.infection_incubation:
                    self.disease_severity = DiseaseSeverity.LOW
                    self.covid_model.global_count.asymptomatic_count -= 1
                    self.covid_model.global_count.symptomatic_count += 1
            elif self.disease_severity == DiseaseSeverity.LOW:
                if flip_coin(self.moderate_severity_prob):
                    self.disease_severity = DiseaseSeverity.MODERATE
                    self.covid_model.global_count.moderate_severity_count += 1
                    if not self.covid_model.reached_hospitalization_limit():
                        self.covid_model.global_count.total_hospitalized += 1
                        self.hospitalized = True
            elif self.disease_severity == DiseaseSeverity.MODERATE:
                if flip_coin(self.high_severity_prob):
                    self.disease_severity = DiseaseSeverity.HIGH
                    self.covid_model.global_count.moderate_severity_count -= 1
                    self.covid_model.global_count.high_severity_count += 1
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

    def setup_work_info(self):
        classes = [WorkClasses.OFFICE,
                   WorkClasses.HOUSEBOND,
                   WorkClasses.FACTORY,
                   WorkClasses.RETAIL,
                   WorkClasses.ESSENTIAL,
                   WorkClasses.TRANSPORTATION]
        roulette = []
        count = 1
        #TODO change to use some realistic distribution
        for wclass in classes:
            roulette.append(count / len(classes))
            count = count + 1
        selected_class = roulette_selection(classes, roulette)
        
        self.work_info = WorkInfo()

        self.work_info.can_work_from_home = \
            selected_class ==  WorkClasses.OFFICE or \
            selected_class == WorkClasses.HOUSEBOND

        self.work_info.meet_non_coworkers_at_work = \
            selected_class == WorkClasses.RETAIL or \
            selected_class == WorkClasses.ESSENTIAL or \
            selected_class == WorkClasses.TRANSPORTATION
           
        self.work_info.essential_worker = WorkClasses.ESSENTIAL

        self.work_info.fixed_work_location = \
            selected_class == WorkClasses.OFFICE or \
            selected_class == WorkClasses.HOUSEBOND or \
            selected_class == WorkClasses.FACTORY or \
            selected_class == WorkClasses.RETAIL or \
            selected_class == WorkClasses.ESSENTIAL

        self.work_info.house_bound_worker = WorkClasses.HOUSEBOND

        self.work_info.earnings = 0.0

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
    def __init__(self, unique_id, covid_model, size):
        super().__init__(unique_id, covid_model)
        self.size = size
        self.covid_model = covid_model
        self.custom_parameters = {}
        count = 0
        for i in range(size):
            human = Human.factory(covid_model, self)
            self.covid_model.global_count.non_infected_people.append(human)
            self.covid_model.global_count.non_infected_count += 1
            if human.immune:
                self.covid_model.global_count.immune_count += 1
            else:
                self.covid_model.global_count.susceptible_count += 1
            if not flip_coin(self.get_parameter('initial_infection_rate')):
                count += 1
            else:
                self.covid_model.global_count.non_infected_people[count].infect(count)

    def get_parameter(self, key):
        if key in self.custom_parameters: return self.custom_parameters[key]
        return parameters.get(key)

    def set_custom_parameters(self, s, args):
        for key in args:
            # Only parameters in s (defined in constructor of super) can
            # be overwritten
            check = False
            for k, v in s: 
                if (k == key): check = True
            assert(check)
        self.custom_parameters = {}
        for key, value in s:
            self.custom_parameters[key] = args.get(key, value)

    def step(self):
        self.disease_evolution()
        if self.covid_model.global_count.susceptible_count < 1:
            return
        infections_count = 0.0
        ics = 1.0 - self.get_parameter('isolation_cheating_severity')
        p = self.get_parameter('daily_interaction_count') * self.get_parameter('contagion_probability')
        me = 1.0 - pow(self.get_parameter('mask_efficacy'), self.get_parameter('me_attenuation'))
        for human in self.covid_model.global_count.infected_people:
            if human.is_contagious():
                if human.is_symptomatic():
                    if human.isolation_cheater:
                        p *= (1.0 - (self.get_parameter('symptomatic_isolation_rate') * ics))
                    else:
                        p *= (1.0 - self.get_parameter('symptomatic_isolation_rate'))
                else:
                    if human.isolation_cheater:
                        p *= (1.0 - (self.get_parameter('asymptomatic_isolation_rate') * ics))
                    else:
                        p *= (1.0 - self.get_parameter('asymptomatic_isolation_rate'))
                if human.mask_user:
                    p *= me
            infections_count += p
        targets = (1 - (self.get_parameter('asymptomatic_isolation_rate') * ics) ) * self.covid_model.global_count.non_infected_count + (1 - (self.get_parameter('asymptomatic_isolation_rate') * ics) ) * self.covid_model.global_count.asymptomatic_count + (1 - (self.get_parameter('symptomatic_isolation_rate') * ics) ) * self.covid_model.global_count.symptomatic_count
        for i in range(int(math.ceil(infections_count))):
            if self.covid_model.global_count.susceptible_count <= 0:
                break
            if targets > 0:
                selected_index = np.random.random_integers(0, targets - 1)
            else:
                selected_index = self.covid_model.global_count.non_infected_count
            if selected_index < self.covid_model.global_count.non_infected_count:
                selected = self.covid_model.global_count.non_infected_people[selected_index]
                if not selected.immune:
                    selected.infect(selected_index)

    def disease_evolution(self):
        for human in self.covid_model.global_count.infected_people:
            human.disease_evolution()

class House(Location):
    def __init__(self, unique_id, covid_model, size, **kwargs):
        super().__init__(unique_id, covid_model, size)
        self.set_custom_parameters([('contagion_probability', 0.9)], kwargs)

class Apartment(Location):
    def __init__(self, unique_id, covid_model, size, **kwargs):
        super().__init__(unique_id, covid_model, size)
        self.set_custom_parameters([('contagion_probability', 0.9)], kwargs)

class Office(Location):
    def __init__(self, unique_id, covid_model, size, **kwargs):
        super().__init__(unique_id, covid_model, size)
        self.set_custom_parameters([('contagion_probability', 0.7)], kwargs)

class Shop(Location):
    def __init__(self, unique_id, covid_model, size, **kwargs):
        super().__init__(unique_id, covid_model, size)
        self.set_custom_parameters([('contagion_probability', 0.6)], kwargs)

class Factory(Location):
    def __init__(self, unique_id, covid_model, size, **kwargs):
        super().__init__(unique_id, covid_model, size)
        self.set_custom_parameters([('contagion_probability', 0.6)], kwargs)

class FunGatheringSpot(Location):
    def __init__(self, unique_id, covid_model, size, **kwargs):
        super().__init__(unique_id, covid_model, size)
        self.set_custom_parameters([('contagion_probability', 0.2)], kwargs)

class Hospital(Location):
    def __init__(self, unique_id, covid_model, size, **kwargs):
        super().__init__(unique_id, covid_model, size)
        self.set_custom_parameters([('contagion_probability', 0.7)], kwargs)

class CovidModel(Model):
    def __init__(self):
        self.global_count = SimulationStatus()
        self.schedule = RandomActivation(self)
        self.locations = []
        self.listeners = []

    def reached_hospitalization_limit(self):
        return (self.global_count.total_hospitalized / self.global_count.total_population) >= parameters.get('hospitalization_capacity')

    def add_listener(self, listener):
        self.listeners.append(listener)

    def add_location(self, location):
        self.schedule.add(location)
        self.locations.append(location)
        self.global_count.total_population += location.size

    def step(self):
        for listener in self.listeners:
            listener.start_cycle(self)
        self.schedule.step()
        for listener in self.listeners:
            listener.end_cycle(self)
