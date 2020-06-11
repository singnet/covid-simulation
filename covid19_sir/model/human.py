import uuid
import numpy as np
from enum import Enum, auto

from model.base import AgentBase, InfectionStatus, DiseaseSeverity, flip_coin, normal_cap, roulette_selection, get_parameters

def human_unique_id():
    return uuid.uuid1()

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

class IndividualProperties:
    base_health = 1.0

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
        self.properties = IndividualProperties()
        self.initialize_individual_properties()
        self.infection_days_count = 0
        self.infection_latency = 0
        self.infection_incubation = 0
        self.infection_duration = 0
        self.infection_status = InfectionStatus.SUSCEPTIBLE
        self.hospitalized = False
        if self.is_worker(): self.setup_work_info()
        self.current_health = self.properties.base_health
        self.parameter_changed()

    def initialize_individual_properties(self):
        pass

    def parameter_changed(self):
        self.mask_user = flip_coin(get_parameters().get('mask_user_rate'))
        self.isolation_cheater = flip_coin(get_parameters().get('isolation_cheater_rate'))
        self.immune = flip_coin(get_parameters().get('imune_rate'))
        if flip_coin(get_parameters().get('weareable_adoption_rate')):
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
            mean = get_parameters().get('latency_period_mean')
            stdev = get_parameters().get('latency_period_stdev')
            self.infection_latency = np.random.normal(mean, stdev) - self.early_symptom_detection
            if self.infection_latency < 1.0:
                self.infection_latency = 1.0
            mean = get_parameters().get('incubation_period_mean')
            stdev = get_parameters().get('incubation_period_stdev')
            self.infection_incubation = np.random.normal(mean, stdev)
            if self.infection_incubation <= self.infection_latency:
                self.infection_incubation = self.infection_latency + 1
            mean = get_parameters().get('disease_period_mean')
            stdev = get_parameters().get('disease_period_stdev')
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
    def initialize_individual_properties(self):
      self.properties.base_health = normal_cap(0.8, 0.2, 0.0, 1.0)
    
class Toddler(Human):
    def initialize_individual_properties(self):
      self.properties.base_health = normal_cap(0.8, 0.2, 0.0, 1.0)
    
class K12Student(Human):
    def initialize_individual_properties(self):
      self.properties.base_health = normal_cap(0.8, 0.2, 0.0, 1.0)
    
class Adult(Human):
    def initialize_individual_properties(self):
      self.properties.base_health = normal_cap(0.8, 0.2, 0.0, 1.0)
    
class Elder(Human):
    def initialize_individual_properties(self):
      self.properties.base_health = normal_cap(0.8, 0.2, 0.0, 1.0)
