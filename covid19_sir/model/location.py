import math
import numpy as np

from model.base import AgentBase, Human, flip_coin, SimulationParameters, get_parameters

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
        return get_parameters().get(key)

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

class Building(Location):
    def __init__(self, unique_id, covid_model, size, **kwargs):
        super().__init__(unique_id, covid_model, size)
        self.apartments = []
        self.fun_spots = []
        self.offices = []

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

