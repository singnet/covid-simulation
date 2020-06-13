import math
import numpy as np

from model.base import AgentBase, flip_coin, SimulationParameters, get_parameters
from model.human import Human

class Location(AgentBase):
    def __init__(self, covid_model):
        super().__init__(unique_id(), covid_model)
        self.custom_parameters = {}
        self.humans = []
        self.locations = []

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

    def push_human(human):
        humans.append(human)

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

class SimpleLocation(Location, size):
    def __init__(self, unique_id, covid_model):
        super().__init__(unique_id, covid_model)
        for i in range(size):
            self.push_human(Human.factory(covid_model))

class BuildingUnit(Location):
    def __init__(self, building, capacity, covid_model, **kwargs):
        super().__init__(unique_id, covid_model)
        self.set_custom_parameters([('contagion_probability', 0.9)], kwargs)
        self.building = building
        self.capacity = capacity
        self.allocation = []

    def allocate(human_id):
        assert len(self.allocation) < self.capacity
        self.allocation.append(human_id)
        self.building.allocation[human_id] = self

class HomogeneousBuilding(Location):
    def __init__(self, building_capacity, unit_capacity, covid_model, **kwargs):
        super().__init__(unique_id, covid_model)
        self.unit_args = kwargs
        self.capacity = building_capacity
        self.unit_capacity = unit_capacity
        self.allocation = {}
        self.units = []

    def get_unit(human_id):
        return self.allocation[human_id]

    def allocate(human_id):
        success = False
        if len(units[-1]) < self.unit_capacity:
            self.allocation[human_id] = units[-1]
            success = True
        else:
            new_unit = get_empty_unit()
            if new_unit is not None: 
                new_unit.allocate(human_id)
                success = True
        return success

    def get_empty_unit():
        new_unity = None
        if len(units) < self.capacity:
            new_unit = BuildingUnit(self, self.unit_capacity, self.covid_model, self.unit_args)
            units.append(new_unit)
        return new_unit

class House(Location):
    def __init__(self, unique_id, covid_model, **kwargs):
        super().__init__(unique_id, covid_model)
        self.set_custom_parameters([('contagion_probability', 0.9)], kwargs)

class Apartment(Location):
    def __init__(self, unique_id, covid_model, **kwargs):
        super().__init__(unique_id, covid_model)
        self.set_custom_parameters([('contagion_probability', 0.9)], kwargs)

class Office(Location):
    def __init__(self, unique_id, covid_model, **kwargs):
        super().__init__(unique_id, covid_model)
        self.set_custom_parameters([('contagion_probability', 0.7)], kwargs)

class Shop(Location):
    def __init__(self, unique_id, covid_model, **kwargs):
        super().__init__(unique_id, covid_model)
        self.set_custom_parameters([('contagion_probability', 0.6)], kwargs)

class Factory(Location):
    def __init__(self, unique_id, covid_model, **kwargs):
        super().__init__(unique_id, covid_model)
        self.set_custom_parameters([('contagion_probability', 0.6)], kwargs)

class FunGatheringSpot(Location):
    def __init__(self, unique_id, covid_model, **kwargs):
        super().__init__(unique_id, covid_model)
        self.set_custom_parameters([('contagion_probability', 0.2)], kwargs)

class Hospital(Location):
    def __init__(self, unique_id, covid_model, **kwargs):
        super().__init__(unique_id, covid_model)
        self.set_custom_parameters([('contagion_probability', 0.7)], kwargs)
