import math
import numpy as np
from enum import Enum, auto

from model.base import (AgentBase, SimulationState, flip_coin, SimulationParameters,
get_parameters, unique_id, random_selection, normal_ci, normal_cap_ci)
from model.utils import RestaurantType
from model.human import Human

class Location(AgentBase):
    def __init__(self, covid_model):
        super().__init__(unique_id(), covid_model)
        self.custom_parameters = {}
        self.humans = []
        self.locations = []
        self.container = None
        self.spreading_rate = get_parameters().get('spreading_rate')

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

    def move_to(self, human, target):
        if human in self.humans:
            self.humans.remove(human)
            target.humans.append(human)

    def step(self):
        pass

    def check_spreading(self, h1, h2):
        if h1.is_contagious() and not h2.is_infected():
            if flip_coin(self.get_parameter('contagion_probability')):
                me = self.get_parameter('mask_efficacy')
                if not h1.is_wearing_mask() or (h1.is_wearing_mask() and not flip_coin(me)):
                    h2.infect()

    def spread_infection(self):
        for h1 in self.humans:
            for h2 in self.humans:
                if h1 != h2:
                    if flip_coin(self.spreading_rate / len(self.humans)):
                        self.check_spreading(h1, h2)

class BuildingUnit(Location):
    def __init__(self, capacity, covid_model, **kwargs):
        super().__init__(covid_model)
        self.set_custom_parameters([\
            ('contagion_probability', 0.0)
        ], kwargs)
        self.capacity = capacity
        self.allocation = []

    def step(self):
        super().step()
        if self.covid_model.current_state == SimulationState.MORNING_AT_HOME or\
           self.covid_model.current_state == SimulationState.MAIN_ACTIVITY:
            self.spread_infection()

class HomogeneousBuilding(Location):
    def __init__(self, building_capacity, covid_model, **kwargs):
        super().__init__(covid_model)
        self.unit_args = kwargs
        self.capacity = building_capacity
        self.allocation = {}
    def get_unit(self, human):
        return self.allocation[human]

class FunGatheringSpot(Location):
    def __init__(self, capacity, covid_model, **kwargs):
        super().__init__(covid_model)
        self.set_custom_parameters([\
            ('contagion_probability', normal_cap_ci(0.0035, 0.005))
        ], kwargs)
        self.capacity = capacity
        self.available = True
    def step(self):
        super().step()
        if self.covid_model.current_state == SimulationState.POST_WORK_ACTIVITY:
            self.spread_infection()

class Restaurant(Location):
    def __init__(self, capacity, restaurant_type, is_outdoor, covid_model, **kwargs):
        super().__init__(covid_model)
        OUTDOOR = True
        INDOOR = False
        # https://docs.google.com/document/d/1imCNXOyoyecfD_sVNmKpmbWVB6xqP-FWlHELAyOg1Vs/edit
        base_fast_food = normal_cap_ci(0.0014, 0.01, 10)
        base_fancy = normal_cap_ci(0.007, 0.02, 10)
        base_bar = normal_cap_ci(0.0174, 0.0796, 10)
        cp = {
            RestaurantType.FAST_FOOD: {
                INDOOR: base_fast_food,
                OUTDOOR: base_fast_food / 2
            },
            RestaurantType.FANCY: {
                INDOOR: base_fancy,
                OUTDOOR: base_fancy / 2
            },
            RestaurantType.BAR: {
                INDOOR: base_bar,
                OUTDOOR: base_bar / 2
            }
        }
        self.capacity = capacity
        self.available = capacity
        self.restaurant_type = restaurant_type
        self.is_outdoor = is_outdoor
        self.set_custom_parameters([\
            ('contagion_probability', cp[restaurant_type][is_outdoor])
        ], kwargs)

    def step(self):
        super().step()
        capacity = self.get_parameter('allowed_restaurant_capacity')
        ci = {}
        # https://docs.google.com/document/d/1imCNXOyoyecfD_sVNmKpmbWVB6xqP-FWlHELAyOg1Vs/edit
        ci[1.0] = (0.19, 1.23)
        ci[0.5] = (0.11, 0.74)
        ci[0.25] = (0.08, 0.50)
        lb, ub = ci[capacity]
        self.spreading_rate = normal_ci(lb, ub, 20)
        if self.covid_model.current_state == SimulationState.POST_WORK_ACTIVITY:
            self.spread_infection()

class District(Location):
    def __init__(self, name, covid_model, **kwargs):
        super().__init__(covid_model)
        self.allocation = {}
        self.name = name

    def get_buildings(self, human):
        if human in self.allocation:
            return self.allocation[human]
        return []

    def get_available_restaurant(self, people_count, outdoor, restaurant_type):
        for location in self.locations:
            if isinstance(location, Restaurant) and \
                location.restaurant_type == restaurant_type and \
                location.is_outdoor == outdoor and \
                (location.available * self.get_parameter('allowed_restaurant_capacity')) >= people_count:
                return location
        return None

    def get_available_gathering_spot(self):
        for location in self.locations:
            if isinstance(location, FunGatheringSpot) and location.available:
                location.available = False
                return location
        return None

    def move_to(self, human, target):
        s = self.get_buildings(human)[0].get_unit(human)
        if isinstance(target, District):
            t = target.get_buildings(human)[0].get_unit(human)
        else:
            t = target
        s.move_to(human, t)

    def move_from(self, human, source):
        t = self.get_buildings(human)[0].get_unit(human)
        if isinstance(source, District):
            s = source.get_buildings(human)[0].get_unit(human)
        else:
            s = source
        s.move_to(human, t)

    def _select(self, building_type, n, same_unit, exclusive):
        count = 0
        while True:
            count += 1
            assert count < (len(self.locations) * 1000) # infinit loop
            building = random_selection(self.locations)
            if not isinstance(building, building_type): continue
            for unit in building.locations:
                if exclusive:
                    if not unit.allocation:
                        return (building, unit)
                else:
                    vacancy = unit.capacity - len(unit.allocation)
                    if vacancy >= n or vacancy == 1 and not same_unit:
                        return (building, unit)

    def _select_different_unit(self, building, invalid_unit):
        for unit in building.locations:
            if unit != invalid_unit and len(unit.allocation) < unit.capacity:
                return unit
        assert False

    def allocate(self, humans, same_building=False, same_unit=False, exclusive=False, building_type=HomogeneousBuilding):
        assert (exclusive and same_unit and same_building) or\
               (not exclusive and same_unit and same_building) or\
               (not exclusive and not same_unit and same_building) or\
               (not exclusive and not same_unit and not same_building)
        building = None
        unit = None
        for human in humans:
            if building is None or (building is not None and not same_building):
                building, unit = self._select(building_type, len(humans), same_unit, exclusive)
            else:
                if not same_unit:
                    unit = self._select_different_unit(building, unit)
            if human not in self.allocation:
                self.allocation[human] = []
            self.allocation[human].append(building)
            building.allocation[human] = unit
            unit.allocation.append(human)

    def __repr__(self):
        txt = f"\n{self.name} district with {len(self.locations)} Buildings\n"
        district_total_humans = 0
        for building in self.locations:
            txt = txt + f"{type(building).__name__}: {building.capacity} units (each with capacity for {building.locations[0].capacity} people.) "
            sum_allocated = 0
            total_allocated = 0
            for unit in building.locations:
                if unit.allocation:
                    total_allocated += 1
                    sum_allocated += len(unit.allocation)
            txt = txt + f"{total_allocated} allocated units with a total of {sum_allocated} people.\n"
            district_total_humans += sum_allocated
        txt = txt + f"Total of {district_total_humans} people allocated in this district.\n"
        return txt

