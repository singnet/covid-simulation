import math
import numpy as np

from model.base import AgentBase, SimulationState, flip_coin, SimulationParameters, get_parameters, unique_id, random_selection
from model.human import Human

class Location(AgentBase):
    def __init__(self, covid_model):
        super().__init__(unique_id(), covid_model)
        self.custom_parameters = {}
        self.humans = []
        self.locations = []
        self.container = None

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
                    if flip_coin(self.get_parameter('spreading_rate')):
                        self.check_spreading(h1, h2)

class BuildingUnit(Location):
    def __init__(self, capacity, covid_model, **kwargs):
        super().__init__(covid_model)
        self.set_custom_parameters([\
            ('contagion_probability', 0.9),\
            ('spreading_rate', 0.5)\
        ], kwargs)
        self.capacity = capacity
        self.allocation = []

    def step(self):
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

class House(Location):
    def __init__(self, covid_model, **kwargs):
        super().__init__(covid_model)
        self.set_custom_parameters([('contagion_probability', 0.9)], kwargs)

class Shop(Location):
    def __init__(self, covid_model, **kwargs):
        super().__init__(covid_model)
        self.set_custom_parameters([('contagion_probability', 0.6)], kwargs)

class Factory(Location):
    def __init__(self, covid_model, **kwargs):
        super().__init__(covid_model)
        self.set_custom_parameters([('contagion_probability', 0.6)], kwargs)

class FunGatheringSpot(Location):
    def __init__(self, capacity, covid_model, **kwargs):
        super().__init__(covid_model)
        self.set_custom_parameters([\
            ('contagion_probability', 0.9),\
            ('spreading_rate', 0.5)\
        ], kwargs)
        self.capacity = capacity
        self.available = True
    def step(self):
        if self.covid_model.current_state == SimulationState.MAIN_ACTIVITY:
            #if self.humans: print(f"SPREADING {len(self.humans)}")
            self.spread_infection()

class Hospital(Location):
    def __init__(self, covid_model, **kwargs):
        super().__init__(covid_model)
        self.set_custom_parameters([('contagion_probability', 0.7)], kwargs)

class District(Location):
    def __init__(self, name, covid_model, **kwargs):
        super().__init__(covid_model)
        self.allocation = {}
        self.name = name

    def get_buildings(self, human):
        if human in self.allocation:
            return self.allocation[human]
        return []

    def get_available_gathering_spot(self):
        for location in self.locations:
            if isinstance(location, FunGatheringSpot) and location.available:
                location.available = False
                #print("GATHERING SPOT AVAILABLE")
                return location
        #print("NO GATHERING SPOT")
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

