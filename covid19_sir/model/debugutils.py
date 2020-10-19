import pandas as pd
from model.base import logger
from model.human import Human, Adult, K12Student, Toddler, Infant, Elder
from model.location import Location, District, Restaurant
from model.utils import InfectionStatus


class DebugUtils:
    def __init__(self, model):
        self.humans = []
        self.adults = []
        self.k12students = []
        self.toddlers = []
        self.infants = []
        self.elders = []
        self.locations = []
        self.districts = []
        self.restaurants = []
        self.model = model
        self.human_status = {}
        self.count_school = 0
        self.count_home = 0
        self.count_restaurant = 0
        self.count_work = 0
        self.max_hospitalized = 0
        self.max_icu = 0
        self._populate()

    def print_world(self):
        for district in self.districts:
            print("{} District:".format(district.name))
            for i, building in enumerate(district.locations):
                num_humans_in_rooms = [len(room.humans) for room in building.locations]
                print("{0}{1}".format(type(building).__name__, i))
                print(num_humans_in_rooms)
            for i, building in enumerate(district.locations):
                for j, room in enumerate(building.locations):
                    humans_in_rooms = [human.unique_id for human in room.humans]
                    print("{0}{1}-room{2}".format(type(building).__name__, i, j))
                    print(humans_in_rooms)
                humans_in_building = [human.unique_id for human in building.humans]
                print("{0}{1}".format(type(building).__name__, i))
                print(humans_in_building)

    def update_human_status(self):
        for human in self.humans:
            if human not in self.human_status:
                self.human_status[human] = {}
            self.human_status[human][self.model.global_count.day_count] = human.info()

    def update_infection_status(self):
        self.count_school = 0
        self.count_home = 0
        self.count_restaurant = 0
        self.count_work = 0
        for human in self.model.global_count.infection_info:
            location = self.model.global_count.infection_info[human]
            if 'School' in location.strid:
                self.count_school += 1
            elif 'Home' in location.strid:
                self.count_home += 1
            elif 'Restaurant' in location.strid:
                self.count_restaurant += 1
            elif 'Work' in location.strid:
                self.count_work += 1
            else:
                logger.warning(f"Unexpected infection location: {location}")

    def update_hospitalization_status(self):
        if self.model.global_count.total_hospitalized > self.max_hospitalized:
            self.max_hospitalized = self.model.global_count.total_hospitalized
        if self.model.global_count.high_severity_count > self.max_icu:
            self.max_icu = self.model.global_count.high_severity_count

    def print_infection_status(self):
        self.update_infection_status()
        print(f"School: {self.count_school}")
        print(f"Home: {self.count_home}")
        print(f"Restaurant: {self.count_restaurant}")
        print(f"Work: {self.count_work}")
        print(f"Total: {self.count_school + self.count_home + self.count_restaurant + self.count_work}")

    def get_R0_stats(self):
        raw_data = []
        for human in self.humans:
            if human.infection_status == InfectionStatus.INFECTED or \
               human.infection_status == InfectionStatus.RECOVERED:
                raw_data.append(human.count_infected_humans)
        return pd.DataFrame({
            #TODO Add series for different location types
            'infections': raw_data
        })

    def get_age_group_stats(self):
        count = [0] * 10
        deaths = [0] * 10
        infected = [0] * 10 # Snapshot. This not an accumulative measure.
        recovered = [0] * 10
        death_mark = [0] * 10
        hospitalized = [0] * 10
        icu = [0] * 10
        for human in self.humans:
            index = human.age // 10
            count[index] += 1
            if human.is_dead:
                deaths[index] += 1
            if human.infection_status == InfectionStatus.INFECTED:
                infected[index] += 1
            if human.infection_status == InfectionStatus.RECOVERED:
                recovered[index] += 1
            if human.death_mark:
                death_mark[index] += 1
            if human.has_been_hospitalized:
                hospitalized[index] += 1
            if human.has_been_icu:
                icu[index] += 1
        return pd.DataFrame({
            'count': count,
            'deaths': deaths,
            'infected': infected, # Snapshot. This not an accumulative measure.
            'recovered': recovered,
            'death_mark': death_mark,
            'hospitalized': hospitalized,
            'icu': icu
        })

    def start_cycle(self, model):
        pass

    def end_cycle(self, model):
        self.update_human_status()
        self.update_infection_status()
        self.update_hospitalization_status()

    def _populate(self):
        for agent in self.model.agents:
            if isinstance(agent, Human):
                self.humans.append(agent)
            if isinstance(agent, Adult):
                self.adults.append(agent)
            if isinstance(agent, K12Student):
                self.k12students.append(agent)
            if isinstance(agent, Toddler):
                self.toddlers.append(agent)
            if isinstance(agent, Infant):
                self.infants.append(agent)
            if isinstance(agent, Elder):
                self.elders.append(agent)
            if isinstance(agent, Location):
                self.locations.append(agent)
            if isinstance(agent, District):
                self.districts.append(agent)
            if isinstance(agent, Restaurant):
                self.restaurants.append(agent)
