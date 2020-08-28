from model.human import Human, Adult, K12Student, Toddler, Infant, Elder
from model.location import Location, District, Restaurant

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
        self._populate(model)

    def print_world(self):
        for district in self.districts:
            print("{} District:".format(district.name))
            for i,building in enumerate(district.locations):
                num_humans_in_rooms = [len(room.humans) for room in building.locations]
                print("{0}{1}".format(type(building).__name__, i))
                print(num_humans_in_rooms)
            for i,building in enumerate(district.locations):
                for j,room in enumerate(building.locations):
                    humans_in_rooms = [human.unique_id for human in room.humans]
                    print("{0}{1}-room{2}".format(type(building).__name__, i,j))
                    print(humans_in_rooms)

    def _populate(self, model):
        for agent in model.agents:
            if isinstance(agent, Human): self.humans.append(agent)
            if isinstance(agent, Adult): self.adults.append(agent)
            if isinstance(agent, K12Student): self.k12students.append(agent)
            if isinstance(agent, Toddler): self.toddlers.append(agent)
            if isinstance(agent, Infant): self.infants.append(agent)
            if isinstance(agent, Elder): self.elders.append(agent)
            if isinstance(agent, Location): self.locations.append(agent)
            if isinstance(agent, District): self.districts.append(agent)
            if isinstance(agent, Restaurant): self.restaurants.append(agent)
        

