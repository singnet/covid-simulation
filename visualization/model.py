import json
import sys
from geojson_utils import point_in_polygon, centroid
from mesa.datacollection import DataCollector
from mesa import Model
from mesa.time import BaseScheduler
from mesa_geo.geoagent import GeoAgent, AgentCreator
from mesa_geo import GeoSpace
from shapely.geometry import Point
import numpy as np

CITY = 'SPRINGFIELD'
LOG_FILE = 'springfield_170000_0.0_log.json'

COORDS = {}
COORDS['TORONTO'] = [43.741667, -79.373333]
COORDS['MEXICO_CITY'] = [19.40, -99.1]
COORDS['NYC'] = [40.74, -73.95]
COORDS['SPRINGFIELD'] = [37.190, -93.30]

max_infected = 0

agent2feature = {}
geojson_neighborhoods = f"{CITY}/neighborhoods.geojson"
geojson_schools = f"{CITY}/schools.geojson"
geojson_zones = f"{CITY}/zoning.geojson"
geojson_restaurants = f"{CITY}/restaurants.geojson"

centroid_cache = {}
point_in_polygon_cache = {}

with open(geojson_neighborhoods) as json_file:
    NEIGHBORHOODS = json.load(json_file)
with open(geojson_schools) as json_file:
    SCHOOLS = json.load(json_file)
with open(geojson_zones) as json_file:
    ZONES = json.load(json_file)
with open(geojson_restaurants) as json_file:
    RESTAURANTS = json.load(json_file)

#for school in SCHOOLS['features']:
#    for neighborhood in NEIGHBORHOODS['features']:
#        if point_in_polygon(school['geometry'], neighborhood['geometry']):
#            print(f"{school['properties']['NAME']} -> {neighborhood['properties']['NAME']}")

def get_zone_centroid(objectid):
    if objectid in centroid_cache:
        return centroid_cache[objectid]
    for feature in ZONES['features']:
        if str(feature['properties']['OBJECTID']) == objectid:
            if feature['geometry']['type'] == 'MultiPolygon':
                feature['geometry']['type'] = 'Polygon'
                feature['geometry']['coordinates'] = feature['geometry']['coordinates'][0]
            c = centroid(feature['geometry'])
            centroid_cache[objectid] = c
            return c
    assert False

def compute_color_intensity(color1, color2, v, lower_bound, upper_bound):

    max_alpha = 255
    min_alpha = 50
    switching = 0.2

    switching_point = ((upper_bound - lower_bound) * switching) + lower_bound

    if v > switching_point:
        v_perc = (v - switching_point) / (upper_bound - switching_point)
        base_color = color2
    else:
        v_perc = (switching_point - v) / (switching_point - lower_bound)
        base_color = color1

    a = (v_perc * (max_alpha - min_alpha)) + min_alpha
    if a > 255: a = 255
    if a < 0: a = 0
    alpha = format(int(round(a)), '02x').upper()

    return base_color + alpha

def compute_point_radius(v, lower_bound, upper_bound):
    return int(round(((v - lower_bound) / (upper_bound - lower_bound)) * 30))

class RestaurantAgent(GeoAgent):

    def __init__(self, unique_id, model, shape, agent_type="safe", hotspot_threshold=1):
        """
        :param unique_id:   Unique identifier for the agent
        :param model:       Model in which the agent runs
        :param shape:       Shape object for the agent
        :param agent_type:  Indicator if agent is infected ("infected", "susceptible", "recovered" or "dead")
        :param hotspot_threshold:   Number of infected agents in region to be considered a hot-spot
        """
        super().__init__(unique_id, model, shape)
        self.feature = None
        self.map_point_radius = 2
        self.model = model
        if model.show_restaurants:
            self.map_color = 'Magenta'
        else:
            self.map_color = '#00000000'
        #self.update_state()

    def feature(self):
        return agent2feature[self.unique_id]

    def update_state(self):
        pass

    def step(self):
        self.update_state()

    def __repr__(self):
        return "Restaurant " + str(self.unique_id)

class SchoolAgent(GeoAgent):

    def __init__(self, unique_id, model, shape, agent_type="safe", hotspot_threshold=1):
        """
        :param unique_id:   Unique identifier for the agent
        :param model:       Model in which the agent runs
        :param shape:       Shape object for the agent
        :param agent_type:  Indicator if agent is infected ("infected", "susceptible", "recovered" or "dead")
        :param hotspot_threshold:   Number of infected agents in region to be considered a hot-spot
        """
        super().__init__(unique_id, model, shape)
        self.feature = None
        self.map_point_radius = 2
        self.model = model
        if model.show_schools:
            self.map_color = 'Blue'
        else:
            self.map_color = '#00000000'
        #self.update_state()

    def feature(self):
        return agent2feature[self.unique_id]

    def update_state(self):
        pass

    def step(self):
        self.update_state()

    def __repr__(self):
        return "School " + str(self.unique_id)

class NeighbourhoodAgent(GeoAgent):
    """Neighbourhood agent. Changes color according to number of infected inside it."""

    def __init__(self, unique_id, model, shape, agent_type="safe", hotspot_threshold=1):
        """
        Create a new Neighbourhood agent.
        :param unique_id:   Unique identifier for the agent
        :param model:       Model in which the agent runs
        :param shape:       Shape object for the agent
        :param agent_type:  Indicator if agent is infected ("infected", "susceptible", "recovered" or "dead")
        :param hotspot_threshold:   Number of infected agents in region to be considered a hot-spot
        """
        super().__init__(unique_id, model, shape)
        self.map_color = 'Black'
        #self.update_state()

    def feature(self):
        return agent2feature[self.unique_id]

    def check_point_in_polygon(self, point, feature, bid, agent_id):
        if bid in point_in_polygon_cache:
            if agent_id in point_in_polygon_cache[bid]:
                return point_in_polygon_cache[bid][agent_id]
        else:
            point_in_polygon_cache[bid] = {}
        check = point_in_polygon(point, feature)
        point_in_polygon_cache[bid][agent_id] = check
        return check

    def compute_infected(self):
        global max_infected
        infected = 0
        for key in self.model.simulation_data[self.model.monitored_statistic]:
            bid = key.split('-')[2]
            centroid = get_zone_centroid(bid)
            if self.check_point_in_polygon(centroid, self.feature()['geometry'], bid, self.unique_id):
                infected += self.model.simulation_data[self.model.monitored_statistic][key][self.model.steps]
        if infected > max_infected:
            print(f"New MAX for monitored statistic INFECTED: {infected}")
            max_infected = infected
        return infected

    def update_state(self):
        infected = self.compute_infected()
        self.map_color = compute_color_intensity(
            "#00FF00", 
            "#FF0000", 
            infected,
            self.model.minimum[self.model.monitored_statistic], 
            self.model.maximum[self.model.monitored_statistic]
        )

    def step(self):
        self.update_state()

    def __repr__(self):
        return "Neighborhood " + str(self.unique_id)

class InfectedModel(Model):
    """Model class for a simplistic infection model."""

    # Geographical parameters for desired map
    MAP_COORDS = COORDS[CITY]
    unique_id = "NAME"

    def __init__(self, monitored_statistic, show_schools, show_restaurants):
        """
        Create a new InfectedModel
        """
        self.schedule = BaseScheduler(self)
        self.grid = GeoSpace()
        self.steps = 0
        self.counts = None
        self.reset_counts()
        self.monitored_statistic = 'infected-per-home-series'
        self.show_schools = show_schools
        self.show_restaurants = show_restaurants
        self.maximum = {
            self.monitored_statistic: 0
        }
        self.minimum = {
            self.monitored_statistic: sys.maxsize
        }

        with open(LOG_FILE) as json_file: 
            self.simulation_data = json.load(json_file) 

        #for key in self.simulation_data[self.monitored_statistic]:
        #    for v in self.simulation_data[self.monitored_statistic][key]:
        #        if v > self.maximum[self.monitored_statistic]: self.maximum[self.monitored_statistic] = v
        #        if v < self.minimum[self.monitored_statistic]: self.minimum[self.monitored_statistic] = v
        self.minimum['infected-per-home-series'] = 0
        self.maximum['infected-per-home-series'] = 500

        self.running = True
        self.datacollector = DataCollector(
            {
                "infected": get_infected_count,
                "susceptible": get_susceptible_count,
                "recovered": get_recovered_count,
                "dead": get_dead_count,
            }
        )

        # Neighboorhoods
        AC = AgentCreator(NeighbourhoodAgent, {"model": self})
        neighbourhood_agents = AC.from_file(
            geojson_neighborhoods, unique_id=self.unique_id
        )
        for agent in neighbourhood_agents:
            for neighborhood in NEIGHBORHOODS['features']:
                if agent.unique_id == neighborhood['properties']['NAME']:
                    agent2feature[agent.unique_id] = neighborhood
                    break
        self.grid.add_agents(neighbourhood_agents)

        # Schools
        AC = AgentCreator(SchoolAgent, {"model": self})
        school_agents = AC.from_file(
            geojson_schools, unique_id=self.unique_id
        )
        for agent in school_agents:
            for school in SCHOOLS['features']:
                if agent.unique_id == school['properties']['NAME']:
                    agent2feature[agent.unique_id] = school
                    break
        self.grid.add_agents(school_agents)

        # Restaurants
        AC = AgentCreator(RestaurantAgent, {"model": self})
        restaurant_agents = AC.from_file(
            geojson_restaurants, unique_id=self.unique_id
        )
        for agent in restaurant_agents:
            for restaurant in RESTAURANTS['features']:
                if agent.unique_id == restaurant['properties']['NAME']:
                    agent2feature[agent.unique_id] = restaurant
                    break
        self.grid.add_agents(restaurant_agents)

        for agent in neighbourhood_agents + school_agents:
            self.schedule.add(agent)

        self.datacollector.collect(self)

    def reset_counts(self):
        self.counts = {
            "susceptible": 0,
            "infected": 0,
            "recovered": 0,
            "dead": 0,
            "safe": 0,
            "hotspot": 0,
        }

    def step(self):
        #if self.steps >= len(self.simulation_data['infected-series']) - 2:
        if self.steps > 50:
            self.running = False
            return
        self.steps += 1
        self.reset_counts()
        self.schedule.step()
        self.grid._recreate_rtree()  # Recalculate spatial tree, because agents are moving
        self.datacollector.collect(self)
        return True

# Functions needed for datacollector
def get_infected_count(model):
    v = model.simulation_data['infected-series']
    if model.steps >= len(v):
        return v[-1]
    else:
        return v[model.steps]


def get_susceptible_count(model):
    v = model.simulation_data['susceptible-series']
    if model.steps >= len(v):
        return v[-1]
    else:
        return v[model.steps]


def get_recovered_count(model):
    v = model.simulation_data['recovered-series']
    if model.steps >= len(v):
        return v[-1]
    else:
        return v[model.steps]


def get_dead_count(model):
    v = model.simulation_data['death-series']
    if model.steps >= len(v):
        return v[-1]
    else:
        return v[model.steps]
