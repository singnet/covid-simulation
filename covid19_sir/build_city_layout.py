import sys
import datetime
import json
import numpy as np
import os.path
from geojson_utils import centroid, point_distance


from model.base import random_selection, beta_range, get_parameters, flip_coin, normal_cap, roulette_selection, build_roulette
from model.human import Adult, K12Student
from model.location import District, HomogeneousBuilding, BuildingUnit, Restaurant
from model.instantiation import FamilyFactory
from model.utils import RestaurantType

class CityLayout:

    def __init__(self, city, population_size, model):

        self.work_zones = ['GM', 'IC', 'GR', 'CS', 'HC', 'O-1', 'O-2', 'LI', 'HM', 'GI', 'LB',
        'WC-1', 'WC-2', 'WC-3', 'RI', 'CC', 'COM-1', 'COM-2', 'CNTY-A-1', 'CNTY-M-1', 'CNTY-C-2']
        self.home_zones = ['R-SF', 'R-LD', 'R-TH', 'R-MD', 'CNTY-R-1', 'R-MHC', 'R-HD']
        self.unused_zones = ['PD', 'CNTY-PAD', None]

        home_low_density_capacity = 20
        home_medium_density_capacity = 150
        home_high_density_capacity = 450

        family_capacity_per_building = {
            'R-SF': 1,
            'R-LD': home_low_density_capacity,
            'R-TH': home_low_density_capacity,
            'R-MD': home_medium_density_capacity,
            'CNTY-R-1': home_medium_density_capacity,
            'R-MHC': home_medium_density_capacity,
            'R-HD': home_high_density_capacity
        }
        family_capacity = {}

        self.model = model
        self.population_size = population_size
        self.zone_centroid = {}
        self.work_building_ids = []
        self.home_building_ids = []
        self.school_building_ids = []
        self.restaurant_building_ids = []
        self.home_building = {}
        self.work_building = {}
        self.school_building = {}
        self.restaurant_building = {}
        self.restaurant_distances = {}
        self.school_distances = {}
        self.work_zone_distances = {}
        self.restaurant_roulette = {}
        self.school_roulette = {}
        self.work_zone_roulette = {}
        self.sigma = get_parameters().params['real_sites_roulette_rescale_sigma']
        self.kappa = get_parameters().params['real_sites_roulette_rescale_kappa']

        home_district = District('HomeDistrict', model, '', '')
        work_district = District('WorkDistrict', model, '', '')
        school_district = District('SchoolDistrict', model, '', '')
        restaurant_district = District('RestaurantDistrict', model, '', '')
        with open(f'mesa-geo/examples/GeoSIR/{city}/neighborhoods.geojson') as json_file:
            self.neighborhoods = json.load(json_file)
            self.neighborhoods_count = len(self.neighborhoods['features'])
            print(f"Total number of neighboorhoods: {self.neighborhoods_count}")
        with open(f'mesa-geo/examples/GeoSIR/{city}/schools.geojson') as json_file:
            self.schools = json.load(json_file)
            self.schools_count = len(self.schools['features'])
            for school in self.schools['features']:
                bid = str(school['properties']['OBJECTID'])
                self.school_building_ids.append(bid)
                self.school_building[bid] = HomogeneousBuilding(1000000, model, 'School', bid)
                school_district.locations.append(self.school_building[bid])
            print(f"Total number of schools: {self.schools_count}")
        with open(f'mesa-geo/examples/GeoSIR/{city}/zoning.geojson') as json_file:
            self.buildings = json.load(json_file)
            print(f"Total number of buildings: {len(self.buildings['features'])}")
            self.all_zones_coordinates = []
            for building in self.buildings['features']:
                bid = str(building['properties']['OBJECTID'])
                self.zone_centroid[bid] = self.compute_centroid(building)
                zone = building['properties']['PLANZONE']
                if building['geometry']['type'] == 'Polygon':
                    self.all_zones_coordinates.append(building['geometry']['coordinates'][0])
                elif building['geometry']['type'] == 'MultiPolygon':
                    for v in building['geometry']['coordinates']:
                        self.all_zones_coordinates.append(v[0])
                else:
                    assert False
                if zone in self.work_zones:
                    self.work_building_ids.append(bid)
                    self.work_building[bid] = HomogeneousBuilding(1000000, model, 'Work', bid)
                    work_district.locations.append(self.work_building[bid])
                elif zone in self.home_zones:
                    family_capacity[bid] = family_capacity_per_building[zone]
                    self.home_building_ids.append(bid)
                    self.home_building[bid] = HomogeneousBuilding(family_capacity[bid], model, 'Home', bid)
                    home_district.locations.append(self.home_building[bid])
                elif zone not in self.unused_zones:
                    print(f"Unknown zone type: {zone}")
                    exit()
        self.restaurants = self.create_geo_restaurants(self.buildings)
        self.restaurants_count = len(self.restaurants['features'])
        for restaurant in self.restaurants['features']:
            bid_int = restaurant['properties']['OBJECTID']
            bid = str(bid_int)
            self.restaurant_building_ids.append(bid)
            self.restaurant_building[bid] = self.create_restaurant_location(bid_int, flip_coin(0.1))
            restaurant_district.locations.append(self.restaurant_building[bid])
        print(f"Total number of restaurants: {self.restaurants_count}")
        with open('restaurants.geojson', 'w') as fp:
            json.dump(self.restaurants, fp)

        distance_cache_file = f'mesa-geo/examples/GeoSIR/{city}/distances_{self.sigma}_{self.kappa}.json'
        if os.path.isfile(distance_cache_file):
            with open(distance_cache_file) as json_file:
                table = json.load(json_file)
            self.restaurant_distances = table['restaurant_distances']
            self.school_distances = table['school_distances']
            self.work_zone_distances = table['work_zone_distances']
            self.restaurant_roulette = table['restaurant_roulette']
            self.school_roulette = table['school_roulette']
            self.work_zone_roulette = table['work_zone_roulette']
        else:
            self.compute_restaurant_distances()
            self.compute_school_distances()
            self.compute_work_zone_distances()
            table = {}
            table['restaurant_distances'] = self.restaurant_distances
            table['school_distances'] = self.school_distances
            table['work_zone_distances'] = self.work_zone_distances
            table['restaurant_roulette'] = self.restaurant_roulette
            table['school_roulette'] = self.school_roulette
            table['work_zone_roulette'] = self.work_zone_roulette
            with open(distance_cache_file, 'w') as json_file:
                json.dump(table, json_file)

        family_factory = FamilyFactory(model)
        family_factory.factory(population_size)
        model.global_count.total_population = family_factory.human_count
        print(f"Total number of families: {len(family_factory.families)}")
        #count_family = 0
        for family in family_factory.families:
            #if count_family % 1000 == 0: print(f"{count_family} {datetime.datetime.now()}")
            #count_family += 1
            assert len(self.home_building_ids) > 0
            home_bid = random_selection(self.home_building_ids)
            selected_home_build = self.home_building[home_bid]
            home_unit = BuildingUnit(10, model, home_bid, '', contagion_probability=beta_range(0.021, 0.12))
            family_capacity[home_bid] -= 1
            if family_capacity[home_bid] == 0:
                self.home_building_ids.remove(home_bid)
            selected_home_build.locations.append(home_unit)
            for human in family:
                # Home
                human.home_district = home_district
                home_district.allocation[human] = [selected_home_build]
                home_unit.allocation.append(human)
                selected_home_build.allocation[human] = home_unit
                assert home_district.get_buildings(human)[0].get_unit(human) == home_unit
                home_unit.humans.append(human)
                # Work
                if isinstance(human, Adult):
                    human.work_district = work_district
                    #work_bid = random_selection(self.work_building_ids)
                    work_bid = roulette_selection(self.work_zone_distances[home_bid]['work_zone_bid'], self.work_zone_distances[home_bid]['distance'], roulette=self.work_zone_roulette[home_bid])
                    selected_work_building = self.work_building[work_bid]
                    work_unit = selected_work_building.locations[-1] if selected_work_building.locations and\
                        len(work_unit.allocation) < work_unit.capacity else None
                    if work_unit is None:
                        work_unit = BuildingUnit(10, model, work_bid, '', contagion_probability=beta_range(0.007, 0.06))
                        selected_work_building.locations.append(work_unit)
                    work_district.allocation[human] = [selected_work_building]
                    work_unit.allocation.append(human)
                    selected_work_building.allocation[human] = work_unit
                    assert work_district.get_buildings(human)[0].get_unit(human) == work_unit
                # School
                if isinstance(human, K12Student):
                    human.school_district = school_district
                    #work_bid = random_selection(self.school_building_ids)
                    school_bid = roulette_selection(self.school_distances[home_bid]['school_bid'], self.school_distances[home_bid]['distance'], roulette=self.school_roulette[home_bid])
                    selected_school_building = self.school_building[school_bid]
                    school_unit = selected_school_building.locations[-1] if selected_school_building.locations and\
                        len(school_unit.allocation) < school_unit.capacity else None
                    if school_unit is None:
                        school_unit = BuildingUnit(20, model, school_bid, '', contagion_probability=beta_range(0.014, 0.08))
                        selected_school_building.locations.append(school_unit)
                    school_district.allocation[human] = [selected_school_building]
                    school_unit.allocation.append(human)
                    selected_school_building.allocation[human] = school_unit
                    assert school_district.get_buildings(human)[0].get_unit(human) == school_unit
                # Restaurants
                if isinstance(human, Adult):
                    #bids = [random_selection(self.restaurant_building_ids) for i in range(10)]
                    bids = roulette_selection(self.restaurant_distances[work_bid]['restaurant_bid'], self.restaurant_distances[work_bid]['distance'], 10, roulette=self.restaurant_roulette[work_bid])
                    human.preferred_restaurants = [self.restaurant_building[bid] for bid in bids]

    def compute_centroid(self, feature):
        if feature['geometry']['type'] == 'MultiPolygon':
            feature['geometry']['type'] = 'Polygon'
            feature['geometry']['coordinates'] = feature['geometry']['coordinates'][0]
        return centroid(feature['geometry'])

    def compute_restaurant_distances(self):
        for zone in self.buildings['features']:
            zone_bid = str(zone['properties']['OBJECTID'])
            if zone_bid not in self.work_building_ids: continue
            self.restaurant_distances[zone_bid] = {
                'restaurant_bid': [],
                'distance': []
            }
            for restaurant in self.restaurants['features']:
                restaurant_centroid = restaurant['geometry']
                restaurant_bid = str(restaurant['properties']['OBJECTID'])
                distance = point_distance(restaurant_centroid, self.zone_centroid[zone_bid])
                self.restaurant_distances[zone_bid]['restaurant_bid'].append(restaurant_bid)
                self.restaurant_distances[zone_bid]['distance'].append(distance)
            self.restaurant_roulette[zone_bid] = build_roulette(
                self.restaurant_distances[zone_bid]['distance'], 
                sigma=self.sigma, 
                kappa=self.kappa, 
                reverse=True)

    def compute_school_distances(self):
        for zone in self.buildings['features']:
            zone_bid = str(zone['properties']['OBJECTID'])
            if zone_bid not in self.home_building_ids: continue
            self.school_distances[zone_bid] = {
                'school_bid': [],
                'distance': []
            }
            for school in self.schools['features']:
                school_centroid = school['geometry']
                school_bid = str(school['properties']['OBJECTID'])
                distance = point_distance(school_centroid, self.zone_centroid[zone_bid])
                self.school_distances[zone_bid]['school_bid'].append(school_bid)
                self.school_distances[zone_bid]['distance'].append(distance)
            self.school_roulette[zone_bid] = build_roulette(
                self.school_distances[zone_bid]['distance'], 
                sigma=self.sigma, 
                kappa=self.kappa, 
                reverse=True)

    def compute_work_zone_distances(self):
        for zone in self.buildings['features']:
            zone_bid = str(zone['properties']['OBJECTID'])
            if zone_bid not in self.home_building_ids: continue
            self.work_zone_distances[zone_bid] = {
                'work_zone_bid': [],
                'distance': []
            }
            for work_zone in self.buildings['features']:
                work_zone_bid = str(work_zone['properties']['OBJECTID'])
                if work_zone_bid not in self.work_building_ids: continue
                distance = point_distance(self.zone_centroid[work_zone_bid], self.zone_centroid[zone_bid])
                self.work_zone_distances[zone_bid]['work_zone_bid'].append(work_zone_bid)
                self.work_zone_distances[zone_bid]['distance'].append(distance)
            self.work_zone_roulette[zone_bid] = build_roulette(
                self.work_zone_distances[zone_bid]['distance'], 
                sigma=self.sigma, 
                kappa=self.kappa, 
                reverse=True)

    def create_restaurant_location(self, index, is_bar):
        if is_bar:
            bar = Restaurant(
                normal_cap(100, 20, 50, 200),
                RestaurantType.BAR,
                flip_coin(0.5),
                self.model,
                'Restaurant',
                str(index))
            return bar
        else:
            if flip_coin(0.5):
                restaurant_type = RestaurantType.FAST_FOOD
                rtype = "FASTFOOD"
            else:
                restaurant_type = RestaurantType.FANCY
                rtype = "FANCY"
            restaurant = Restaurant(
                normal_cap(
                    get_parameters().params['restaurant_capacity_mean'], 
                    get_parameters().params['restaurant_capacity_stdev'], 
                    16, 
                    200
                ),
                restaurant_type,
                flip_coin(0.5),
                self.model,
                'Restaurant',
                str(index))
            return restaurant
        
    def select_random_coordinates(self):
        v = random_selection(self.all_zones_coordinates)
        p = random_selection(v, 2)
        return [(p[0][0] + p[1][0]) / 2, (p[0][1] + p[1][1]) / 2]

    def create_geo_restaurants(self, zones):
        n = 100
        prob_bar = 0.10
        restaurants = {
            "type": "FeatureCollection",
            "name": "Restaurants",
            "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
            "features": []
        }
        for i in range(n):
            feature = {
                "type": "Feature", 
                "properties": {
                    "OBJECTID": i,
                    "NAME": f"Restaurant {i}"
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": self.select_random_coordinates()
                }
            }
            restaurants['features'].append(feature)
        return restaurants
        
