from model import location
from model.base import CovidModel, SimulationParameters, set_parameters
from model.human import Human, Adult, InfectionStatus
from model.utils import SimulationState, RestaurantType

parameters = SimulationParameters(mask_user_rate=0.0)
set_parameters(parameters)
model = CovidModel()
capacity = 2000


def check_spread(sample_location, spreading_states=None, check_step=True, in_social_event=False):
    if spreading_states is None:
        spreading_states = list(SimulationState)
    # Creates an infected and a susceptible human
    sample_human_1 = Human.factory(covid_model=model, forced_age=20)
    sample_human_1.infection_status = InfectionStatus.INFECTED
    sample_human_1.infection_days_count = 1
    sample_human_1.infection_latency = 0
    assert sample_human_1.is_infected() and sample_human_1.is_contagious()
    sample_human_2 = Adult(covid_model=model, age=50, msp=0.05, hsp=0.1, mfd=False)
    sample_human_2.strid = 'human_2'
    sample_human_2.infection_status = InfectionStatus.SUSCEPTIBLE
    assert not sample_human_2.is_infected()
    # If in social event
    if in_social_event:
        sample_human_1.social_event = sample_location
        sample_human_2.social_event = sample_location
    # Adding humans to location
    sample_location.humans += [sample_human_1, sample_human_2]
    # Changing location contagion probability to 1
    sample_location.set_custom_parameters([('contagion_probability', 1.0)], {'contagion_probability': 1.0})
    for state in SimulationState:
        sample_location.covid_model.current_state = state
        if sample_location.covid_model.current_state in spreading_states:
            print(state)
            if check_step:
                sample_location.step()
            else:
                sample_location.check_spreading(sample_human_1, sample_human_2)
            assert sample_human_2.is_infected()
            sample_human_2.infection_status = InfectionStatus.SUSCEPTIBLE
        else:
            print(state)
            if check_step:
                sample_location.step()
            assert not sample_human_2.is_infected()
        print(spreading_states)
    return True


def test_location():
    sample_hb_1 = location.Location(model, '', '')
    sample_hb_2 = location.Location(model, '', '')
    assert sample_hb_1.strid == 'Location'
    # Tests get_parameter
    assert sample_hb_1.get_parameter('contagion_probability') == 0.0
    # Tests set_custom_parameter
    sample_hb_1.set_custom_parameters([('contagion_probability', 1.0)], {'contagion_probability': 1.0})
    assert sample_hb_1.get_parameter('contagion_probability') == 1.0
    # Test move_to
    sample_human_1 = Human.factory(covid_model=model, forced_age=20)
    sample_hb_1.humans.append(sample_human_1)
    assert sample_human_1 in sample_hb_1.humans
    assert not sample_hb_2.humans
    sample_hb_1.move_to(sample_human_1, sample_hb_2)
    assert sample_human_1 not in sample_hb_1.humans
    assert sample_human_1 in sample_hb_2.humans
    # Test check_spreading
    assert check_spread(sample_hb_1, [SimulationState.POST_WORK_ACTIVITY], check_step=False)


def test_building_unit():
    # Test initialization
    sample_bu = location.BuildingUnit(capacity=capacity, covid_model=model, strid_prefix='dummy', strid_suffix='')
    assert sample_bu
    assert sample_bu.capacity == capacity
    assert not sample_bu.allocation
    # Test that infections only occur in the appropriate steps
    check_spread(sample_bu, spreading_states=[SimulationState.MORNING_AT_HOME, SimulationState.MAIN_ACTIVITY])


def test_homogeneous_building():
    sample_hb = location.HomogeneousBuilding(building_capacity=capacity, covid_model=model, strid_prefix='dummy',
                                             strid_suffix='')
    assert sample_hb


def test_restaurant():
    sample_restaurant = location.Restaurant(capacity=capacity, restaurant_type=RestaurantType.FAST_FOOD,
                                            is_outdoor=False, covid_model=model, strid_prefix='dummy', strid_suffix='')
    assert sample_restaurant
    assert not sample_restaurant.is_outdoor
    # Test spread infection
    assert check_spread(sample_restaurant, check_step=False, in_social_event=True)
    # Test step
    assert check_spread(sample_restaurant, [SimulationState.POST_WORK_ACTIVITY], check_step=True, in_social_event=True)


def test_district():
    sample_district_1 = location.District(name="district", covid_model=model, strid_prefix='dummy', strid_suffix='')
    assert isinstance(sample_district_1, location.District)
    # Test get_buildings
    sample_human_1 = Human.factory(covid_model=model, forced_age=20)
    sample_hb_1 = location.HomogeneousBuilding(building_capacity=capacity, covid_model=model, strid_prefix='dummy',
                                               strid_suffix='1')
    sample_hb_1.humans.append(sample_human_1)
    sample_district_1.locations.append(sample_hb_1)
    sample_hb_1.allocation[sample_human_1] = sample_hb_1
    sample_district_1.allocation[sample_human_1] = [sample_hb_1]
    assert sample_district_1.get_buildings(sample_human_1) == [sample_hb_1]
    # Test get_available_restaurant
    sample_restaurant = location.Restaurant(capacity=capacity, restaurant_type=RestaurantType.FAST_FOOD,
                                            is_outdoor=False, covid_model=model, strid_prefix='dummy', strid_suffix='')
    sample_restaurant.available = 100
    sample_district_1.locations.append(sample_restaurant)
    assert sample_district_1.get_available_restaurant(people_count=100,
                                                      outdoor=False,
                                                      restaurant_type=RestaurantType.FAST_FOOD) == sample_restaurant
    assert not sample_district_1.get_available_restaurant(people_count=101,
                                                          outdoor=False,
                                                          restaurant_type=RestaurantType.FAST_FOOD)
    assert not sample_district_1.get_available_restaurant(people_count=100,
                                                          outdoor=True,
                                                          restaurant_type=RestaurantType.FAST_FOOD)
    assert not sample_district_1.get_available_restaurant(people_count=100,
                                                          outdoor=False,
                                                          restaurant_type=RestaurantType.FANCY)
    # Test move_to
    sample_district_2 = location.District(name="district", covid_model=model, strid_prefix='dummy', strid_suffix='')
    sample_human_2 = Human.factory(covid_model=model, forced_age=20)
    sample_hb_2 = location.HomogeneousBuilding(building_capacity=capacity, covid_model=model, strid_prefix='dummy',
                                               strid_suffix='2')
    sample_hb_2.humans.append(sample_human_2)
    sample_district_2.locations.append(sample_hb_2)
    sample_district_2.allocation[sample_human_2] = sample_hb_2
    sample_district_1.move_to(sample_human_1, sample_hb_2)
    assert sample_human_1 in sample_hb_2.humans
    assert sample_human_2 in sample_hb_2.humans
    # Test allocate
    sample_bu_1 = location.BuildingUnit(capacity=capacity, covid_model=model, strid_prefix='dummy', strid_suffix='')
    sample_district_1.locations.append(sample_bu_1)
    sample_hb_1.locations.append(sample_bu_1)
    sample_district_1.allocate([sample_human_1, sample_human_2], True, True, True)
    assert sample_human_1 in sample_district_1.allocation.keys()
    assert sample_human_2 in sample_district_1.allocation.keys()
