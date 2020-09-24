import pytest
import numpy as np
from utils import setup_city_layout
from model import base
from model.human import Human
from model.utils import WeekDay


def test_flip_coin():
    assert type(base.flip_coin(0.5)) == bool
    with pytest.raises(TypeError):
        base.flip_coin("50")


def test_random_selection():
    random_numbers = np.random.rand(5)
    assert base.random_selection(random_numbers, 1) in random_numbers


# Setting up parameters
hospitalization_capacity = 0.05
parameters = base.SimulationParameters(hospitalization_capacity=hospitalization_capacity)
base.set_parameters(parameters)
model = base.CovidModel()
population_size = 1000
setup_city_layout(model, population_size)


def test_agent_base():
    random_id = np.random.randint(50)
    sample_agent = base.AgentBase(random_id, model)
    assert sample_agent.id == random_id
    assert sample_agent in model.agents


def test_initial_model_state():
    # Assert initial model state
    assert model.agents
    assert model.reached_hospitalization_limit() is False
    assert isinstance(model.get_week_day(), WeekDay)
    assert model.is_week_day(WeekDay.MONDAY)
    assert model.current_state == base.SimulationState.MORNING_AT_HOME


def test_reached_hospitalization_limit():
    assert model.reached_hospitalization_limit() is False
    # Hospitalizing population
    hospitalization_limit = np.ceil(hospitalization_capacity * population_size) + 1
    for agent in model.agents:
        if isinstance(agent, Human):
            agent.hospitalized = True
            model.global_count.total_hospitalized += 1
            if model.global_count.total_hospitalized >= hospitalization_limit:
                break
    model.step()
    assert model.reached_hospitalization_limit() is True


def test_reroll_human_properties():
    sample_human = None
    for agent in model.agents:
        if isinstance(agent, Human):
            sample_human = agent
            break
    assert sample_human, "No human agent was found in the model, please make sure humans have been added."
    previous_extroversion = sample_human.properties.extroversion
    model.reroll_human_properties()
    assert sample_human.properties.extroversion != previous_extroversion


def test_add_listener():
    class DummyListener:
        def __init__(self, covid_model):
            self.started_cycles = 0
            self.ended_cycles = 0
            self.model = covid_model

        def start_cycle(self, _):
            self.started_cycles += 1

        def end_cycle(self, _):
            self.ended_cycles += 1

    listener = DummyListener(model)
    model.add_listener(listener)
    model.step()
    assert listener.started_cycles == 1
    assert listener.ended_cycles == 1


def test_model_step():
    day_count = model.global_count.day_count
    weekday = model.get_week_day()
    model.step()
    # Asserts the day has changed
    assert model.global_count.day_count == day_count + 1
    assert not model.is_week_day(weekday)
    # Asserts the weekday repeats in 7 days
    while model.global_count.day_count != day_count + 7:
        model.step()
    assert model.global_count.day_count == day_count + 7
    assert model.is_week_day(weekday)
