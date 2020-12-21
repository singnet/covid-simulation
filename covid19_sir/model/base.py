import uuid
import math
import sys
import numpy as np
import logging
from mesa import Agent, Model
from mesa.time import RandomActivation
from model.utils import SimulationState, WeekDay

LOG_FILE_NAME = './simulation.log'
LOGGING_LEVEL = logging.CRITICAL


def flip_coin(prob):
    if np.random.random() < prob:
        return True
    else:
        return False


def _random_selection(v):
    return v[np.random.randint(len(v))]


def random_selection(v, n=1):
    if n == 1:
        return _random_selection(v)
    assert n <= (len(v) / 2)
    a = v.copy()
    selected = []
    for i in range(n):
        s = _random_selection(a)
        a.remove(s)
        selected.append(s)
    return selected

def build_roulette(w):
    answer = []
    s = sum(w)
    acc = 0
    for v in w:
        acc += v / s
        answer.append(acc)
    answer[-1] = 1
    return answer

def _roulette_selection(values, weights, max_weigth=None, delete_selected=False):
    assert len(values) == len(weights)
    if max_weigth == None:
        max_weigth = max(weights)
    while True:
        selected = np.random.randint(len(values))
        if np.random.random() < weights[selected] / max_weigth:
            answer = values[selected]
            if delete_selected:
                del values[selected]
                del weights[selected]
            return answer

def _find_nearest(already_selected, roulette, pos, rand, delta):
    if not already_selected[pos]:
        return (pos, 0)
    while (delta < 0 and pos > 0) or (delta > 0 and pos < (len(already_selected) -1)):
        pos += delta
        if not already_selected[pos]:
            return (pos, abs(rand - roulette[pos]))
    return (-1, sys.float_info.max)


def roulette_selection(values, weights, num_selections=None):
    assert len(values) == len(weights)
    if num_selections is None:
        num_selections = 1
        return_scalar = True
    else:
        return_scalar = False
    assert 0 < num_selections <= len(values)
    num_values = len(values)
    roulette = build_roulette(weights)
    answer = []
    already_selected = [False] * num_values
    rand = sorted([np.random.random() for i in range(num_selections)])
    rand_used = [False] * num_selections
    current_rand = 0
    for i in range(len(values)):
        if rand[current_rand] <= roulette[i]:
            answer.append(values[i])
            already_selected[i] = True
            rand_used[current_rand] = True
            while current_rand < num_selections and rand[current_rand] <= roulette[i]:
                current_rand += 1
            if current_rand >= num_selections: break
    if len(answer) < num_selections:
        current_rand = 0
        while rand_used[current_rand]: current_rand += 1
        assert current_rand < num_selections
        for i in range(len(values)):
            while rand[current_rand] <= roulette[i]:
                nearest_left, distance_left = _find_nearest(already_selected, roulette, i, rand[current_rand], -1)
                nearest_right, distance_right = _find_nearest(already_selected, roulette, i, rand[current_rand], 1)
                nearest = nearest_left if distance_left < distance_right else nearest_right
                assert nearest >= 0
                answer.append(values[nearest])
                already_selected[nearest] = True
                rand_used[current_rand] = True
                while current_rand < num_selections and rand_used[current_rand]:
                    current_rand += 1
                if current_rand >= num_selections:
                    break
            if current_rand >= num_selections:
                break
    assert len(answer) == num_selections
    if return_scalar:
        return answer[0]
    else:
        return answer


def convert_parameters(mean, stdev):
    """Converts mean and standard deviation parameters into alpha and beta to be used in the beta distribution."""
    var = stdev ** 2
    assert var < mean * (1 - mean), "Variance check failed when converting mean and stdev parameters to alpha and beta."
    alpha = mean * (mean * (1-mean)/var - 1)
    beta = (1-mean) * (mean * (1-mean)/var - 1)
    return alpha, beta


def beta_distribution(mean, stdev):
    """Draws random values from a beta distribution using alpha and beta parameters derived from mean and standard
    deviation."""
    alpha, beta = convert_parameters(mean, stdev)
    return np.random.beta(alpha, beta)


def beta_range(lower_bound, upper_bound):
    """Draws a random number from a beta distribution with parameters alpha = 2 and beta = 2 (this assures that the
    values will stay in the range [0, 1]. Then rescales the values to the desired range."""
    assert lower_bound < upper_bound, "Parameter lower_bound must be smaller than upper_bound."
    return (np.random.beta(2, 2) * (upper_bound - lower_bound)) + lower_bound


# DEPRECATED
def normal_cap(mean, stdev, lower_bound=0, upper_bound=1):
    r = np.random.normal(mean, stdev)
    if r < lower_bound:
        r = lower_bound
    if r > upper_bound:
        r = upper_bound
    return r


# DEPRECATED
def normal_ci(ci_lower, ci_upper, n):
    # Assumption of 95% CI
    mean = (ci_lower + ci_upper) / 2
    stdev = math.sqrt(n) * (ci_upper - ci_lower) / 3.92
    return normal_cap(mean, stdev, ci_lower, ci_upper)


def linear_rescale(x, l2, u2, l1=0, u1=1):
    return ((x / (u1 - l1)) * (u2 - l2)) + l2


def unique_id():
    return uuid.uuid1()


def set_parameters(new_parameters):
    global parameters
    parameters = new_parameters
    logger().info(f"Setting new simulation parameters\n{parameters}")


def get_parameters():
    global parameters
    return parameters


def change_parameters(**kwargs):
    global parameters
    for key in kwargs:
        parameters.params[key] = kwargs.get(key)


class Logger:
    __instance = None

    @staticmethod
    def get_instance():
        if Logger.__instance is None:
            return Logger()
        return Logger.__instance

    def __init__(self):
        if Logger.__instance is not None:
            raise Exception("Invalid re-instantiation of Logger")
        else:
            # print("log initialized")
            logging.basicConfig(
                filename=LOG_FILE_NAME,
                level=LOGGING_LEVEL,
                format='%(levelname)s: %(message)s')
            Logger.__instance = self
            Logger.model = None

    def prefix(self):
        return f"Day {self.model.global_count.day_count} " if self.model else ''

    def debug(self, msg):
        logging.debug(self.prefix() + msg)

    def info(self, msg):
        logging.info(self.prefix() + msg)

    def warning(self, msg):
        logging.warning(self.prefix() + msg)

    def error(self, msg):
        logging.error(self.prefix() + msg)


def logger():
    return Logger.get_instance()


class SimulationStatus:
    def __init__(self):
        self.day_count = 0
        self.infected_count = 0
        self.non_infected_count = 0
        self.susceptible_count = 0
        self.immune_count = 0
        self.recovered_count = 0
        self.moderate_severity_count = 0
        self.high_severity_count = 0
        self.death_count = 0
        self.symptomatic_count = 0
        self.asymptomatic_count = 0
        self.total_hospitalized = 0
        self.total_population = 0
        self.work_population = 0
        self.total_income = 0.0
        self.new_symptomatic_count = {}
        self.infection_info = {}


class SimulationParameters:
    def __init__(self, **kwargs):
        self.params = {'social_policies': kwargs.get("social_policies", []),
                       'mask_user_rate': kwargs.get("mask_user_rate", 0.0),
                       'mask_efficacy': kwargs.get("mask_efficacy", 0.0),
                       'imune_rate': kwargs.get("imune_rate", 0.01),
                       'initial_infection_rate': kwargs.get("initial_infection_rate", 0.01),
                       'hospitalization_capacity': kwargs.get("hospitalization_capacity", 0.50),
                       'icu_capacity': kwargs.get("icu_capacity", 0.03),
                       'icu_period_duration_shape': kwargs.get("icu_period_duration_shape", 10.0),
                       'icu_period_duration_scale': kwargs.get("icu_period_duration_scale", 1.0),
                       'latency_period_shape': kwargs.get("latency_period_shape", 2.0),
                       'latency_period_scale': kwargs.get("latency_period_scale", 1.0),
                       'incubation_period_shape': kwargs.get("incubation_period_shape", 6.0),
                       'incubation_period_scale': kwargs.get("incubation_period_scale", 1.0),
                       'mild_period_duration_shape': kwargs.get("mild_period_duration_shape", 14.0),
                       'mild_period_duration_scale': kwargs.get("mild_period_duration_scale", 1.0),
                       'hospitalization_period_duration_shape': kwargs.get("hospitalization_period_duration_shape",
                                                                           12.0),
                       'hospitalization_period_duration_scale': kwargs.get("hospitalization_period_duration_scale",
                                                                           1.0),
                       'weareable_adoption_rate': kwargs.get("weareable_adoption_rate", 0.0),
                       'contagion_probability': kwargs.get("contagion_probability", 0.0),
                       'spreading_rate': kwargs.get("spreading_rate", 0.0),
                       'symptomatic_isolation_rate': kwargs.get("symptomatic_isolation_rate", 0.0),
                       'asymptomatic_contagion_probability': kwargs.get("asymptomatic_contagion_probability", 0.1),
                       'risk_tolerance_mean': kwargs.get("risk_tolerance_mean", 0.4),
                       'risk_tolerance_stdev': kwargs.get("risk_tolerance_stdev", 0.3),
                       'herding_behavior_mean': kwargs.get("herding_behavior_mean", 0.4),
                       'herding_behavior_stdev': kwargs.get("herding_behavior_stdev", 0.3),
                       'allowed_restaurant_capacity': kwargs.get("allowed_restaurant_capacity", 1.0),
                       'typical_restaurant_event_size': kwargs.get("typical_restaurant_event_size", 6),
                       'extroversion_mean': kwargs.get("extroversion_mean", 0.5),
                       'extroversion_stdev': kwargs.get("extroversion_stdev", 0.3),
                       'restaurant_count_per_work_district': kwargs.get("restaurant_count_per_work_district", 10),
                       'restaurant_capacity_mean': kwargs.get("restaurant_capacity_mean", 50),
                       'restaurant_capacity_stdev': kwargs.get("restaurant_capacity_stdev", 20),
                       'min_behaviors_to_copy': kwargs.get("min_behaviors_to_copy", 3),
                       'num_communities': kwargs.get("num_communities",1),
                       'num_features':kwargs.get("num_features",10),
                       'temperature':kwargs.get("temperature",-1)}

    def get(self, key):
        return self.params[key]

    def set(self, key, value):
        self.params[key] = value

    def __repr__(self):
        answer = "{\n"
        for key in self.params:
            answer += f"'{key}': {self.params[key]}\n"
        answer += "}"
        return answer


parameters = None


class AgentBase(Agent):
    # MESA agent
    def __init__(self, unique_id, covid_model):
        super().__init__(unique_id, covid_model)
        self.id = unique_id
        self.covid_model = covid_model
        covid_model.schedule.add(self)
        covid_model.agents.append(self)
        self.debug = False
        self.debug_each_n_cycles = covid_model.debug_each_n_cycles
        self.strid = None

    def __repr__(self):
        return self.strid

    def initialize_individual_properties(self):
        pass

    def _debug(self):
        pass

    def step(self):
        if self.debug and self.covid_model.global_count.day_count % self.debug_each_n_cycles == 0:
            self._debug()


class CovidModel(Model):
    def __init__(self, debug=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = debug
        self.debug_each_n_cycles = 1
        self.agents = []
        self.global_count = SimulationStatus()
        self.schedule = RandomActivation(self)
        self.listeners = []
        self.current_state = SimulationState.MORNING_AT_HOME
        # State machine which controls agent's behavior
        self.next_state = {
            SimulationState.MORNING_AT_HOME: SimulationState.COMMUTING_TO_MAIN_ACTIVITY,
            SimulationState.COMMUTING_TO_MAIN_ACTIVITY: SimulationState.MAIN_ACTIVITY,
            SimulationState.MAIN_ACTIVITY: SimulationState.COMMUTING_TO_POST_WORK_ACTIVITY,
            SimulationState.COMMUTING_TO_POST_WORK_ACTIVITY: SimulationState.POST_WORK_ACTIVITY,
            SimulationState.POST_WORK_ACTIVITY: SimulationState.COMMUTING_TO_HOME,
            SimulationState.COMMUTING_TO_HOME: SimulationState.EVENING_AT_HOME,
            SimulationState.EVENING_AT_HOME: SimulationState.MORNING_AT_HOME
        }

    def reached_hospitalization_limit(self):
        return (self.global_count.total_hospitalized / self.global_count.total_population) >= parameters.get(
            'hospitalization_capacity')

    def reached_icu_limit(self):
        return (self.global_count.high_severity_count / self.global_count.total_population) >= parameters.get(
            'icu_capacity')

    def get_week_day(self):
        wd = [WeekDay.MONDAY,
              WeekDay.TUESDAY,
              WeekDay.WEDNESDAY,
              WeekDay.THURSDAY,
              WeekDay.FRIDAY,
              WeekDay.SATURDAY,
              WeekDay.SUNDAY]
        return wd[self.global_count.day_count % 7]

    def is_week_day(self, wd):
        return self.get_week_day() == wd

    def reroll_human_properties(self):
        for agent in self.agents:
            agent.initialize_individual_properties()

    def add_listener(self, listener):
        # listeners are external entities which are notified just before a cycle begin
        # and just after its end.
        self.listeners.append(listener)

    def _debug(self):
        pass

    def step(self):
        assert self.current_state == SimulationState.MORNING_AT_HOME
        logger().info(f"Day count: {self.global_count.day_count}")
        if self.debug and self.global_count.day_count % self.debug_each_n_cycles == 0:
            self._debug()

        for listener in self.listeners:
            listener.start_cycle(self)

        if not self.is_week_day(WeekDay.SUNDAY):
            self.global_count.total_income = 0.0
        flag = False
        # Cycles thru all the states before ending a simulation step
        while not flag:
            logger().info(f"STATE: {self.current_state}")
            self.schedule.step()
            for listener in self.listeners:
                listener.state_change(self)
            self.current_state = self.next_state[self.current_state]
            if self.current_state == SimulationState.MORNING_AT_HOME:
                flag = True

        for listener in self.listeners:
            listener.end_cycle(self)
        self.global_count.day_count += 1
