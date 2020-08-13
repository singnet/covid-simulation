import math
import numpy as np
from enum import Enum, auto

from model.base import (AgentBase, flip_coin, normal_cap, roulette_selection,
get_parameters, unique_id, linear_rescale, normal_cap_ci)
from model.utils import (WorkClasses, WeekDay, InfectionStatus, DiseaseSeverity,
SocialPolicy, SocialPolicyUtil, InfectionStatus, SimulationState, Dilemma,DilemmaDecisionHistory,
TribeSelector, RestaurantType)

class WorkInfo:
    work_class = None
    can_work_from_home = False
    meet_non_coworkers_at_work = False
    essential_worker = False
    fixed_work_location = False
    house_bound_worker = False
    base_income = 0.0
    income_loss_isolated = 0.0
    isolated = False
    work_days = [
        WeekDay.MONDAY,
        WeekDay.TUESDAY,
        WeekDay.WEDNESDAY,
        WeekDay.THURSDAY,
        WeekDay.FRIDAY,
        WeekDay.SATURDAY
    ]

    def current_income(self):
        if self.isolated:
            return self.base_income * (1.0 - self.income_loss_isolated)
        else:
            return self.base_income

class IndividualProperties:
    # All in [0..1]
    risk_tolerance = 0.0
    herding_behavior = 0.0
    extroversion = 0.5

class Human(AgentBase):

    @staticmethod
    def factory(covid_model, forced_age):
        # https://docs.google.com/document/d/14C4utmOi4WiBe7hOVtRt-NgMLh37pr_ntou-xUFAOjk/edit
        moderate_severity_probs = [
            0,
            normal_cap_ci(0.000243, 0.000832, 13),
            normal_cap_ci(0.00622, 0.0213, 50),
            normal_cap_ci(0.0204, 0.07, 437),
            normal_cap_ci(0.0253, 0.0868, 733),
            normal_cap_ci(0.0486, 0.167, 743),
            normal_cap_ci(0.0701, 0.24, 790),
            normal_cap_ci(0.0987, 0.338, 560),
            normal_cap_ci(0.11, 0.376, 263),
            normal_cap_ci(0.11, 0.376, 76)
        ]
        high_severity_probs = [0.05, 0.05, 0.05, 0.05, 0.063, 0.122, 0.274, 0.432, 0.709, 0.709]
        death_probs = [0.002, 0.00006, 0.0003, 0.0008, 0.0015, 0.006, 0.022, 0.051, 0.093, 0.093]
        if forced_age is None:
            age = int(np.random.beta(2, 5, 1) * 100)
        else:
            age = forced_age
        index = age // 10
        msp = moderate_severity_probs[index]
        hsp = high_severity_probs[index]
        mfd = flip_coin(death_probs[index])
        if age <= 1: 
            human = Infant(covid_model, age, msp, hsp, mfd)
        elif age <= 4: 
            human = Toddler(covid_model, age, msp, hsp, mfd)
        elif age <= 18: 
            human = K12Student(covid_model, age, msp, hsp, mfd)
        elif age <= 64: 
            human = Adult(covid_model, age, msp, hsp, mfd)
        else:
            human = Elder(covid_model, age, msp, hsp, mfd)

        covid_model.global_count.non_infected_count += 1
        if human.immune:
            covid_model.global_count.immune_count += 1
        else:
            covid_model.global_count.susceptible_count += 1
        if flip_coin(get_parameters().get('initial_infection_rate')):
            human.infect()
        return human

    def __init__(self, covid_model, age, msp, hsp, mfd):
        super().__init__(unique_id(), covid_model)
        self.properties = IndividualProperties()
        self.initialize_individual_properties()
        self.home_district = None
        self.work_district = None
        self.school_district = None
        self.age = age
        self.moderate_severity_prob = msp
        self.high_severity_prob = hsp
        self.death_mark = mfd
        self.infection_days_count = 0
        self.infection_latency = 0
        self.infection_incubation = 0
        self.mild_duration = 0
        self.hospitalization_duration = 0
        self.infection_status = InfectionStatus.SUSCEPTIBLE
        self.hospitalized = False
        if self.is_worker(): 
            self.setup_work_info()
            self.covid_model.global_count.work_population += 1
        self.is_dead = False
        self.tribe = {}
        for sel in TribeSelector:
            self.tribe[sel] = []
        self.parameter_changed()

    def initialize_individual_properties(self):
        super().initialize_individual_properties()
        self.properties.extroversion = normal_cap(get_parameters().get('extroversion_mean'),
                get_parameters().get('extroversion_stdev'), 0.0, 1.0)
        self.dilemma_history = DilemmaDecisionHistory()

    def parameter_changed(self):
        # When a parameter is changed in the middle of simulation
        # the user may want to reroll some human's properties
        self.mask_user = flip_coin(get_parameters().get('mask_user_rate'))
        self.isolation_cheater = flip_coin(get_parameters().get('isolation_cheater_rate'))
        self.immune = flip_coin(get_parameters().get('imune_rate'))
        if flip_coin(get_parameters().get('weareable_adoption_rate')):
            self.early_symptom_detection = 1 # number of days
        else:
            self.early_symptom_detection = 0
        self.initialize_individual_properties()
        
    def step(self):
        # The default nehavior for Humans are just stay at home all day. Disease is
        # evolved in EVENING_AT_HOME
        if self.is_dead: return
        if self.covid_model.current_state == SimulationState.EVENING_AT_HOME:
            self.disease_evolution()

    def infect(self):
        # https://www.acpjournals.org/doi/10.7326/M20-0504
        # https://media.tghn.org/medialibrary/2020/06/ISARIC_Data_Platform_COVID-19_Report_8JUN20.pdf
        # https://www.ecdc.europa.eu/en/covid-19/latest-evidence
        if not self.immune:
            # Evolve disease severity based in this human's specific
            # attributes and update global counts
            self.covid_model.global_count.infected_count += 1
            self.covid_model.global_count.non_infected_count -= 1
            self.covid_model.global_count.susceptible_count -= 1
            self.infection_status = InfectionStatus.INFECTED
            self.disease_severity = DiseaseSeverity.ASYMPTOMATIC
            self.covid_model.global_count.asymptomatic_count += 1
            shape = get_parameters().get('latency_period_shape')
            scale = get_parameters().get('latency_period_scale')
            self.infection_latency = np.random.gamma(shape, scale) - self.early_symptom_detection
            if self.infection_latency < 1.0:
                self.infection_latency = 1.0
            shape = get_parameters().get('incubation_period_shape')
            scale = get_parameters().get('incubation_period_scale')
            self.infection_incubation = self.infection_latency + np.random.gamma(shape, scale)
            shape = get_parameters().get('mild_period_duration_shape')
            scale = get_parameters().get('mild_period_duration_scale')
            self.mild_duration = np.random.gamma(shape, scale) + self.infection_incubation

    def disease_evolution(self):
        # https://media.tghn.org/medialibrary/2020/06/ISARIC_Data_Platform_COVID-19_Report_8JUN20.pdf
        # https://www.ecdc.europa.eu/en/covid-19/latest-evidence
        if self.is_infected():
            self.infection_days_count += 1
            if self.disease_severity == DiseaseSeverity.ASYMPTOMATIC:
                if self.infection_days_count >= self.infection_incubation:
                    self.disease_severity = DiseaseSeverity.LOW
                    self.covid_model.global_count.asymptomatic_count -= 1
                    self.covid_model.global_count.symptomatic_count += 1
            elif self.disease_severity == DiseaseSeverity.LOW:
                if self.infection_days_count > self.mild_duration:
                    # By the end of this period, either the pacient is already with antibodies at
                    # a level sufficient to cure the disease or the simptoms will get worse and he/she
                    # will require hospitalization
                    if flip_coin(self.moderate_severity_prob):
                        # MODERATE cases requires hospitalization
                        self.disease_severity = DiseaseSeverity.MODERATE
                        self.covid_model.global_count.moderate_severity_count += 1
                        if not self.covid_model.reached_hospitalization_limit():
                            self.covid_model.global_count.total_hospitalized += 1
                            self.hospitalized = True
                        shape = get_parameters().get('hospitalization_period_duration_shape')
                        scale = get_parameters().get('hospitalization_period_duration_scale')
                        self.hospitalization_duration = np.random.gamma(shape, scale) + self.infection_days_count
                    else:
                        self.recover()
            elif self.disease_severity == DiseaseSeverity.MODERATE:
                if self.infection_days_count > self.hospitalization_duration:
                    self.recover()
                else:
                    if flip_coin(self.high_severity_prob):
                        self.disease_severity = DiseaseSeverity.HIGH
                        self.covid_model.global_count.moderate_severity_count -= 1
                        self.covid_model.global_count.high_severity_count += 1
                        # If the disease evolves to HIGH and the person could not
                        # be accomodated in a hospital, he/she will die.
                        if not self.hospitalized or self.death_mark:
                            self.die()
            elif self.disease_severity == DiseaseSeverity.HIGH:
                if self.death_mark:
                    self.die()

    def recover(self):
        self.covid_model.global_count.recovered_count += 1
        if self.disease_severity == DiseaseSeverity.MODERATE:
            self.covid_model.global_count.moderate_severity_count -= 1
        elif self.disease_severity == DiseaseSeverity.HIGH:
            self.covid_model.global_count.high_severity_count -= 1
        self.covid_model.global_count.infected_count -= 1
        if self.hospitalized:
            self.covid_model.global_count.total_hospitalized -= 1
            self.hospitalized = False
        self.infection_status = InfectionStatus.RECOVERED
        self.disease_severity = DiseaseSeverity.ASYMPTOMATIC
        self.covid_model.global_count.symptomatic_count -= 1
        self.covid_model.global_count.asymptomatic_count += 1
        self.immune = True

    def die(self):
        self.covid_model.global_count.symptomatic_count -= 1
        self.disease_severity = DiseaseSeverity.DEATH
        self.covid_model.global_count.high_severity_count -= 1
        self.covid_model.global_count.infected_count -= 1
        self.covid_model.global_count.death_count += 1
        if self.hospitalized:
            self.covid_model.global_count.total_hospitalized -= 1
            self.hospitalized = False
        self.is_dead = True
        
    def is_infected(self):
        return self.infection_status == InfectionStatus.INFECTED

    def is_contagious(self):
        if self.is_infected() and self.infection_days_count >= self.infection_latency:
            if self.is_symptomatic() or flip_coin(get_parameters().get('asymptomatic_contagion_probability')):
                return True
        return False
    
    def is_symptomatic(self):
        return self.is_infected() and self.infection_days_count >= self.infection_incubation

    def _standard_decision(self, pd, hd):
        if hd is None:
            return pd
        else:
            if flip_coin(self.properties.herding_behavior):
                return hd
            else:
                return pd
        
    def personal_decision(self, dilemma):
        answer = False
        if dilemma == Dilemma.GO_TO_WORK_ON_LOCKDOWN:
            if self.work_info.work_class == WorkClasses.RETAIL:
                pd = flip_coin(self.properties.risk_tolerance)
                hd = self.dilemma_history.herding_decision(self,dilemma, TribeSelector.FRIEND,
                        get_parameters().get('min_behaviors_to_copy'))
                answer = self._standard_decision(pd, hd)
            else:
                answer = False
        elif dilemma == Dilemma.INVITE_FRIENDS_TO_RESTAURANT:
            if self.social_event is not None or self.is_symptomatic():
                # don't update dilemma_history since it's a compulsory decision
                return False
            rt = self.properties.risk_tolerance
            if SocialPolicy.SOCIAL_DISTANCING in get_parameters().get('social_policies'):
                rt = rt * rt
            k = 3 #TODO parameter
            d = self.covid_model.global_count.infected_count / self.covid_model.global_count.total_population
            rt = rt * math.exp(-k * d)
            pd = flip_coin(rt)
            hd = self.dilemma_history.herding_decision(self,dilemma, TribeSelector.FRIEND,
                    get_parameters().get('min_behaviors_to_copy'))
            answer = self._standard_decision(pd, hd)
        elif dilemma == Dilemma.ACCEPT_FRIEND_INVITATION_TO_RESTAURANT:
            if self.social_event is not None or self.is_symptomatic():
                # don't update dilemma_history since it's a compulsory decision
                return False
            rt = self.properties.risk_tolerance
            if SocialPolicy.SOCIAL_DISTANCING in get_parameters().get('social_policies'):
                rt = rt * rt
            k = 3 # TODO parameter
            d = self.covid_model.global_count.infected_count / self.covid_model.global_count.total_population
            rt = rt * math.exp(-k * d)
            pd = flip_coin(rt)
            hd = self.dilemma_history.herding_decision(self,dilemma, TribeSelector.FRIEND,
                    get_parameters().get('min_behaviors_to_copy'))
            answer = self._standard_decision(pd, hd)
        else: assert False
        for tribe in TribeSelector:
            self.dilemma_history.history[dilemma][tribe].append(answer)
        return answer

    def main_activity_isolated(self):
        if self.is_infected():
            if self.disease_severity == DiseaseSeverity.MODERATE or \
               self.disease_severity == DiseaseSeverity.HIGH:
                return True
            if self.is_symptomatic():
                ir = get_parameters().get('symptomatic_isolation_rate')
                if flip_coin(ir):
                    return True
        if isinstance(self, Adult):
            for policy in get_parameters().get('social_policies'):
                if policy in SocialPolicyUtil.locked_work_classes and \
                   self.work_info.work_class in SocialPolicyUtil.locked_work_classes[policy]:
                    return not self.personal_decision(Dilemma.GO_TO_WORK_ON_LOCKDOWN)
        elif isinstance(self, K12Student):
            for policy in get_parameters().get('social_policies'):
                if policy in SocialPolicyUtil.locked_student_ages:
                    lb, ub = SocialPolicyUtil.locked_student_ages[policy]
                    if self.age >= lb and self.age <= ub:
                        return True
        return False
        
    def is_wearing_mask(self):
        mur = get_parameters().get('mask_user_rate')
        return flip_coin(mur)
        

    def is_worker(self):
        return self.age >= 19 and self.age <= 64

    def get_tribe(self, tribe_selector):
        pass
            
    def setup_work_info(self):
        income = {
            # Income (in [0..1]) when the people is working and when he/she is isolated at home
            WorkClasses.OFFICE: (1.0, 0.0),
            WorkClasses.HOUSEBOUND: (1.0, 0.0),
            WorkClasses.FACTORY: (1.0, 1.0),
            WorkClasses.RETAIL: (1.0, 1.0),
            WorkClasses.ESSENTIAL: (1.0, 1.0),
        }
        classes = [key for key in income.keys()]
        roulette = []
        self.work_info = WorkInfo()

        #TODO change to use some realistic distribution
        count = 1
        for wclass in classes:
            roulette.append(count / len(classes))
            count = count + 1
        selected_class = roulette_selection(classes, roulette)
        self.work_info.work_class = selected_class
        self.work_info.base_income, self.work_info.income_loss_isolated = income[selected_class]

        self.work_info.can_work_from_home = \
            selected_class ==  WorkClasses.OFFICE or \
            selected_class == WorkClasses.HOUSEBOUND

        self.work_info.meet_non_coworkers_at_work = \
            selected_class == WorkClasses.RETAIL or \
            selected_class == WorkClasses.ESSENTIAL
           
        self.work_info.essential_worker = \
            selected_class == WorkClasses.ESSENTIAL

        self.work_info.fixed_work_location = \
            selected_class == WorkClasses.OFFICE or \
            selected_class == WorkClasses.HOUSEBOUND or \
            selected_class == WorkClasses.FACTORY or \
            selected_class == WorkClasses.RETAIL or \
            selected_class == WorkClasses.ESSENTIAL

        self.work_info.house_bound_worker = WorkClasses.HOUSEBOUND


class Infant(Human):
    def initialize_individual_properties(self):
      super().initialize_individual_properties()
    
class Toddler(Human):
    def initialize_individual_properties(self):
      super().initialize_individual_properties()
    
class K12Student(Human):
    def initialize_individual_properties(self):
      super().initialize_individual_properties()

    def step(self):
        if self.is_dead: return
        if self.covid_model.current_state == SimulationState.COMMUTING_TO_MAIN_ACTIVITY:
            if not self.main_activity_isolated():
                self.home_district.move_to(self, self.school_district)
        elif self.covid_model.current_state == SimulationState.COMMUTING_TO_HOME:
            self.school_district.move_to(self, self.home_district)
        elif self.covid_model.current_state == SimulationState.EVENING_AT_HOME:
            self.disease_evolution()
    
class Adult(Human):
    def __init__(self, covid_model, age, msp, hsp, mfd):
        super().__init__(covid_model, age, msp, hsp, mfd)
        self.social_event = None
        self.days_since_last_social_event = 0

    def initialize_individual_properties(self):
      super().initialize_individual_properties()
      mean = get_parameters().get('risk_tolerance_mean')
      stdev = get_parameters().get('risk_tolerance_stdev')
      self.properties.risk_tolerance = normal_cap(mean, stdev, 0.0, 1.0)
      mean = get_parameters().get('herding_behavior_mean')
      stdev = get_parameters().get('herding_behavior_stdev')
      self.properties.herding_behavior = normal_cap(mean, stdev, 0.0, 1.0)

    def is_working_day(self):
        return self.covid_model.get_week_day() in self.work_info.work_days

    def invite_friends_to_get_out(self):
        event = self.home_district.get_available_gathering_spot()
        if event is not None:
            assert not event.humans
            flag = False
            for human in self.tribe[TribeSelector.FRIEND]:
                if human != self and human.personal_decision(Dilemma.ACCEPT_FRIEND_INVITATION_TO_GET_OUT):
                    flag = True
                    human.social_event = event
            if flag:
                self.social_event = event
            else:
                event.available = True

    def invite_friends_to_restaurant(self):
        shape = self.properties.risk_tolerance * get_parameters().get('typical_restaurant_event_size')
        event_size = np.random.gamma(shape, 1)
        accepted = [self]
        for human in self.tribe[TribeSelector.FRIEND]:
            if human != self and human.personal_decision(Dilemma.ACCEPT_FRIEND_INVITATION_TO_RESTAURANT):
                accepted.append(human)
                if len(accepted) >= event_size: break
        if len(accepted) == 1: return
        outdoor = flip_coin(linear_rescale(self.properties.risk_tolerance, 0, 0.5))
        if flip_coin(linear_rescale(self.work_info.base_income, 0, 1/5)):
            restaurant_type = RestaurantType.FANCY
        else:
            restaurant_type = RestaurantType.FAST_FOOD
        event = self.work_district.get_available_restaurant(len(accepted), outdoor, restaurant_type)
        if event is not None and not outdoor:
            event = self.work_district.get_available_restaurant(len(accepted), True, restaurant_type)
        if event is None: return
        event.available -= len(accepted)
        for human in accepted:
            human.social_event = event

    def working_day(self):
        if self.covid_model.current_state == SimulationState.COMMUTING_TO_MAIN_ACTIVITY:
            if self.main_activity_isolated():
                self.work_info.isolated = True
            else:
                self.work_info.isolated = False
                self.home_district.move_to(self, self.work_district)
            self.covid_model.global_count.total_income += self.work_info.current_income()
        elif self.covid_model.current_state == SimulationState.MAIN_ACTIVITY:
            if self.personal_decision(Dilemma.INVITE_FRIENDS_TO_RESTAURANT):
                self.invite_friends_to_restaurant()
        elif self.covid_model.current_state == SimulationState.COMMUTING_TO_POST_WORK_ACTIVITY:
            if self.social_event is not None:
                self.home_district.move_to(self, self.social_event)
                self.work_district.move_to(self, self.social_event)
        elif self.covid_model.current_state == SimulationState.COMMUTING_TO_HOME:
            if self.social_event is not None:
                self.home_district.move_from(self, self.social_event)
                self.social_event.available = True
                self.days_since_last_social_event = 0
                self.social_event = None
            else:
                self.work_district.move_to(self, self.home_district)
        elif self.covid_model.current_state == SimulationState.EVENING_AT_HOME:
            self.disease_evolution()
            self.days_since_last_social_event += 1

    def non_working_day(self):
        if self.covid_model.current_state == SimulationState.EVENING_AT_HOME:
            self.disease_evolution()

    def step(self):
        if self.is_dead: return
        if self.is_working_day():
            self.working_day()
        else:
            self.non_working_day()
    
class Elder(Human):
    def initialize_individual_properties(self):
      super().initialize_individual_properties()
