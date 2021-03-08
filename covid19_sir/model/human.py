import math
import numpy as np

from model.base import (AgentBase, flip_coin, get_parameters, unique_id, linear_rescale, roulette_selection,
                        beta_distribution, logger, ENABLE_WORKER_CLASS_SPECIAL_BUILDINGS)
from model.utils import (WorkClasses, WeekDay, DiseaseSeverity, SocialPolicy, SocialPolicyUtil, InfectionStatus,
                         SimulationState, Dilemma, DilemmaDecisionHistory, TribeSelector, RestaurantType)

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
    count = 1

    @staticmethod
    def factory(covid_model, forced_age):
        # https://docs.google.com/document/d/14C4utmOi4WiBe7hOVtRt-NgMLh37pr_ntou-xUFAOjk/edit

        #moderate_severity_probs = [
        #    0,
        #    normal_ci(0.000243, 0.000832, 13),
        #    normal_ci(0.00622, 0.0213, 50),
        #    normal_ci(0.0204, 0.07, 437),
        #    normal_ci(0.0253, 0.0868, 733),
        #    normal_ci(0.0486, 0.167, 743),
        #    normal_ci(0.0701, 0.24, 790),
        #    normal_ci(0.0987, 0.338, 560),
        #    normal_ci(0.11, 0.376, 263),
        #    normal_ci(0.11, 0.376, 76)
        #]
        moderate_severity_probs = [0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 0.99, 0.99, 0.99]
        high_severity_probs = [0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 0.99, 0.99, 0.99]
        death_probs = [0] * 10
        death_probs[2] = 0.003
        death_probs[0] = death_probs[2] / 9
        death_probs[1] = death_probs[2] / 16
        death_probs[3] = death_probs[2] * 4
        death_probs[4] = death_probs[2] * 10
        death_probs[5] = death_probs[2] * 30
        death_probs[6] = death_probs[2] * 90
        death_probs[7] = death_probs[2] * 220
        death_probs[8] = death_probs[2] * 630
        death_probs[9] = death_probs[2] * 1000

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

        human.strid = f"human_{Human.count}"
        Human.count += 1
        covid_model.global_count.non_infected_count += 1
        if human.immune:
            covid_model.global_count.immune_count += 1
        else:
            covid_model.global_count.susceptible_count += 1
        #if flip_coin(get_parameters().get('initial_infection_rate')):
            #human.infect()
        return human

    def __init__(self, covid_model, age, msp, hsp, mfd):
        super().__init__(unique_id(), covid_model)
        self.properties = IndividualProperties()
        self.disease_severity = None
        self.dilemma_history = None
        self.initialize_individual_properties()
        self.home_district = None
        self.work_district = None
        self.school_district = None
        self.hospital_district = None
        self.age = age
        self.moderate_severity_prob = msp
        self.high_severity_prob = hsp
        self.death_mark = mfd
        self.infection_days_count = 0
        self.infection_latency = 0
        self.infection_incubation = 0
        self.mild_duration = 0
        self.hospitalization_duration = 0
        self.icu_duration = 0
        self.infection_status = InfectionStatus.SUSCEPTIBLE
        self.hospitalized = False
        self.hospital = None
        self.work_info = None
        if self.is_worker():
            self.setup_work_info()
            self.covid_model.global_count.work_population += 1
        self.is_dead = False
        self.tribe = {}
        for sel in TribeSelector:
            self.tribe[sel] = []
        self.mask_user = None
        self.immune = None
        self.early_symptom_detection = None
        self.count_infected_humans = 0
        self.has_been_hospitalized = False
        self.has_been_icu = False
        self.parameter_changed()
        self.social_event = None
        self.vaccinated = False
        

    def initialize_individual_properties(self):
        super().initialize_individual_properties()
        self.properties.extroversion = beta_distribution(get_parameters().get('extroversion_mean'),
                                                         get_parameters().get('extroversion_stdev'))
        self.dilemma_history = DilemmaDecisionHistory()

    def info(self):
        s = ''
        s += f'Human: {self.strid}' + '\n'
        s += f'Risk tolerance: {self.properties.risk_tolerance}' + '\n'
        s += f'Herding behavior: {self.properties.herding_behavior}' + '\n'
        s += f'Extroversion: {self.properties.extroversion}' + '\n'
        s += f'Age: {self.age}' + '\n'
        s += f'Infection status: {self.infection_status}' + '\n'
        s += f'Infection days count: {self.infection_days_count}' + '\n'
        s += f'Infection latency: {self.infection_latency}' + '\n'
        s += f'Infection incubation: {self.infection_incubation}' + '\n'
        s += f'Infection mild: {self.mild_duration}' + '\n'
        s += f'Hospitalization: {self.hospitalization_duration}' + '\n'
        s += f'ICU: {self.icu_duration}' + '\n'
        s += f'Is dead: {self.is_dead}' + '\n'
        s += f'Is hospitalized: {self.hospitalized}' + '\n'
        s += f'Is infected: {self.is_infected()}' + '\n'
        s += f'Is symptomatic: {self.is_symptomatic()}' + '\n'
        s += f'Is contagious: {self.is_contagious()}' + '\n'

    def parameter_changed(self):
        # When a parameter is changed in the middle of simulation
        # the user may want to reroll some human's properties
        self.mask_user = flip_coin(get_parameters().get('mask_user_rate'))
        self.immune = flip_coin(get_parameters().get('imune_rate'))
        if flip_coin(get_parameters().get('weareable_adoption_rate')):
            self.early_symptom_detection = 1  # number of days
        else:
            self.early_symptom_detection = 0
        self.initialize_individual_properties()

    def step(self):
        super().step()
        # The default behavior for Humans are just stay at home all day. Disease is
        # evolved in EVENING_AT_HOME
        if self.is_dead:
            return
        if self.covid_model.current_state == SimulationState.EVENING_AT_HOME:
            self.disease_evolution()
            #if not self.is_infected() and not self.is_dead and flip_coin(0.0002):
                    #self.infect()

    def vaccinate(self):
        self.vaccinated = True
        if flip_coin(get_parameters().get('vaccine_immunization_rate')):
            self.immune = True
        else:
            symptom_attenuation = get_parameters().get('vaccine_symptom_attenuation')
            self.moderate_severity_prob = self.moderate_severity_prob * (1 - symptom_attenuation)
            self.high_severity_prob = self.moderate_severity_prob * (1 - symptom_attenuation)


    def infect(self,unit):
        # https://www.acpjournals.org/doi/10.7326/M20-0504
        # https://media.tghn.org/medialibrary/2020/06/ISARIC_Data_Platform_COVID-19_Report_8JUN20.pdf
        # https://www.ecdc.europa.eu/en/covid-19/latest-evidence
        if not self.immune and not self.is_infected():
            # Evolve disease severity based in this human's specific
            # attributes and update global counts
            logger().info(f"Infected {self}")
            vec = self.covid_model.hrf.feature_vector[self]
            blob = self.covid_model.hrf.vector_to_blob[vec]
            
            if blob is not None:
                self.covid_model.actual_infections["blob"].append(blob)
                self.covid_model.actual_infections["strid"].append(self.strid)
                self.covid_model.actual_infections["unit"].append(unit.strid if unit is not None else None)
                self.covid_model.actual_infections["day"].append(self.covid_model.global_count.day_count)

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
            logger().debug(f"Infection latency of {self} is {self.infection_latency}")
            shape = get_parameters().get('incubation_period_shape')
            scale = get_parameters().get('incubation_period_scale')
            self.infection_incubation = np.random.gamma(shape, scale)
            logger().debug(f"Infection incubation of {self} is {self.infection_incubation}")
            shape = get_parameters().get('mild_period_duration_shape')
            scale = get_parameters().get('mild_period_duration_scale')
            self.mild_duration = np.random.gamma(shape, scale)
            logger().debug(f"Mild duration of {self} is {self.mild_duration}")

    def disease_evolution(self):
        # https://media.tghn.org/medialibrary/2020/06/ISARIC_Data_Platform_COVID-19_Report_8JUN20.pdf
        # https://www.ecdc.europa.eu/en/covid-19/latest-evidence
        if self.is_infected():
            self.infection_days_count += 1
            if self.disease_severity == DiseaseSeverity.ASYMPTOMATIC:
                if self.infection_days_count >= self.infection_incubation:
                    logger().info(f"{self} evolved from ASYMPTOMATIC to LOW")
                    self.disease_severity = DiseaseSeverity.LOW
                    self.covid_model.global_count.asymptomatic_count -= 1
                    self.covid_model.global_count.symptomatic_count += 1
                    day = self.covid_model.global_count.day_count
                    if day not in self.covid_model.global_count.new_symptomatic_count:
                        self.covid_model.global_count.new_symptomatic_count[day] = 0
                    self.covid_model.global_count.new_symptomatic_count[day] += 1
            elif self.disease_severity == DiseaseSeverity.LOW:
                if self.infection_days_count > self.infection_incubation + self.mild_duration:
                    # By the end of this period, either the patient is already with antibodies at
                    # a level sufficient to cure the disease or the symptoms will get worse and he/she
                    # will require hospitalization
                    if self.death_mark or flip_coin(self.moderate_severity_prob):
                        # MODERATE cases requires hospitalization
                        logger().info(f"{self} evolved from LOW to MODERATE")
                        self.disease_severity = DiseaseSeverity.MODERATE
                        self.covid_model.global_count.moderate_severity_count += 1
                        if not self.covid_model.reached_hospitalization_limit():
                            self.hospitalize()
                        else:
                            logger().info(f"{self} couldn't be hospitalized (hospitalization limit reached)")
                    else:
                        self.recover()
            elif self.disease_severity == DiseaseSeverity.MODERATE:
                if self.infection_days_count >= self.infection_incubation + self.mild_duration + self.hospitalization_duration:
                    if self.death_mark or flip_coin(self.high_severity_prob):
                        logger().info(f"{self} evolved from MODERATE to HIGH")
                        self.disease_severity = DiseaseSeverity.HIGH
                        self.covid_model.global_count.moderate_severity_count -= 1
                        self.covid_model.global_count.high_severity_count += 1
                        # If the disease evolves to HIGH and the person could not
                        # be accommodated in a hospital, he/she will die.
                        if not self.hospitalized or self.covid_model.reached_icu_limit():
                            self.die()
                        else:
                            shape = get_parameters().get('icu_period_duration_shape')
                            scale = get_parameters().get('icu_period_duration_scale')
                            self.icu_duration = np.random.gamma(shape, scale)
                            self.has_been_icu = True
                            logger().debug(f"ICU duration of {self} is {self.icu_duration}")
                    else:
                        self.recover()
            elif self.disease_severity == DiseaseSeverity.HIGH:
                if self.infection_days_count >= self.infection_incubation + self.mild_duration +\
                self.hospitalization_duration + self.icu_duration:
                    if self.death_mark:
                        self.die()
                    else:
                        self.recover()

    def recover(self):
        logger().info(f"{self} is recovered after a disease of severity {self.disease_severity}")
        self.covid_model.global_count.recovered_count += 1
        if self.disease_severity == DiseaseSeverity.MODERATE:
            self.covid_model.global_count.moderate_severity_count -= 1
        elif self.disease_severity == DiseaseSeverity.HIGH:
            self.covid_model.global_count.high_severity_count -= 1
        self.covid_model.global_count.infected_count -= 1
        if self.hospitalized:
            self.leave_hospital()
        self.infection_status = InfectionStatus.RECOVERED
        self.disease_severity = DiseaseSeverity.ASYMPTOMATIC
        self.covid_model.global_count.symptomatic_count -= 1
        self.covid_model.global_count.asymptomatic_count += 1
        self.immune = True

    def die(self):
        logger().info(f"{self} died")
        self.covid_model.global_count.symptomatic_count -= 1
        self.disease_severity = DiseaseSeverity.DEATH
        self.covid_model.global_count.high_severity_count -= 1
        self.covid_model.global_count.infected_count -= 1
        self.covid_model.global_count.death_count += 1
        if self.hospitalized:
            self.leave_hospital()
        self.is_dead = True

    def is_infected(self):
        return self.infection_status == InfectionStatus.INFECTED

    def is_contagious(self):
        if self.is_infected():
            return self.infection_days_count >= self.infection_latency or flip_coin(
                get_parameters().get('asymptomatic_contagion_probability'))
        return False

    def is_symptomatic(self):
        return self.is_infected() and self.infection_days_count >= self.infection_incubation

    def is_hospitalized(self):
        return self.hospitalized

    def hospitalize(self):
        self.hospitalized = True
        self.has_been_hospitalized = True
        if self.hospital_district is not None:
            self.hospital = self.hospital_district.get_available_hospital()
            self.hospital.patients.append(self)
        self.covid_model.global_count.total_hospitalized += 1
        logger().info(f"{self} is now hospitalized")
        shape = get_parameters().get('hospitalization_period_duration_shape')
        scale = get_parameters().get('hospitalization_period_duration_scale')
        self.hospitalization_duration = np.random.gamma(shape, scale)
        logger().debug(f"Hospital duration of {self} is {self.hospitalization_duration}")

    def leave_hospital(self):
        self.covid_model.global_count.total_hospitalized -= 1
        self.hospitalized = False
        if self.hospital_district is not None:
            self.hospital.patients.remove(self)
            self.hospital = None

    def _standard_decision(self, pd, hd):
        if hd is None:
            return pd
        else:
            if flip_coin(self.properties.herding_behavior):
                return hd
            else:
                return pd

    def personal_decision(self, dilemma):
        if dilemma == Dilemma.GO_TO_WORK_ON_LOCKDOWN:
            if self.work_info.work_class == WorkClasses.RETAIL:
                pd = flip_coin(self.properties.risk_tolerance)
                hd = self.dilemma_history.herding_decision(self, dilemma, TribeSelector.FRIEND,
                                                           get_parameters().get('min_behaviors_to_copy'))
                answer = self._standard_decision(pd, hd)
                logger().debug(f'{self}({self.unique_id}) had risk tolerance of {self.properties.risk_tolerance} in decision to work retail, making a personal decision of {pd} but a herding decision of {hd}')
            else:
                answer = False
            if answer:
                logger().info(f"{self} decided to get out to work on lockdown")
        elif dilemma == Dilemma.INVITE_FRIENDS_TO_RESTAURANT:
            if self.social_event is not None or self.is_symptomatic():
                # don't update dilemma_history since it's a compulsory decision
                return False
            rt = self.properties.risk_tolerance
            if SocialPolicy.SOCIAL_DISTANCING in get_parameters().get('social_policies'):
                rt = rt * rt
            k = 3  # TODO parameter
            d = self.covid_model.global_count.infected_count / self.covid_model.global_count.total_population
            rt = rt * math.exp(-k * d)
            pd = flip_coin(rt)
            hd = self.dilemma_history.herding_decision(self,dilemma, TribeSelector.FRIEND,
                    get_parameters().get('min_behaviors_to_copy'))
            answer = self._standard_decision(pd, hd)
            logger().debug(f'{self}({self.unique_id}) had risk tolerance of {rt} in decision to invite, making a personal decision of {pd} but a herding decision of {hd} and answer of {answer}')
            

            if answer: logger().info(f"{self} decided to invite friends to a restaurant")
        elif dilemma == Dilemma.ACCEPT_FRIEND_INVITATION_TO_RESTAURANT:
            if self.social_event is not None or self.is_symptomatic():
                # don't update dilemma_history since it's a compulsory decision
                return False
            rt = self.properties.risk_tolerance
            if SocialPolicy.SOCIAL_DISTANCING in get_parameters().get('social_policies'):
                rt = rt * rt
            k = 3  # TODO parameter
            d = self.covid_model.global_count.infected_count / self.covid_model.global_count.total_population
            rt = rt * math.exp(-k * d)
            pd = flip_coin(rt)
            hd = self.dilemma_history.herding_decision(self,dilemma, TribeSelector.FRIEND,
                    get_parameters().get('min_behaviors_to_copy'))
            answer = self._standard_decision(pd, hd)
            logger().debug(f'{self}({self.unique_id}) had risk tolerance of {rt} in decision to accept invite, making a personal decision of {pd} but a herding decision of {hd} and answer of {answer}')
            
            if answer:
                logger().info(f"{self} decided to accept an invitation to go to a restaurant")
        else:
            assert False
        for tribe in TribeSelector:
            self.dilemma_history.history[dilemma][tribe].append(answer)
        return answer

    def is_isolated(self):
        if self.is_symptomatic():
            return flip_coin(get_parameters().get('symptomatic_isolation_rate'))
        if isinstance(self, Adult):
            for policy in get_parameters().get('social_policies'):
                if policy in SocialPolicyUtil.locked_work_classes and \
                        self.work_info.work_class in SocialPolicyUtil.locked_work_classes[policy]:
                    return not self.personal_decision(Dilemma.GO_TO_WORK_ON_LOCKDOWN)
        elif isinstance(self, K12Student):
            for policy in get_parameters().get('social_policies'):
                if policy in SocialPolicyUtil.locked_student_ages:
                    lb, ub = SocialPolicyUtil.locked_student_ages[policy]
                    if lb <= self.age <= ub:
                        return True
        return False

    def is_wearing_mask(self):
        mur = get_parameters().get('mask_user_rate')
        return flip_coin(mur)

    def is_worker(self):
        return 19 <= self.age <= 64

    def get_tribe(self, tribe_selector):
        pass

    def setup_work_info(self):
        # Teachers are kept out of roulette because they are assigned by hand depending on the number of classrooms
        work_class_base_info = [
            # (class, weight, income, income when in lockdown)
            (WorkClasses.OFFICE, 1, 1, 0),
            (WorkClasses.HOUSEBOUND, 1, 1, 1),
            (WorkClasses.FACTORY, 1, 1, 0),
            (WorkClasses.RETAIL, 1, 1, 0)
        ]
        if ENABLE_WORKER_CLASS_SPECIAL_BUILDINGS:
            work_class_base_info.append((WorkClasses.HOSPITAL, 0.1, 1, 1))

        work_classes = []
        work_classes_weights = [] # used to determine the number of workers of each class
        income = {}
        for work_class, income_no_lockdown, income_lockdown, weight in work_class_base_info:
            work_classes.append(work_class)
            work_classes_weights.append(weight)
            income[work_class] = (income_no_lockdown, income_lockdown)

        self.work_info = WorkInfo()
        # TODO change to use some realistic distribution
        selected_class = roulette_selection(work_classes, work_classes_weights)
        self.work_info.work_class = selected_class
        self.work_info.base_income, self.work_info.income_loss_isolated = income[selected_class]

        self.work_info.can_work_from_home = \
            selected_class == WorkClasses.OFFICE or \
            selected_class == WorkClasses.HOUSEBOUND

        self.work_info.meet_non_coworkers_at_work = \
            selected_class == WorkClasses.RETAIL or \
            selected_class == WorkClasses.HOSPITAL

        self.work_info.essential_worker = \
            selected_class == WorkClasses.HOSPITAL

        self.work_info.fixed_work_location = \
            selected_class == WorkClasses.OFFICE or \
            selected_class == WorkClasses.HOUSEBOUND or \
            selected_class == WorkClasses.FACTORY or \
            selected_class == WorkClasses.RETAIL or \
            selected_class == WorkClasses.HOSPITAL

        self.work_info.house_bound_worker = selected_class == WorkClasses.HOUSEBOUND


    def change_work_info_to_teacher(self):
        self.work_info = WorkInfo()
        self.work_info.work_class = WorkClasses.TEACHER
        self.work_info.base_income, self.work_info.income_loss_isolated = (1.0, 0.0)
        self.work_info.can_work_from_home = False
        self.work_info.meet_non_coworkers_at_work = True
        self.work_info.essential_worker = False
        self.work_info.fixed_work_location = True
        self.work_info.house_bound_worker = False


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
        super().step()
        if self.is_dead:
            return
        if self.covid_model.current_state == SimulationState.COMMUTING_TO_MAIN_ACTIVITY:
            if not self.is_isolated() and self.school_district is not None:
                self.home_district.move_to(self, self.school_district)
        elif self.covid_model.current_state == SimulationState.COMMUTING_TO_HOME and self.school_district is not None:
            self.school_district.move_to(self, self.home_district)


class Adult(Human):
    def __init__(self, covid_model, age, msp, hsp, mfd):
        super().__init__(covid_model, age, msp, hsp, mfd)
        self.social_event = None
        self.days_since_last_social_event = 0
        self.restaurants = []

    def initialize_individual_properties(self):
        super().initialize_individual_properties()
        mean = get_parameters().get('risk_tolerance_mean')
        stdev = get_parameters().get('risk_tolerance_stdev')
        self.properties.risk_tolerance = beta_distribution(mean, stdev)
        mean = get_parameters().get('herding_behavior_mean')
        stdev = get_parameters().get('herding_behavior_stdev')
        self.properties.herding_behavior = beta_distribution(mean, stdev)

    def is_working_day(self):
        return self.covid_model.get_week_day() in self.work_info.work_days

    def invite_friends_to_restaurant(self):
        shape = self.properties.risk_tolerance * get_parameters().get('typical_restaurant_event_size')
        event_size = np.random.gamma(shape, 1)
        logger().debug(f"Restaurant event size of {self} is {event_size}")
        accepted = [self]
        for human in self.tribe[TribeSelector.FRIEND]:
            if human != self and human.personal_decision(Dilemma.ACCEPT_FRIEND_INVITATION_TO_RESTAURANT):
                accepted.append(human)
                if len(accepted) >= event_size:
                    break
        if len(accepted) == 1:
            return
        outdoor = flip_coin(linear_rescale(self.properties.risk_tolerance, 0, 0.5))
        if flip_coin(linear_rescale(self.work_info.base_income, 0, 1 / 5)):
            restaurant_type = RestaurantType.FANCY
        else:
            restaurant_type = RestaurantType.FAST_FOOD
        if self.work_district is not None:
            event = self.work_district.get_available_restaurant(len(accepted), outdoor, restaurant_type,self.restaurants)
            if event is not None and not outdoor and self.work_district is not None:
                event = self.work_district.get_available_restaurant(len(accepted), True, restaurant_type,self.restaurants)
            if event is None:
                return
            event.available -= len(accepted)
            for human in accepted:
                human.social_event = (self, event)

    def working_day(self):
        if self.covid_model.current_state == SimulationState.COMMUTING_TO_MAIN_ACTIVITY:
            if self.is_isolated():
                self.work_info.isolated = True
            else:
                self.work_info.isolated = False
                if self.work_district is not None:
                    self.home_district.move_to(self, self.work_district)
            self.covid_model.global_count.total_income += self.work_info.current_income()
        elif self.covid_model.current_state == SimulationState.MAIN_ACTIVITY:
            if self.personal_decision(Dilemma.INVITE_FRIENDS_TO_RESTAURANT):
                self.invite_friends_to_restaurant()
        elif self.covid_model.current_state == SimulationState.COMMUTING_TO_POST_WORK_ACTIVITY:
            if self.social_event is not None and self.work_district is not None:
                table, restaurant = self.social_event
                self.home_district.move_to(self, restaurant)
                self.work_district.move_to(self, restaurant)
        elif self.covid_model.current_state == SimulationState.COMMUTING_TO_HOME:
            if self.social_event is not None:
                table, restaurant = self.social_event
                self.home_district.move_from(self, restaurant)
                restaurant.available += 1
                self.days_since_last_social_event = 0
                self.social_event = None
            elif self.work_district is not None:
                self.work_district.move_to(self, self.home_district)
        elif self.covid_model.current_state == SimulationState.EVENING_AT_HOME:
            self.days_since_last_social_event += 1

    def non_working_day(self):
        if self.covid_model.current_state == SimulationState.MAIN_ACTIVITY:
            if self.personal_decision(Dilemma.INVITE_FRIENDS_TO_RESTAURANT):
                self.invite_friends_to_restaurant()
        elif self.covid_model.current_state == SimulationState.COMMUTING_TO_POST_WORK_ACTIVITY:
            if self.social_event is not None and self.work_district is not None:
                table, restaurant = self.social_event
                self.home_district.move_to(self, restaurant)
                self.work_district.move_to(self, restaurant)
        elif self.covid_model.current_state == SimulationState.COMMUTING_TO_HOME:
            if self.social_event is not None:
                table, restaurant = self.social_event
                self.home_district.move_from(self, restaurant)
                restaurant.available += 1
                self.days_since_last_social_event = 0
                self.social_event = None
                #print ("social event on weekend")

    def step(self):
        super().step()
        if self.is_dead:
            return
        if self.is_working_day():
            self.working_day()
        else:
            self.non_working_day()


class Elder(Human):
    def initialize_individual_properties(self):
        super().initialize_individual_properties()
