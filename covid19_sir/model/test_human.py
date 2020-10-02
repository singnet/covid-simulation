import copy
from model import human
from model import base
from model.utils import InfectionStatus, DiseaseSeverity, SocialPolicy, WorkClasses

parameters = base.SimulationParameters(mask_user_rate=1.0,
                                       isolation_cheater_rate=1.0,
                                       imune_rate=1.0,
                                       weareable_adoption_rate=1.0,
                                       hospitalization_capacity=1.0)
base.set_parameters(parameters)
model = base.CovidModel()
sample_human_1 = human.Human.factory(covid_model=model, forced_age=None)


def test_human_factory():
    assert sample_human_1
    assert human.Human.count == 2
    sample_human_2 = human.Human.factory(covid_model=model, forced_age=None)
    assert sample_human_2
    assert human.Human.count == 3
    assert model.global_count.immune_count \
           + model.global_count.susceptible_count \
           + model.global_count.infected_count == 2


def test_human_initialize_individual_properties():
    sample_human_1.initialize_individual_properties()
    previous_extroversion = sample_human_1.properties.extroversion
    sample_human_1.initialize_individual_properties()
    assert sample_human_1.properties.extroversion != previous_extroversion


def test_human_parameter_changed():
    # Store old human properties
    old_sample_human_1 = copy.copy(sample_human_1)
    # Change parameters and reroll properties
    parameters = base.SimulationParameters(mask_user_rate=0.0,
                                           isolation_cheater_rate=0.0,
                                           imune_rate=0.0,
                                           weareable_adoption_rate=0.0)
    base.set_parameters(parameters)
    sample_human_1.parameter_changed()
    # Make sure they're different
    assert old_sample_human_1.mask_user != sample_human_1.mask_user
    assert old_sample_human_1.isolation_cheater != sample_human_1.isolation_cheater
    assert old_sample_human_1.immune != sample_human_1.immune
    assert old_sample_human_1.early_symptom_detection != sample_human_1.early_symptom_detection


def test_human_disease_stages():
    # Test infection
    common_human = human.Human.factory(covid_model=model, forced_age=None)
    common_human.immune = False
    immune_human = copy.deepcopy(common_human)
    immune_human.immune = True
    common_human.infection_status = InfectionStatus.SUSCEPTIBLE
    immune_human.infect()
    common_human.infect()
    assert not immune_human.is_infected()
    assert common_human.is_infected()
    # Test disease evolution
    # ASYMPTOMATIC
    assert common_human.disease_severity == DiseaseSeverity.ASYMPTOMATIC
    common_human.infection_days_count = common_human.infection_incubation
    common_human.disease_evolution()
    assert common_human.disease_severity == DiseaseSeverity.LOW
    assert common_human.is_symptomatic()
    assert common_human.is_contagious()
    # LOW
    common_human.infection_days_count = common_human.mild_duration + 1
    recover_human = copy.deepcopy(common_human)
    common_human.moderate_severity_prob = 1.0
    recover_human.moderate_severity_prob = 0.0
    model.global_count.total_population = 2
    common_human.disease_evolution()
    recover_human.disease_evolution()
    assert common_human.disease_severity == DiseaseSeverity.MODERATE
    assert common_human.hospitalized
    assert not recover_human.is_infected()
    # MODERATE
    recover_human = copy.deepcopy(common_human)  # exceeded hospitalization time
    recover_human.infection_days_count = recover_human.hospitalization_duration + 1
    recover_human.disease_evolution()
    assert not recover_human.is_infected()
    recover_human = copy.deepcopy(common_human)  # kept moderate due to low "high_severity_probability"
    recover_human.high_severity_prob = 0.0
    recover_human.disease_evolution()
    assert recover_human.disease_severity == DiseaseSeverity.MODERATE
    common_human.high_severity_prob = 1.0
    dead_human = copy.deepcopy(common_human)  # death due to death_mark
    dead_human.death_mark = True
    dead_human.disease_evolution()
    assert dead_human.is_dead
    dead_human = copy.deepcopy(common_human)  # death due to lack of hospitalization
    dead_human.hospitalized = False
    dead_human.disease_evolution()
    assert dead_human.is_dead
    common_human.disease_evolution()
    assert common_human.disease_severity == DiseaseSeverity.HIGH
    # HIGH
    dead_human = copy.deepcopy(common_human)
    dead_human.death_mark = True
    dead_human.disease_evolution()
    assert dead_human.is_dead
    common_human.death_mark = False
    common_human.disease_evolution()
    assert not common_human.is_infected()


def test_human_main_activity_isolated():
    adult_human = human.Human.factory(covid_model=model, forced_age=40)
    student_human = human.Human.factory(covid_model=model, forced_age=15)
    elder_human = human.Human.factory(covid_model=model, forced_age=80)
    # Test is worker
    assert adult_human.is_worker()
    assert not student_human.is_worker()
    assert not elder_human.is_worker()
    # Infected human
    adult_human.immune = False
    adult_human.infection_status = InfectionStatus.SUSCEPTIBLE
    adult_human.infect()
    assert adult_human.infection_status == InfectionStatus.INFECTED
    adult_human.disease_severity = DiseaseSeverity.LOW
    assert not adult_human.main_activity_isolated()
    adult_human.disease_severity = DiseaseSeverity.MODERATE
    assert adult_human.main_activity_isolated()
    adult_human.disease_severity = DiseaseSeverity.HIGH
    assert adult_human.main_activity_isolated()
    # Healthy humans
    # Adult (different policies and risk tolerance)
    base.change_parameters(social_policies=[SocialPolicy.LOCKDOWN_RETAIL])
    adult_human.recover()
    adult_human.work_info.work_class = WorkClasses.RETAIL
    adult_human.properties.risk_tolerance = 1.0
    adult_human.properties.herding_behavior = 0.0
    assert not adult_human.main_activity_isolated()
    adult_human.properties.risk_tolerance = 0.0
    adult_human.properties.herding_behavior = 0.0
    assert adult_human.main_activity_isolated()
    base.change_parameters(social_policies=[])
    adult_human.properties.risk_tolerance = 0.0
    adult_human.properties.herding_behavior = 0.0
    assert not adult_human.main_activity_isolated()
    # Student
    base.change_parameters(social_policies=[SocialPolicy.LOCKDOWN_HIGH_SCHOOL])
    assert student_human.main_activity_isolated()
    base.change_parameters(social_policies=[SocialPolicy.LOCKDOWN_MIDDLE_SCHOOL])
    assert not student_human.main_activity_isolated()
    # Elder
    assert not elder_human.main_activity_isolated()


def test_human_is_wearing_mask():
    sample_human = human.Human.factory(covid_model=model, forced_age=None)
    base.change_parameters(mask_user_rate=1.0)
    assert sample_human.is_wearing_mask()
    base.change_parameters(mask_user_rate=0.0)
    assert not sample_human.is_wearing_mask()
