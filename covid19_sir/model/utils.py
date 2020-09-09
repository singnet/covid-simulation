from enum import Enum, auto

class InfectionStatus(Enum):
    SUSCEPTIBLE = auto()
    INFECTED = auto()
    RECOVERED = auto()

class DiseaseSeverity(Enum):
    ASYMPTOMATIC = auto()
    LOW = auto() # No hospitalization
    MODERATE = auto() # hospitalization
    HIGH = auto() # hospitalization in ICU
    DEATH = auto()

class WorkClasses(Enum): 
    OFFICE = auto()
    HOUSEBOUND = auto()
    FACTORY = auto()
    RETAIL = auto()
    ESSENTIAL = auto()

class WeekDay(Enum):
    SUNDAY = auto()
    MONDAY = auto()
    TUESDAY = auto()
    WEDNESDAY = auto()
    THURSDAY = auto()
    FRIDAY = auto()
    SATURDAY = auto()

class SocialPolicy(Enum):
    SOCIAL_DISTANCING = auto()
    LOCKDOWN_ALL = auto()
    LOCKDOWN_OFFICE = auto()
    LOCKDOWN_FACTORY = auto()
    LOCKDOWN_RETAIL = auto()
    LOCKDOWN_ELEMENTARY_SCHOOL = auto()
    LOCKDOWN_MIDDLE_SCHOOL = auto()
    LOCKDOWN_HIGH_SCHOOL = auto()

class SocialPolicyUtil(): 
    locked_work_classes = { 
        SocialPolicy.LOCKDOWN_ALL: [WorkClasses.OFFICE, WorkClasses.FACTORY, WorkClasses.RETAIL, WorkClasses.ESSENTIAL, WorkClasses.HOUSEBOUND], 
        SocialPolicy.LOCKDOWN_OFFICE: [WorkClasses.OFFICE], 
        SocialPolicy.LOCKDOWN_FACTORY: [WorkClasses.FACTORY], 
        SocialPolicy.LOCKDOWN_RETAIL: [WorkClasses.RETAIL] 
    } 
    locked_student_ages = { 
        SocialPolicy.LOCKDOWN_ALL: (5, 18),
        SocialPolicy.LOCKDOWN_ELEMENTARY_SCHOOL: (5, 11), 
        SocialPolicy.LOCKDOWN_MIDDLE_SCHOOL: (12, 14), 
        SocialPolicy.LOCKDOWN_HIGH_SCHOOL: (15, 18) 
    } 

class RestaurantType(Enum):
    FAST_FOOD = auto()
    FANCY = auto()
    BAR = auto()

class SimulationState(Enum):
    COMMUTING_TO_MAIN_ACTIVITY = auto()
    COMMUTING_TO_POST_WORK_ACTIVITY = auto()
    COMMUTING_TO_HOME = auto()
    MAIN_ACTIVITY = auto()
    POST_WORK_ACTIVITY = auto()
    MORNING_AT_HOME = auto()
    EVENING_AT_HOME = auto()

class TribeSelector(Enum):
    FAMILY = auto()
    COWORKER = auto()
    CLASSMATE = auto()
    AGE_GROUP = auto()
    FRIEND = auto()

class Dilemma(Enum):
    GO_TO_WORK_ON_LOCKDOWN = auto()
    INVITE_FRIENDS_TO_RESTAURANT = auto()
    ACCEPT_FRIEND_INVITATION_TO_RESTAURANT = auto()

class DilemmaDecisionHistory:
    def __init__(self):
        self.history = {}
        for dilemma in Dilemma:
            self.history[dilemma] = {}
            for tribe in TribeSelector:
                self.history[dilemma][tribe] = []

    def herding_decision(self, decider,dilemma, tribe, n):
        #find the last thing that every person in the relevant tribe did
        answer = None
        count = 0
        num_peers_can_copy =0
        for peer in decider.tribe[tribe]:
            if len (peer.dilemma_history.history[dilemma][tribe]) > 0:
                num_peers_can_copy += 1
                if peer.dilemma_history.history[dilemma][tribe][-1]: count += 1
        if num_peers_can_copy >= n:
            #if int((decider.model.schedule.time-2)/7) % 10 ==0:
                #print ('decider.model.schedule.time')
                #print (decider.model.schedule.time)
                #print ('count')
                #print (count)
                #print ('num_peers_can_copy')
                #print (num_peers_can_copy)
            answer = count > (num_peers_can_copy/2)
        return (answer)
             
