import copy
import math
import numpy as np
import random
from model.base import random_selection, roulette_selection, linear_rescale
from model.human import Human, Infant, Toddler, K12Student, Adult, Elder
from model.location import District
from sklearn.datasets import make_blobs
from gensim.models import KeyedVectors


class FamilyFactory:

    def __init__(self, model):
        self.covid_model = model
        self.families = []
        self.pending = []
        self._schema_collection = [
            [Adult],
            [Elder],
            [Adult, Adult],
            [Adult, Elder],
            [Elder, Elder],
            [Adult, Toddler],
            [Adult, K12Student],
            [Adult, Adult, Infant],
            [Adult, Adult, Toddler],
            [Adult, Adult, K12Student],
            [Adult, Adult, Adult],
            [Adult, Adult, Elder],
            [Adult, Adult, Infant, Infant],
            [Adult, Adult, Infant, Toddler],
            [Adult, Adult, Infant, K12Student],
            [Adult, Adult, Toddler, Toddler],
            [Adult, Adult, Toddler, K12Student],
            [Adult, Adult, K12Student, K12Student],
            [Adult, Adult, K12Student, K12Student, K12Student],
            [Adult, Adult, K12Student, K12Student, Toddler],
            [Adult, Adult, K12Student, Toddler, Toddler],
            [Adult, Adult, K12Student, Infant, Toddler]]
        self.human_count = 0
        self.done = False

    def _select_family_schema(self, human):
        #first filter out compatables, then assign
        #else this would take longer than necessary
        #compatable = [clist for clist in self._schema_collection if self._is_compatable(human,clist)]
        compatable = [clist for clist in self._schema_collection ]#fixme
        schema = random_selection(compatable)
        # TODO use a realistic distribution
        return copy.deepcopy(schema)

    def _is_compatible(self, human, schema):
        for human_type in schema:
            if isinstance(human, human_type):
                return True
        return False

    def _assign(self, human, family, schema):
        family.append(human)
        schema.remove(type(human))
        if not schema:
            self.pending.remove((schema, family))
            self.families.append(family)
            self.human_count += len(family)

    def factory(self, population_size):
        for i in range(population_size):
            self._push(Human.factory(self.covid_model, None))
        self._flush_pending_families()

    def _flush_pending_families(self):
        self.done = True
        for schema, family in self.pending:
            flag = False
            for human in family:
                if isinstance(human, Adult) or isinstance(human, Elder):
                    flag = True
                    break
            if not flag:
                family.append(Human.factory(self.covid_model, 30))
            self.families.append(family)
            self.human_count += len(family)

    def _push(self, human):
        assert not self.done
        flag = False
        for schema, family in self.pending:
            if self._is_compatible(human, schema):
                self._assign(human, family, schema)
                flag = True
                break
        if not flag:
            while not flag:
                schema = self._select_family_schema(human)
                family = []
                self.pending.append((schema, family))
                if self._is_compatible(human, schema):
                    self._assign(human, family, schema)
                    flag = True

    def __repr__(self):
        txt = f"{len(self.families)} families - {self.human_count} people\n"
        for family in self.families:
            family.sort(reverse=True, key=lambda human: human.age)
            txt = txt + str([type(human).__name__ for human in family]) + "\n"
        return txt



class HomophilyRelationshipFactory:

    def __init__(self, model, population_size,n_blobs, n_features,home_district_in_position,iseed=None):

        self.model = model
        self.distributions ={}
        self.distributions[0]={}
        self.feature_vector = {}
        self.vector_to_home ={}
        self.vector_to_classroom = {}
        self.vector_to_office = {}
        self.vector_to_restaurant = {}
        self.unit_info_map = self.unit_info_map()
        n_vec = population_size
        blobs,assignments = make_blobs(
            n_samples=n_vec,
            n_features=n_features,
            centers=n_blobs,
            cluster_std=1.0,
            center_box=(-10.0, 10.0),
            shuffle=False,
            random_state=iseed
        )
        self.n_blobs = n_blobs
        self.home_district_in_position = home_district_in_position
        self.blob_dict ={}
        for vec,assignment in zip(blobs,assignments):
            if assignment not in self.blob_dict:
                self.blob_dict[assignment] = []
            self.blob_dict[assignment].append(vec)
        self.vectors = blobs
        #self.vectors = KeyedVectors(n_features)
        #numlist = range(n_vec)
        #self.vectors.add(numlist,blobs[:])
        #for i in range(n_vec):
            #self.vectors.add_vector(i, blobs[i,:])
            #vectors.add_vector(str(i), blobs[i,:])
        #print (numlist)
        #print(blobs)
        #print (self.vectors)
        for i in range(n_vec):
            #vector1 = self.vectors.get_vector(i)
            vector1 = self.vectors[i]
            tuple_vec1 = tuple(vector1)
            distances = KeyedVectors.cosine_similarities(vector1,self.vectors)
            #distances = self.vectors.cosine_similarities(vector1,self.vectors)
            self.distributions[0][tuple_vec1] = {}
            sum_distances = (distances-distances.min()).sum()
            for j in range(n_vec):
                if i != j:
                    vector2 = self.vectors[j]
                    tuple_vec2 = tuple(vector2)
                    self.distributions[0] [tuple_vec1][tuple_vec2] = (distances[j] - distances.min()) / sum_distances
            

    def create_choice_distribution(self, temperature):
        if temperature not in self.distributions:
            self.distributions[temperature] = {}
            if temperature > 0:
                random_portion = (1/len (self.vectors))* temperature
                non_random_portion = 1-temperature
                for tup_vec1,tup_vec2_dict in self.distributions[0].items():
                    if tup_vec1 not in self.distributions[temperature]:
                        self.distributions[temperature][tup_vec1]= {}
                        for tup_vec2,distance in tup_vec2_dict.items():
                            self.distributions[tempurature][tup_vec1][tup_vec2]=(non_random_portion*
                                    self.distributions[0][tup_vec1][tup_vec2])+random_portion
            elif temperature < 0:
                for tup_vec1,tup_vec2_dict in self.distributions[0].items():
                    if tup_vec1 not in self.distributions[temperature]:
                        remaining_space = 1.0
                        self.distributions[temperature][tup_vec1] = {}
                        for tup_vec2, distance in tup_vec2_dict.items():
                            random_portion = -temperature * remaining_space
                            self.distributions[temperature][tup_vec1][tup_vec2] = self.distributions[0
                                    ][tup_vec1][tup_vec2]+ random_portion if (
                                self.distributions[0][tup_vec1][tup_vec2] +random_portion < remaining_space) else remaining_space
                            remaining_space = remaining_space - self.distributions[temperature][tup_vec1][tup_vec2]

        
    def roulette_wheel(self,distribution):
        choice = None
        total = sum(distribution.values())
        rand = random.uniform(0.,total)
        cumulative = 0.
        for tup_vec, dist in distribution.items():
            cumulative +=dist
            if cumulative > rand:
                choice = tup_vec
                break
        return choice


    def choice (self,distribution,temperature=-1):
        #A routine in which you send in a set of vectors - This can be the vectors of persons that have been previously filtered 
        #to be of a certain type, vectors that are chosen without replacement, as is needed for filling in classrooms and workplaces. 
        #Also sent in is the "temperature" parameter , a single vector that you want to find the closest choice to, and n, how many 
        #vectors to be returned.  Returned are the vectors that obey the temperature, which are randomly drawn when the parameter 
        #is 1, roulette wheel at 0, and the n closest at -1, varying smoothly between. At -0.5 the closest value takes its given value
        #plus half the remaining space, the second most takes its value plus half the remaining space, etc, truncating at 100% of the
        #space.  At 0.5, the minimum amount all values can take is 50% of 1/n of the space, while the remaining space is allocated 
        #roulette wheel for the draw.  A slow reduction in temperature anneals the choice.
        choice = None
        #print ("temperature")
        #print (temperature)
        #print ("len(distribution)")
        #print (len(distribution))
        #print ("distribution")
        #print (distribution)
        if len(distribution) > 0:
            if temperature == -1:
                choice = max(distribution, key=distribution.get)
            elif temperature == 1:
                choice = random.choice(list(distribution))
            else:
                choice = self.roulette_wheel(distribution)
        return choice


    def remove_tup_vec(self,distribution,tup_vec):
        if tup_vec in distribution:
            distribution.pop(tup_vec)
        keys = copy.deepcopy(list(distribution.keys()))
        for key in keys:
            if tup_vec in distribution[key]:
                distribution[key].pop(tup_vec)

    def copy_dist(self,temperature = -0.9):
        self.create_choice_distribution(temperature)
        distribution = copy.deepcopy(self.distributions[temperature])
        return distribution

    def assign_features_to_families(self,families,temperature= -0.9):
        distribution = self.copy_dist(temperature)
        tup_vec1 = None
        for family in families:
            for human in family:
                if len(distribution) > 0:
                    if tup_vec1 is None:
                        tup_vec1 = random.choice(list(distribution.keys())) 
                        if tup_vec1 is None:
                            print (f"{human} None vector, random_choice of len {len(distribution)}")
                        self.feature_vector[human] = tup_vec1
                    else:
                        tup_vec2 = self.choice(distribution[tup_vec1],temperature)
                        if tup_vec2 is None:
                            print (f"{human} None vector, not matching {tup_vec1}  of len {len(distribution[tup_vec1])}")
                        #else:
                             #print (f"{human} {tup_vec2}  vector,matching {tup_vec1}  of len {len(distribution[tup_vec1])}")
                        self.feature_vector[human] = tup_vec2
                        self.remove_tup_vec(distribution,tup_vec1)
                        tup_vec1 = tup_vec2


    def sub_rectangles(self,xfactor,yfactor,westlimit=0, southlimit=0, eastlimit=2, northlimit=2):
        table=list()
        #Divide the difference between the limits by the factor
        lat_adj_factor=(northlimit-southlimit)/yfactor
        lon_adj_factor=(eastlimit-westlimit)/xfactor
        #Create longitude and latitude lists
        lat_list=[]
        lon_list=[]
        for i in range(xfactor+1):
            lon_list.append(westlimit)
            westlimit+=lon_adj_factor
        for i in range(yfactor+1):
            lat_list.append(southlimit)
            southlimit+=lat_adj_factor
        #Build a list of longitude and latitude pairs
        for i in range(0,len(lon_list)-1):
            for j in range(0,len(lat_list)-1):
                table.append([(lon_list[i],lat_list[j]),(lon_list[i+1],lat_list[j]),
                    (lon_list[i],lat_list[j+1]),(lon_list[i+1],lat_list[j+1])])

        return table
        
    
    def factors(self,n):    
        factor_list = [(i, n//i) for i in range(1, int(n**0.5) + 1) if n % i == 0]
        return factor_list
    
    def find_best_factoring(self,n):
        #subrract until you find factors other than 1* the number, if a prime was entered.
        #then return the factoring the maxiizes x*y

        prime = True
        factor_list = []
        factor_list = self.factors(n)
        while prime and n > 0:
            if len(factor_list) < 2 and n > 1:
                n += -1
            else:
                prime = False
            factor_list = self.factors(n)

        maxfactor = 0
        index =1
        for i,l  in enumerate(factor_list):
            factor = l[0] * l[1]
            if factor > maxfactor:
                maxfactor = factor
                index = i
        return factor_list[index][0], factor_list[index][1]
            



    def map_home_districts_to_blobs(self,grid_height,grid_width):
        #map the name of each district to the blob that most represents its residents
        #devide the grid into rectangles to do so.  This can be eventually replaced
        #with demographic data on real features. 
        self.home_districts_to_blobs = {}
        xfactor, yfactor = self.find_best_factoring(self.n_blobs)
        print("factor")
        print (xfactor)
        print (yfactor)
        rectangles = self.sub_rectangles(xfactor,yfactor,eastlimit = grid_width, northlimit = grid_height)
        print("rectangles")
        print (rectangles)
        blobnum = 0
        for rectangle in rectangles:
           Xs= [x[0]*grid_width for x in rectangle]
           print ("Xs")
           print (Xs)
           Ys= [y[1]* grid_height for y in rectangle]
           print ("Ys")
           print (Ys)
           minx = min(Xs)
           maxx = max(Xs)
           miny = min(Ys)
           maxy = max(Ys)
           xstep = (maxx-minx)/xfactor
           ystep = (maxy-miny)/yfactor
           for x in np.arange(min(Xs),max(Xs),xstep):
               for y in np.arange(min(Ys),max(Ys),ystep):
                   intx = int(x)
                   inty = int(y)
                   print ("(x,y)")
                   print ((intx,inty))
                   district = self.home_district_in_position[(intx,inty)]
                   #print ("district")
                   #print (district)
                   if district.strid not in self.home_districts_to_blobs:
                       self.home_districts_to_blobs[district.strid]=blobnum
                       blobnum = blobnum+1 if blobnum <= self.n_blobs else 0
                   
    def only_keep_columns_in_set(self,distribution,allocated):
        removeset=set()
        for tup1_vec,tup1_map in distribution.items():
            for tup2_vec,val in tup1_map.items():
                if tup2_vec not in allocated:
                    removeset.add(tup2_vec)
        for tup_vec in removeset:
            keys = copy.deepcopy(list(distribution.keys()))
            for key in keys:
                if tup_vec in distribution[key]:
                    distribution[key].pop(tup_vec)

 
    def remove_column(self,distribution,tup_vec):
        keys = copy.deepcopy(list(distribution.keys()))
        for key in keys:
            if tup_vec in distribution[key]:
                distribution[key].pop(tup_vec)


    def assign_features_to_homes(self,temperature=-0.9):
        distribution = self.copy_dist(temperature)
        #print("distribution")
        #print(distribution)
        allocated = set()
        self.home_distribution = self.copy_dist(temperature)
        home_districts = [ agent for agent in self.model.agents if isinstance(agent,District) and 'Home' in agent.strid]
        
        #print (self.home_districts_to_blobs)
        for home_district in home_districts:
            vectors_for_home_district = self.blob_dict[self.home_districts_to_blobs[home_district.strid]]
            tup_vec1 = None
            for apartment_buildings in home_district.locations:
                for apartment in apartment_buildings.locations:
                    if tup_vec1 is None:
                        tup_vec1 = tuple(random.choice(vectors_for_home_district)) 
                        self.vector_to_home[tup_vec1] = apartment.strid
                        self.unit_info_map[apartment.strid]["vector"] = tup_vec1
                    else:
                        tup_vec2 = self.choice(distribution[tup_vec1],temperature)
                        self.vector_to_home[tup_vec2] = apartment.strid
                        self.unit_info_map[apartment.strid]["vector"] = tup_vec2
                        self.remove_tup_vec(distribution,tup_vec1)
                        tup_vec1=tup_vec2
                    allocated.add(tup_vec1) 
        #print("self.vector_to_home")
        #print(self.vector_to_home)
        #print("self.unit_info_map")
        #print(self.unit_info_map)
        self.only_keep_columns_in_set(self.home_distribution,allocated)
        #print ( "distribution")
        #print(distribution)

    def assign_features_to_schools(self,temperature=-0.9):
        distribution = self.copy_dist(temperature)
        #allocated = set()
        #self.school_distribution = self.copy_dist(temperature)
        school_districts = [ agent for agent in self.model.agents if isinstance(agent,District) and 'School' in agent.strid]
        for school_district in school_districts:
            vectors_for_school_district = []
            for home_district_position in school_district.home_district_list:
                home_district_strid = self.home_district_in_position[home_district_position].strid
                vectors_for_school_district.extend(self.blob_dict[self.home_districts_to_blobs[home_district_strid]])
            tup_vec1=None
            for school in school_district.locations:
                for classroom in school.locations:
                    if tup_vec1 is None:
                        tup_vec1 = tuple(random.choice(vectors_for_school_district)) 
                        self.vector_to_classroom[tup_vec1] = classroom.strid
                        self.unit_info_map[classroom.strid]["vector"] = tup_vec1
                    else:
                        tup_vec2 = self.choice(distribution[tup_vec1],temperature)
                        self.vector_to_classroom[tup_vec2] = classroom.strid
                        self.unit_info_map[classroom.strid]["vector"] = tup_vec2
                        self.remove_tup_vec(distribution,tup_vec1)
                        tup_vec1=tup_vec2
                    #allocated.add(tup_vec1) 
        #self.only_keep_columns_in_set(self.school_distribution,allocated)




    def assign_features_to_offices(self,temperature=-0.9):
        distribution = self.copy_dist(temperature)
        #allocated = set()
        #self.work_distribution = self.copy_dist(temperature)
        allocated_restaurants = set()
        self.restaurant_distribution = self.copy_dist(temperature)
        work_districts = [ agent for agent in self.model.agents if isinstance(agent,District) and 'Work' in agent.strid]
        for work_district in work_districts:
            vectors_for_work_district = []
            for home_district_position in work_district.home_district_list:
                home_district_strid = self.home_district_in_position[home_district_position].strid
                vectors_for_work_district.extend(self.blob_dict[self.home_districts_to_blobs[home_district_strid]])
            tup_vec1=None
            for office_building in work_district.locations:
                if 'Restaurant' in office_building.strid:
                    tup_vec1 = tuple(random.choice(vectors_for_work_district)) 
                    self.vector_to_restaurant[tup_vec1] = office_building.strid
                    self.unit_info_map[office_building.strid]["vector"] = tup_vec1
                    allocated_restaurants.add(tup_vec1)
                for office in office_building.locations:
                    if tup_vec1 is None:
                        tup_vec1 = tuple(random.choice(vectors_for_work_district)) 
                        self.vector_to_office[tup_vec1] = office.strid
                        self.unit_info_map[office.strid]["vector"] = tup_vec1
                    else:
                        tup_vec2 = self.choice(distribution[tup_vec1],temperature)
                        self.vector_to_office[tup_vec2] = office.strid
                        self.unit_info_map[office.strid]["vector"]=tup_vec2
                        self.remove_tup_vec(distribution,tup_vec1)
                        tup_vec1=tup_vec2
                    #allocated.add(tup_vec1) 
        #self.only_keep_columns_in_set(self.work_distribution,allocated)
        self.only_keep_columns_in_set(self.restaurant_distribution,allocated_restaurants)



    def unit_info_map(self):
        unit_info_map = {}
        districts = [ agent for agent in self.model.agents if isinstance(agent,District) ]
        for district in districts:
            for building in district.locations:
                unit_info_map[building.strid]= {}
                unit_info_map[building.strid]["district"] = district
                unit_info_map[building.strid]["building"] = building
                for unit in building.locations:
                    unit_info_map[unit.strid] = {}
                    unit_info_map[unit.strid]["district"]= district 
                    unit_info_map[unit.strid]["building"]= building 
                    unit_info_map[unit.strid]["unit"]= unit 
        return unit_info_map


    def allocate_home(self, home,family):
        for human in family:
            if human not in self.unit_info_map[home]["district"].allocation:
                self.unit_info_map[home]["district"].allocation[human] = []
            self.unit_info_map[home]["district"].allocation[human].append(self.unit_info_map[home]["building"])
            self.unit_info_map[home]["building"].allocation[human] = self.unit_info_map[home]["unit"]
            self.unit_info_map[home]["unit"].allocation.append(human)
            human.home_district = self.unit_info_map[home]["district"]
            self.unit_info_map[home]["district"].get_buildings(human)[0].get_unit(human).humans.append(human)
    
                
    def allocate_homes(self, families, temperature):
        for family in families:
            person = random.choice(family)
            #print("person")
            #print (person)
            #print ("self.feature_vector")
            #print (self.feature_vector)
            vector = self.feature_vector[person]
            #print("vector")
            #print (vector)
            #print("self.home_distribution[vector]")
            #print (self.home_distribution[vector])
            choice = self.choice(self.home_distribution[vector], temperature)
            home = self.vector_to_home[choice]
            self.allocate_home(home,family)
            self.remove_column(self.home_distribution,choice)


    def allocate_school(self,classroom,student):
        if student not in self.unit_info_map[classroom]["district"].allocation:
            self.unit_info_map[classroom]["district"].allocation[student] = []
        self.unit_info_map[classroom]["district"].allocation[student].append(self.unit_info_map[classroom]["building"])
        self.unit_info_map[classroom]["building"].allocation[student] = self.unit_info_map[classroom]["unit"]
        self.unit_info_map[classroom]["unit"].allocation.append(student)
        student.school_district = self.unit_info_map[classroom]["district"]

           
    def allocate_schools(self, school_district_distribution,students,temperature):
        for student in students:
            vector = self.feature_vector[student]
            choice = self.choice(school_district_distribution[vector], temperature)
            classroom = self.vector_to_classroom[choice]
            self.allocate_school(classroom,student)
            vacancy = classroom.capacity - len(classroom.allocation)
            if vacancy <=0:
                self.remove_column(school_district_distribution,choice)

    def allocate_school_districts(self,school_districts,temperature):
        for school_district in school_districts:
            included_set = set()
            students=set()
            for school in school_district.locations:
                for classroom in school.locations:
                    included_set.add(self.unit_info_map[classroom.strid]["vector"])   
            school_district_distribution = self.copy_dist(temperature)
            self.only_keep_columns_in_set(school_district__distribution,allocated)
            for grid_tuple in school_district.school_home_list:
                home_district = self.home_district_in_position[grid_tuple]
                for apartment_building in home_district.locations:
                    for apartment in apartment_building.locations:
                        for humans in apartment.humans:
                            for human in humans:
                                if isinstance(human,K12Student):
                                    students.add(human)
            self.allocate_schools(school_district_distribution,students,temperature)


    def allocate_workplace(self,office,worker):
        if worker not in self.unit_info_map[office]["district"].allocation:
            self.unit_info_map[office]["district"].allocation[worker] = []
        self.unit_info_map[office]["district"].allocation[worker].append(self.unit_info_map[office]["building"])
        self.unit_info_map[office]["building"].allocation[worker] = self.unit_info_map[office]["unit"]
        self.unit_info_map[office]["unit"].allocation.append(worker)
        worker.work_district = self.unit_info_map[office]["district"]

           
    def allocate_workplaces(self, work_district_distribution, workers,temperature):
        for worker in workers:
            vector = self.feature_vector[worker]
            choice = self.choice(work_district_distribution[vector],temperature)
            office = self.vector_to_office[choice]
            self.allocate_workplace(office,worker)
            vacancy = office.capacity - len(office.allocation)
            if vacancy <=0:
                self.remove_column(work_district_distribution,choice)

    def allocate_work_districts(self,work_districts,temperature):
        for work_district in work_districts:
            included_set = set()
            workers=set()
            for office_building in work_district.locations:
                for office in office_building.locations:
                    included_set.add(self.unit_info_map[office.strid]["vector"])   
            work_district_distribution = self.copy_dist(temperature)
            self.only_keep_columns_in_set(work_district__distribution,allocated)
            for grid_tuple in work_district.work_home_list:
                home_district = self.home_district_in_position[grid_tuple]
                for apartment_building in home_district.locations:
                    for apartment in apartment_building.locations:
                        for humans in apartment.humans:
                            for human in humans:
                                if isinstance(human,Adult):
                                    workers.add(human)
            self.allocate_workplaces(work_district_distribution,workers,temperature)



  
    def allocate_favorite_restaurants(self, adults,temperature,n_favorites):
        for adult in adults:
            adult.restaurants =[] 
            vector = self.feature_vector[adult]
            for n in range(n_favorites):
                choice = self.choice(self.restaurant_distribution[vector], temperature)
                adult.restaurants.append(self.vector_to_restaurant[choice])
                vacancy = restaurant.capacity - len(restaurant.allocation)
                if vacancy <=0:
                    self.remove_column(self.restaurant_distribution,choice)
    
    def copy_allocated(self,distribution,choosers,candidates):
        new_dist = {}
        for chooser in choosers:
            new_dist[chooser] = {}
            for candidate in candidates:
                new_dist[chooser][candidate] = distribution[chooser][candidate]
        return new_dist
        

    def find_friends(self,human,humans,n,temperature=-0.9):
        friends=[]
        self.create_choice_distribution(temperature)
        chooser = self.feature_vector[human]
        candidates = [self.feature_vector[human] for human in humans]
        friends_dist = self.copy_allocated(self.distributions[temperature],chooser, candidates)
        for i in range (n):
            friends.append(choice(friends_dist[chooser],temperature))
            self.remove_tup_vec(friends_dist,chooser)
        return friends



    def build_tribe(self, human, humans, mininum, maximum, temperature = -0.9):
        n = int(round(linear_rescale(human.properties.extroversion, mininum, maximum)))
        if n >= len(humans):
            n = len(humans) - 1
        if n <= 0:
            return [human]
        tribe_candidates = humans.copy()
        tribe_candidates.remove(human)
        tribe = find_friends(human, tribe_candidates,n,temperature)
        tribe.append(human)
        return tribe

