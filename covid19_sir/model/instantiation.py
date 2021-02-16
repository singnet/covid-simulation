import copy
import math
import numpy as np
import random
import sys
import statistics
from statistics import mean
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
        compatable = [clist for clist in self._schema_collection if self._is_compatible(human,clist)]
        #compatable = [clist for clist in self._schema_collection ]#fixme
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
        self.roulette_distribution ={}
        self.feature_vector = {}
        self.vector_to_human = {}
        self.vector_to_home ={}
        self.vector_to_classroom = {}
        self.vector_to_office = {}
        self.vector_to_restaurant = {}
        self.unit_info_map = self.unit_info_map()
        self.strid_to_human = self.strid_to_human()
        n_vec = population_size 
        blobs,assignments = make_blobs(
            n_samples=n_vec,
            n_features=n_features,
            centers=n_blobs,
            cluster_std=0.1,#1.0
            center_box=(-10.0, 10.0),
            shuffle=False,
            random_state=iseed
        )
        self.n_blobs = n_blobs
        self.home_district_in_position = home_district_in_position
        self.blob_dict ={}
        self.vector_to_blob = {}
        for vec,assignment in zip(blobs,assignments):
            if assignment not in self.blob_dict:
                self.blob_dict[assignment] = []
            self.vector_to_blob[tuple(vec)] = assignment
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
            similarities = KeyedVectors.cosine_similarities(vector1,self.vectors)
            #print (distances)
            #distances = self.vectors.cosine_similarities(vector1,self.vectors)
            #self.roulette_distribution[tuple_vec1] = {}
            temp ={}
            sum_similarities = (similarities-similarities.min()).sum()
            for j in range(n_vec):
                if i != j:
                    vector2 = self.vectors[j]
                    tuple_vec2 = tuple(vector2)
                    temp[tuple_vec2] = (similarities[j] - similarities.min()) / sum_similarities
            
            self.roulette_distribution[tuple_vec1]=dict(sorted(temp.items(), key=lambda item: -item[1]))
            #print (self.roulette_distribution[tuple_vec1].values())
            #last = None
            #descending = True
            #for k,v in self.roulette_distribution[tuple_vec1].items():
                #if last is not None and v >= last:
                    #descending = False
                #last = v
            #print (f"descending:{descending}")

                    

    def similarity(self,tup_vec1, tup_vec2):
        vec1 = np.array(list(tup_vec1))
        vec2 = np.array(list(tup_vec2))
        sim = KeyedVectors.cosine_similarities(vec1,[vec2])
        return sim[0]


    def roulette_wheel(self,distribution):
        choice = None
        if len(distribution) >1:
            total = sum(distribution.values())
            rand = random.uniform(0.,total)
            cumulative = 0.
            for tup_vec, dist in distribution.items():
                cumulative +=dist
                if cumulative > rand:
                    choice = tup_vec
                    break
        elif len(distribution) == 1:
            choice = list(distribution.keys())[0]
        return choice


    def choice (self,chooser,keepset = set(),temperature=-1):
        choice = None
        #print ("temperature")
        #print (temperature)
        #print ("chooser")
        #print (chooser)
        #print ("len keepset")
        #print (len(keepset))
        if len(keepset) > 0:
            #print ("dist len")
            #print (len(self.roulette_distribution[chooser]))
            N=0
            if temperature < 0:
                cummu =0
                for key, sim in self.roulette_distribution[chooser].items():
                    cummu += sim
                    N+=1
                    if cummu > 1+temperature:
                        break
            else:
                N=len(self.roulette_distribution[chooser])
            #print("N")
            #print(N)
            short_dist = {}

            #last = None
            #descending = True
            #for k,v in self.roulette_distribution[chooser].items():
                #if last is not None and v >= last:
                    #descending = False
                #last = v
            #print (f"roulette descending:{descending}")

             
            dist_list = list(self.roulette_distribution[chooser].items())
            count = 0
            while len(short_dist) < N and count < len(dist_list):
                if dist_list[count][0] in keepset:
                    short_dist[dist_list[count][0]] = dist_list[count][1]
                count +=1
 
            #last = None
            #descending = True
            #for tup in self.roulette_distribution[chooser].items():
                #if last is not None and tup[1] >= last:
                    #descending = False
                #last = tup[1]
            #print (f"dist list descending:{descending}")

           #print("len short_dist")
            #print(len(short_dist))
            if temperature <= 0 or np.random.random()>temperature:
                choice = self.roulette_wheel(short_dist) if len(short_dist) > 0 else None
            else:
                choice = random.choice(list(short_dist))if len (short_dist)> 0 else None
        else:
            print("empty keepset")
        if choice is None:
            equals = len(keepset) ==1 and chooser in keepset
            print(f"None choice. Only choice is self ?  {equals}")
        else:
            sim = self.similarity(chooser,choice)
           # print(f"sim {sim} temperature {temperature}")
           # print("most similar in the keepset:")
            max = -1
            #for v in keepset:
                #sim2 = self.similarity(chooser,v)
                #if sim2 > max and v != chooser:
                    #max = sim2
            #if max > sim:
                #print (f"Max is {max} but chosen is {sim} for temperature {temperature}")
        return choice

    from statistics import mean

    def assign_features_to_families(self,families,temperature= -0.9):
        #print ("temperature-features to families")
        #print (temperature)
        #distribution = copy.deepcopy(self.roulette_distribution)
        keepset = set(self.roulette_distribution.keys())
        similarities = []
        tup_vec1 = None
        for family in families:
            if tup_vec1 is not None:
                keepset.remove(tup_vec1)
            tup_vec1 = None
            for human in family:
                if len(keepset) > 0:
                    if tup_vec1 is None:
                        #tup_vec1 = random.choice(list(distribution.keys())) 
                        tup_vec1 = random.choice(tuple(keepset)) 
                        if tup_vec1 is None:
                            print (f"{human} None vector, random_choice of len {len(distribution)}")
                        self.feature_vector[human] = tup_vec1
                        self.vector_to_human[tup_vec1] = human
                    else:
                        tup_vec2 = self.choice(tup_vec1,keepset,temperature)
                        if tup_vec2 is None:
                            print (f"{human} None vector, not matching {tup_vec1}  of len {len(distribution[tup_vec1])}")
                        else:
                            sim = self.similarity(tup_vec1,tup_vec2)
                            similarities.append(sim)
                            #print (f"{human} {tup_vec2}  vector,matching {tup_vec1}  of len {
                            #len(distribution[tup_vec1])} and similarity {sim}")
                        self.feature_vector[human] = tup_vec2
                        self.vector_to_human[tup_vec2] = human
                        keepset.remove(tup_vec1)
                        #self.remove_tup_vec(distribution,tup_vec1)
                        tup_vec1 = tup_vec2
        avg_sim = mean(similarities)
        print (f"Average similarity between family members is {avg_sim} at temperature {temperature}")

    
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

        minfactor = sys.maxsize
        index =0
        for i,l  in enumerate(factor_list):
            factor = l[0] + l[1]
            if factor < minfactor:
                minfactor = factor
                index = i
        return factor_list[index][0], factor_list[index][1]
   

    def convert_grid_to_blob_grid(self,grid_pos, grid_height,grid_width,blob_grid_height,blob_grid_width):
        blob_grid_pos_x = math.floor((grid_pos[1] * blob_grid_width)/grid_width)
        blob_grid_pos_y = math.floor((grid_pos[0] * blob_grid_height)/grid_height)
        return(blob_grid_pos_y,blob_grid_pos_x)


    def map_home_districts_to_blobs(self,grid_height,grid_width):
        #map the name of each district to the blob that most represents its residents
        #devide the grid into rectangles to do so.  This can be eventually replaced
        #with demographic data on real features. 
        #First make a blob grid that has the same number of grid squares as there are
        #blobs, or slightly less if prime is given, to a blob.  There should be many fewer
        #blobs then home districts.  Then map the home district grids to the blobs by 
        #first mapping their upper left corner to the blob grid

        self.home_districts_to_blobs = {}
        blob_grid_to_blobs = {}
        xfactor, yfactor = self.find_best_factoring(self.n_blobs)
        #print("xfactor")
        #print (xfactor)
        #print("yfactor")
        #print (yfactor)
        #print ("grid_width")
        #print (grid_width)
        #print("grid_height")
        #print(grid_height)
        blobnum =0   
        for y in range(yfactor):
            for x in range (xfactor):
                blob_grid_pos = (int(y),int(x))
                #print ("blob_grid_pos")
                #print (blob_grid_pos)
                if blob_grid_pos not in blob_grid_to_blobs:
                    blob_grid_to_blobs[blob_grid_pos]= blobnum
                    blobnum = blobnum+1 if blobnum <= self.n_blobs else 0
        for y in range(grid_height):
            for x in range(grid_width):
                grid_pos = (int(y),int(x))
                #print ("grid_pos")
                #print (grid_pos)
                blob_grid_pos = self.convert_grid_to_blob_grid(grid_pos, grid_height, grid_width, yfactor,xfactor)
                #print ("blob_grid_pos")
                #print (blob_grid_pos)
                blobnum = blob_grid_to_blobs[blob_grid_pos]
                #print("blobnum")
                #print(blobnum)
                district = self.home_district_in_position[grid_pos]
                #print ("district")
                #print (district)
                if district.strid not in self.home_districts_to_blobs:
                    self.home_districts_to_blobs[district.strid]=blobnum


    def assign_features_to_homes(self,temperature=-0.9):
        #print("temperature - assign features to homes")
        #print(temperature)
        #distribution = copy.deepcopy(self.roulette_distribution)
        #print("distribution")
        #print(distribution)
        self.home_keepset = set()
        #self.home_distribution = copy.deepcopy(self.roulette_distribution)
        home_districts = [ agent for agent in self.model.agents if isinstance(agent,District) and 'Home' in agent.strid]
        #print (self.home_districts_to_blobs)
        tup_vec1 = None
        for home_district in home_districts:
            vectors_for_home_district = self.blob_dict[self.home_districts_to_blobs[home_district.strid]]
            keepset = set([tuple(v) for v in vectors_for_home_district])
            for apartment_buildings in home_district.locations:
                tup_vec1 = None
                for apartment in apartment_buildings.locations:
                    if len(keepset) == 0 or (len(keepset) == 1 and tup_vec1 in keepset):
                        #distribution = copy.deepcopy(self.roulette_distribution)
                        #self.filter_distribution(distribution,vectors_for_home_district)
                        keepset = set([tuple(v) for v in vectors_for_home_district])
                    if len(keepset) > 0:
                        if tup_vec1 is None:
                            tup_vec1 = random.choice(tuple(keepset))
                            keepset.remove(tup_vec1)
                            #tup_vec1 = random.choice(keepset) 
                            if tup_vec1 not in self.vector_to_home:
                                self.vector_to_home [tup_vec1] = []
                            self.vector_to_home[tup_vec1].append(apartment.strid)
                            self.unit_info_map[apartment.strid]["vector"] = tup_vec1
                        else:
                            tup_vec2 = self.choice(tup_vec1,keepset,temperature)
                            keepset.remove(tup_vec2)
                            if tup_vec2 not in self.vector_to_home:
                                self.vector_to_home [tup_vec2] = []
                            self.vector_to_home[tup_vec2].append(apartment.strid)
                            self.unit_info_map[apartment.strid]["vector"] = tup_vec2
                                                        #self.remove_tup_vec(distribution,tup_vec1)
                            #self.remove_column(distribution,tup_vec1)
                            tup_vec1=tup_vec2
                        self.home_keepset.add(tup_vec1)
        #print("self.vector_to_home")
        #print(self.vector_to_home)
        #print("self.unit_info_map")
        #print(self.unit_info_map)
        #self.only_keep_columns_in_set(self.home_distribution,allocated)
        #print ( "distribution")
        #print(distribution)

    def assign_features_to_schools(self,temperature=-0.9):
        #distribution = copy.deepcopy(self.roulette_distribution)
        #allocated = set()
        #self.school_distribution = copy.deepcopy(self.roulette_distribution)
        school_districts = [ agent for agent in self.model.agents if isinstance(agent,District) and 'School' in agent.strid]
        tup_vec1=None
        for school_district in school_districts:
            vectors_for_school_district = []
            for home_district_position in school_district.home_district_list:
                home_district_strid = self.home_district_in_position[home_district_position].strid
                vectors_for_school_district.extend(self.blob_dict[self.home_districts_to_blobs[home_district_strid]])
            keepset = set([tuple(v) for v in vectors_for_school_district])
            #print (vectors_for_school_district)
            #keepset = set(self.roulette_distribution.keys())
            #self.filter_keepset(keepset,vectors_for_school_district)
            #self.filter_distribution(distribution,vectors_for_school_district)
            for school in school_district.locations:
                #tup_vec1 = None
                for classroom in school.locations:
                    tup_vec1 = None
                    if len(keepset) == 0 or (len(keepset) == 1 and tup_vec1 in keepset):
                        keepset = set([tuple(v) for v in vectors_for_school_district])
                    if len(keepset) > 0:
                        if tup_vec1 is None:
                            tup_vec1 = random.choice(tuple(keepset)) 
                            keepset.remove(tup_vec1)
                            if tup_vec1 not in self.vector_to_classroom:
                                self.vector_to_classroom [tup_vec1] = []
                            self.vector_to_classroom[tup_vec1].append(classroom.strid)
                            self.unit_info_map[classroom.strid]["vector"] = tup_vec1
                        else:
                            tup_vec2 = self.choice(tup_vec1,keepset,temperature)
                            keepset.remove(tup_vec2)
                            if tup_vec2 not in self.vector_to_classroom:
                                self.vector_to_classroom [tup_vec2] = []
                            self.vector_to_classroom[tup_vec2].append(classroom.strid)
                            self.unit_info_map[classroom.strid]["vector"] = tup_vec2
                            #self.remove_tup_vec(distribution,tup_vec1)
                            #self.remove_column(distribution,tup_vec1)
                            tup_vec1=tup_vec2
                        #allocated.add(tup_vec1) 
        #self.only_keep_columns_in_set(self.school_distribution,allocated)


    def assign_features_to_offices(self,temperature=-0.9):
        #distribution = copy.deepcopy(self.roulette_distribution)
        #allocated = set()
        #self.work_distribution = copy.deepcopy(self.roulette_distribution)
        self.restaurant_keepset = set()
        #self.restaurant_distribution = copy.deepcopy(self.roulette_distribution)
        work_districts = [ agent for agent in self.model.agents if isinstance(agent,District) and 'Work' in agent.strid]
        tup_vec1 = None
        for work_district in work_districts:
            vectors_for_work_district = []
            for home_district_position in work_district.home_district_list:
                home_district_strid = self.home_district_in_position[home_district_position].strid
                vectors_for_work_district.extend(self.blob_dict[self.home_districts_to_blobs[home_district_strid]])
            keepset = set([tuple(v) for v in vectors_for_work_district])
            #self.filter_distribution(distribution,vectors_for_work_district)
            for office_building in work_district.locations:
                #tup_vec1=None
                if 'Restaurant' in office_building.strid:
                    tup_vec3 = tuple(random.choice(vectors_for_work_district)) 
                    if tup_vec3 not in self.vector_to_restaurant:
                        self.vector_to_restaurant [tup_vec3] = []
                    self.vector_to_restaurant[tup_vec3].append(office_building.strid)
                    self.unit_info_map[office_building.strid]["vector"] = tup_vec3
                    self.restaurant_keepset.add(tup_vec3)
                for office in office_building.locations:
                    if len(keepset) == 0 or len(keepset) == 1 and tup_vec1 in keepset:
                        keepset = set([tuple(v) for v in vectors_for_work_district])
                        #distribution = copy.deepcopy(self.roulette_distribution)
                        #self.filter_distribution(distribution,vectors_for_work_district)
                    tup_vec1 = None
                    if len(keepset) > 0:
                        if tup_vec1 is None:
                            tup_vec1 = random.choice(tuple(keepset)) 
                            keepset.remove(tup_vec1)
                            if tup_vec1 not in self.vector_to_office:
                                self.vector_to_office [tup_vec1] = []
                            self.vector_to_office[tup_vec1].append(office.strid)
                            self.unit_info_map[office.strid]["vector"] = tup_vec1
                        else:
                            tup_vec2 = self.choice(tup_vec1,keepset,temperature)
                            keepset.remove(tup_vec2)
                            if tup_vec2 not in self.vector_to_office:
                                self.vector_to_office [tup_vec2] = []
                            self.vector_to_office[tup_vec2].append(office.strid)
                            self.unit_info_map[office.strid]["vector"]=tup_vec2
                            #self.remove_tup_vec(distribution,tup_vec1)
                            #self.remove_column(distribution,tup_vec1)
                            tup_vec1=tup_vec2
                        #allocated.add(tup_vec1) 
        #self.only_keep_columns_in_set(self.work_distribution,allocated)
        #self.only_keep_columns_in_set(self.restaurant_distribution,allocated_restaurants)


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

    def strid_to_human(self):
        strid_to_human = {}
        humans = [agent for agent in self.model.agents if isinstance(agent,Human)]
        for human in humans:
            strid_to_human [human.strid] = human
        return strid_to_human


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
        similarities = []
        #shuffled_families = copy.deepcopy(families)
        #random.shuffle(shuffled_families)
        for family in families:
            person = random.choice(family)
            #print("person")
            #print (type(person))
            #print (person)
            #print ("self.feature_vector")
            #print (type(list(self.feature_vector.items())[0][0]))
            #print (self.feature_vector)
            vector = self.feature_vector[person] if person in self.feature_vector else None
            if vector is not None:
                #print("vector")
                #print (vector)
                #print("self.home_distribution[vector]")
                #print (self.home_distribution[vector])
                choice = self.choice(vector,self.home_keepset,temperature)
                home = self.vector_to_home[choice].pop()
                self.allocate_home(home,family)
                #print (f"allocated home {home} to family {family}")
                if len(self.vector_to_home[choice]) == 0:
                    #self.remove_column(self.home_distribution,choice)
                    self.home_keepset.remove(choice)
                sim = self.similarity(vector,choice)
                similarities.append(sim)
            else:
                print (f"{person} not found")
        avg_sim = mean(similarities)
        print (f"Average similarity between family and home is {avg_sim} at temperature {temperature}")


    def allocate_school(self,classroom,student):
        if student not in self.unit_info_map[classroom]["district"].allocation:
            self.unit_info_map[classroom]["district"].allocation[student] = []
        self.unit_info_map[classroom]["district"].allocation[student].append(self.unit_info_map[classroom]["building"])
        self.unit_info_map[classroom]["building"].allocation[student] = self.unit_info_map[classroom]["unit"]
        self.unit_info_map[classroom]["unit"].allocation.append(student)
        student.school_district = self.unit_info_map[classroom]["district"]
        #print (f"student {student} assigned classroom {classroom} in district {student.school_district}")

           
    def allocate_schools(self, keepset,students,temperature):
        #shuffled_students = copy.deepcopy(students)
        #random.shuffle(shuffled_students)
        for student in students:
            vector = self.feature_vector[student]
            choice = self.choice(vector,keepset,temperature)
            if choice is None:
                print (f"student {student} not assigned classroom out of {len(keepset)} choices")
            classroom_str = random.choice(self.vector_to_classroom[choice])
            self.allocate_school(classroom_str,student)
            #print(f"student {student} put in classroom {classroom_str}")
            classroom = self.unit_info_map[classroom_str]["unit"]
            vacancy = classroom.capacity - len(classroom.allocation)
            if vacancy <=0 and len(self.vector_to_classroom[choice])==0:
                keepset.remove(choice)
                #self.remove_column(school_district_distribution,choice)
                self.vector_to_classroom[choice].remove(classroom_str)
                

    def allocate_school_districts(self,school_districts,temperature):
        similarities = []
        room_sizes = {}
        count = 0
        for school_district in school_districts:
            included_set = set()
            students=set()
            for school in school_district.locations:
                for classroom in school.locations:
                    if "vector" in self.unit_info_map[classroom.strid]:
                        included_set.add(self.unit_info_map[classroom.strid]["vector"])   
            #school_district_distribution = copy.deepcopy(self.roulette_distribution)
            #self.only_keep_columns_in_set(school_district_distribution,included_set)
            for grid_tuple in school_district.home_district_list:
                home_district = self.home_district_in_position[grid_tuple]
                for apartment_building in home_district.locations:
                    for apartment in apartment_building.locations:
                        for human in apartment.allocation:
                            if isinstance(human,K12Student):
                                students.add(human)
            #print (f"{len(students)} to be allocated into school districts")
            self.allocate_schools(included_set,students,temperature)
            for school in school_district.locations:
                for classroom in school.locations:
                    if len(classroom.allocation) > 0: 
                        room_sizes[count]= len(classroom.allocation)
                        count +=1
                    for student in classroom.allocation:
                        tup_vec1 = self.unit_info_map[classroom.strid]["vector"]
                        tup_vec2 = self.feature_vector[student]
                        sim = self.similarity(tup_vec1,tup_vec2)
                        similarities.append (sim)
        avg_sim = mean(similarities)
        print (f"Average similarity between students and their classroom is {avg_sim} at temperature {temperature}")
        avg_size = mean(list(room_sizes.values()))
        print (f"Average classroom occupancy is {avg_size} and number classrooms is {len(room_sizes)}")


    def allocate_workplace(self,office,worker):
        if worker not in self.unit_info_map[office]["district"].allocation:
            self.unit_info_map[office]["district"].allocation[worker] = []
        self.unit_info_map[office]["district"].allocation[worker].append(self.unit_info_map[office]["building"])
        self.unit_info_map[office]["building"].allocation[worker] = self.unit_info_map[office]["unit"]
        self.unit_info_map[office]["unit"].allocation.append(worker)
        worker.work_district = self.unit_info_map[office]["district"]
        #print (f"worker {worker} assigned office {office} in district {worker.work_district}")

           
    def allocate_workplaces(self, keepset, workers,temperature):
        #shuffled_workers = copy.deepcopy(workers)
        #random.shuffle(shuffled_workers)
        for worker in workers:
            vector = self.feature_vector[worker]
            choice = self.choice(vector,keepset,temperature)
            office_str = random.choice(self.vector_to_office[choice])
            self.allocate_workplace(office_str,worker)
            office = self.unit_info_map[office_str]["unit"]
            vacancy = office.capacity - len(office.allocation)
            if vacancy <=0 and len(self.vector_to_office[choice]) == 0:
                keepset.remove(choice)
                #self.remove_column(work_district_distribution,choice)
                self.vector_to_office[choice].remove(office_str)

    def allocate_work_districts(self,work_districts,temperature):
        similarities=[]
        for work_district in work_districts:
            included_set = set()
            workers=set()
            for office_building in work_district.locations:
                for office in office_building.locations:
                    if "vector" in self.unit_info_map[office.strid]:
                        included_set.add(self.unit_info_map[office.strid]["vector"])   
            #work_district_distribution = copy.deepcopy(self.roulette_distribution)
            #self.only_keep_columns_in_set(work_district_distribution,included_set)
            for grid_tuple in work_district.home_district_list:
                home_district = self.home_district_in_position[grid_tuple]
                for apartment_building in home_district.locations:
                    for apartment in apartment_building.locations:
                        for human in apartment.allocation:
                            if isinstance(human,Adult):
                                workers.add(human)
            self.allocate_workplaces(included_set,workers,temperature)
            room_sizes = {}
            count = 0
            for office_building in work_district.locations:
                for office in office_building.locations:
                    if len(office.allocation) > 0:
                        room_sizes[count] = len(office.allocation)
                        count += 1
                    for worker in office.allocation:
                        tup_vec1 = self.unit_info_map[office.strid]["vector"]
                        tup_vec2 = self.feature_vector[worker]
                        sim = self.similarity(tup_vec1,tup_vec2)
                        similarities.append (sim)
        avg_sim = mean(similarities)
        print (f"Average similarity between workers is {avg_sim} at temperature {temperature}")
        avg_size = mean(list(room_sizes.values()))
        print (f"Average office occupancy is {avg_size} and number offices is {len(room_sizes)}")
  
    def allocate_favorite_restaurants(self, adults,temperature,n_favorites):
        #shuffled_adults = copy.deepcopy(adults)
        #random.shuffle(shuffled_adults)
        for adult in adults:
            adult.restaurants =[] 
            if adult in self.feature_vector:
                vector = self.feature_vector[adult]
                keepset = copy.deepcopy(self.restaurant_keepset)
                #print("self.restaurant_distribution[vector]")
                #print (self.restaurant_distribution[vector])
                #distribution = self.copy_distribution(self.restaurant_distribution,[vector])
                #print ("vector_to_restuarant")
                #print (self.vector_to_restaurant.keys())
                for n in range(n_favorites):
                    choice = self.choice(vector, keepset,temperature)
                    if choice in self.vector_to_restaurant:
                        restaurant_strs = self.vector_to_restaurant[choice]
                        restaurant_str = random.choice(restaurant_strs)
                        restaurant = self.unit_info_map[restaurant_str]["building"]
                        adult.restaurants.append(restaurant)
                        #print(f"adult {adult} chooses restaurant {restaurant}")
                        keepset.remove(choice)
                        #self.remove_column(distribution,choice)
                        #vacancy = restaurant.capacity - len(restaurant.allocation)
                        #if vacancy <=0:
                            #self.remove_column(self.restaurant_distribution,choice)
            else:
                print(f"adult {adult} was not assigned a feature vector")


    def find_friends(self,human,humans,n,temperature=-0.9):
        friends=[]
        chooser = self.feature_vector[human]
        keepset = set([self.feature_vector[human2] for human2 in humans])
        #friends_dist = self.copy_distribution(self.roulette_distribution,[chooser], candidates)
        num_friends = min(n,len(keepset))
        for i in range (num_friends):
            chosen = self.choice(chooser,keepset,temperature)
            #print ('chosen')
            #print (chosen)
            if chosen in self.vector_to_human:
                friend = self.vector_to_human[chosen]
                friends.append(friend)
                keepset.remove(chosen)
                #self.remove_column(friends_dist,chosen)
        return friends



    def build_tribe(self, human, humans, mininum, maximum, temperature = -0.9):
        n = int(round(linear_rescale(human.properties.extroversion, mininum, maximum)))
        if n >= len(humans):
            n = len(humans) - 1
        if n <= 0:
            return [human]
        tribe_candidates = humans.copy()
        tribe_candidates.remove(human)
        tribe = self.find_friends(human, tribe_candidates,n,temperature)
        tribe.append(human)
        return tribe

