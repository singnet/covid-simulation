import copy
import math
import numpy as np
from model.base import roulette_selection, linear_rescale
from model.human import Human, Infant, Toddler, K12Student, Adult, Elder
from sklearn.datasets import make_blobs


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
        # TODO use a realistic distribution
        self._weights = []
        for i in range(len(self._schema_collection)):
            self._weights.append((i + 1) * (1.0 / len(self._schema_collection)))
        self.human_count = 0
        self.done = False

    def _select_family_schema(self, human):
        while True:
            schema = roulette_selection(self._schema_collection, self._weights)
            if self._is_compatible(human, schema):
                break
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

    def __init__(self, model, humans, **kwargs):
        self.model = model
        self.humans = humans
        self.similarity = {}
        self.feature_vector = {}
        blobs = make_blobs(
            n_samples=len(humans),
            n_features=10,
            centers=None,
            cluster_std=1.0,
            center_box=(-10.0, 10.0),
            shuffle=True,
            random_state=None
        )
        p = 0
        for h1 in humans:
            self.feature_vector[h1] = blobs[0][p]
            p += 1
        for h1 in humans:
            self.similarity[h1] = {}
            for h2 in humans:
                self.similarity[h1][h2] = self._compute_similarity(h1, h2)

    def build_tribe(self, human, humans, mininum, maximum):
        n = linear_rescale(human.properties.extroversion, mininum, maximum)
        w = [self.similarity[human][h] for h in humans]
        tribe = [human]
        count = 0
        while n > 0 and count < len(humans):
            count += 1
            selected = roulette_selection(humans, w)
            if selected not in tribe:
                tribe.append(selected)
                n -= 1
        return tribe

    def _compute_similarity(self, h1, h2):
        assert len(self.feature_vector[h1]) == len(self.feature_vector[h2])
        d = np.linalg.norm(self.feature_vector[h1] - self.feature_vector[h2])
        return linear_rescale(d, 1, 0, 0, math.sqrt(len(self.feature_vector[h1])))
