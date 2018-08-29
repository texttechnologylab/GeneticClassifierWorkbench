from GeneticClassifier.NeuralNetworkCreature import NeuralNetworkCreature
from GeneticClassifier.SVMCreature import SVMCreature
from GeneticClassifier.IOUtil import IOUtil
import random
import codecs
import sys
import copy
import functools
from GeneticClassifier.CreatureThread import CreatureThread
import threading

class CategorySensitivityStudy(object):

    def __init__(self, data_matrix, labels, params_list, creature_type=SVMCreature, max_threads=8, direction="asc"):
        """
        :data_matrix: 2D data float or integer matrix containing vectors in rows and features in columns
        :labels: 1D array of integer labels matching the feature vectors of the data matrix
        :params_list: list of dictionaries representing parameters to be evaluated by the classifiers
        :column_labels: list of column labels    
        :creature_type: Class to be used as classifier        
        :max_threads: Maximal number of threads to use. Default 16
        :direction: asc|desc Ascending to start from 1 and go to many. desc to start from all and reduce to one 
        """
        self._data_matrix = data_matrix
        self._labels = labels
        self._params_list = params_list        
        self._creature_type = creature_type
        self._feature_count = len(data_matrix[0])        
        self._max_threads = max_threads
        self._direction = direction

    def _extract_labels(self, label_set):
        """
        Get sub - labels from complete list based on specified label-set.
        Returned ids may not match the ones from label_set!
        This is because we want to ensure the category labels are continous 0, 1, 2, ...
        That is, if the label_set contains 1,3, the result will only contain labels 0, 1
        """
        doc_count = 0
        catmap = dict()
        for i in self._labels:
            if i in label_set:
                if not (i in catmap):
                    catmap[i] = len(catmap);
                doc_count = doc_count + 1
        result = [0 for y in range(doc_count)]
        doc_count = 0
        y = 0
        for i in self._labels:
            if i in label_set:                                
                result[doc_count] = catmap[self._labels[y]]
                doc_count = doc_count + 1
            y = y + 1
        return result

    def _extract_data_matrix(self, label_set):
        """
        Get sub - data_matrix from complete data_matrix based on specified label-set 
        """
        doc_count = 0
        for i in self._labels:
            if i in label_set:
                doc_count = doc_count + 1
        result = [[0 for x in range(self._feature_count)] for y in range(doc_count)]
        doc_count = 0
        y = 0
        for i in self._labels:
            if i in label_set:                
                for x in range(self._feature_count):
                    result[doc_count][x] = self._data_matrix[y][x]
                doc_count = doc_count + 1
            y = y + 1
        return result
                    
    def compute(self):                
        category_chain_list = []
        category_chain_creature_list = []
        
        if self._direction=="asc":
            label_set = set(self._labels)
            for seed_label in label_set:
                category_chain = []
                category_chain_creature = []
                category_chain_list.append(category_chain)
                category_chain_creature_list.append(category_chain_creature)
                category_chain.append(seed_label)
                category_chain_creature.append(None)
                test_label_set = set()
                test_label_set.add(seed_label)
                for cycle in range(1, len(label_set)):
                    best_category = None
                    best_creature = None
                    for add_category in label_set:
                        if not(add_category in test_label_set):
                            exp_set = set(test_label_set)
                            exp_set.add(add_category)
                            creature = self._creature_type(self._extract_data_matrix(exp_set), self._extract_labels(exp_set), IOUtil.get_full_features(len(self._data_matrix[0])), self._params_list)
                            creature.compute()
                            if best_creature is None:
                                best_creature = creature
                                best_category = add_category
                            elif creature.get_best_params_result().avg_fscore > best_creature.get_best_params_result().avg_fscore:
                                best_creature = creature
                                best_category = add_category
                    category_chain.append(best_category)                    
                    category_chain_creature.append(best_creature)
                    test_label_set.add(best_category)        
            for i in range(len(category_chain_list)):
                if i > 0:
                    print()
                category_chain = category_chain_list[i]
                print(repr(category_chain[0])+"\t1.0")
                for k in range(1, len(category_chain)):
                    print(repr(category_chain[k])+"\t"+repr(category_chain_creature_list[i][k].get_best_params_result().avg_fscore))
        else:
            category_chain = []
            category_chain_creature = []
            label_set = set(self._labels)
            test_label_set = set(label_set)
            base_creature = self._creature_type(self._data_matrix, self._labels, IOUtil.get_full_features(len(self._data_matrix[0])), self._params_list)
            base_creature.compute()
            category_chain.append(None)
            category_chain_creature.append(base_creature)
            for cycle in range(1, len(label_set)-1):
                best_category = None
                best_creature = None
                for remove_category in label_set:
                    if remove_category in test_label_set:
                        exp_set = set(test_label_set)
                        exp_set.remove(remove_category)
                        creature = self._creature_type(self._extract_data_matrix(exp_set), self._extract_labels(exp_set), IOUtil.get_full_features(len(self._data_matrix[0])), self._params_list)
                        creature.compute()
                        if best_creature is None:
                            best_creature = creature
                            best_category = remove_category
                        elif creature.get_best_params_result().avg_fscore > best_creature.get_best_params_result().avg_fscore:
                            best_creature = creature
                            best_category = remove_category
                category_chain.append(best_category)                    
                category_chain_creature.append(best_creature)
                test_label_set.remove(best_category)     
            remaining_category = None
            for i in test_label_set:
                remaining_category = i
               
            print("All\t"+repr(category_chain_creature[0].get_best_params_result().avg_fscore))
            for k in range(1, len(category_chain)):
                print(repr(category_chain[k])+"\t"+repr(category_chain_creature[k].get_best_params_result().avg_fscore))
            print(repr(remaining_category)+"\t1.0")