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

class MinimizeLossStudy(object):

    def __init__(self, data_matrix, labels, params_list, column_labels, column_descriptions = None, creature_type=SVMCreature, max_threads=8):
        """
        :data_matrix: 2D data float or integer matrix containing vectors in rows and features in columns
        :labels: 1D array of integer labels matching the feature vectors of the data matrix
        :params_list: list of dictionaries representing parameters to be evaluated by the classifiers
        :column_labels: list of column labels    
        :creature_type: Class to be used as classifier        
        :max_threads: Maximal number of threads to use. Default 16
        """
        self._data_matrix = data_matrix
        self._labels = labels
        self._params_list = params_list
        self._column_labels = column_labels
        self._column_descriptions = column_descriptions
        self._creature_type = creature_type
        self._feature_count = len(data_matrix[0])        
        self._max_threads = max_threads
                    
    def compute(self):        
        lock = threading.Lock()
        threads = []   
        sleep_event = threading.Event()               
        best_creatures = [];
        best_remove_features = [];
        root_creature = self._creature_type(self._data_matrix, self._labels, IOUtil.get_full_features(len(self._data_matrix[0])), self._params_list)
        best_creatures.append(root_creature)
        best_remove_features.append(-1)        
        root_creature.compute()
        
        # Take up to _feature_count-2 features away
        for layer_id in range(self._feature_count-2):
            # Compute Current Layer
            current_creatures = [];
            current_removedfeature = []
            base = best_creatures[layer_id].get_best_params_result().features
            for i in range(len(base)):
                if (base[i] == True):
                    features = copy.deepcopy(base)
                    features[i] = False;
                    current_creatures.append(self._creature_type(self._data_matrix, self._labels, features, self._params_list))
                    current_removedfeature.append(i)
                    
            for creature in current_creatures:  
                # Make sure we have room for another thread
                while True:
                    lock.acquire()
                    active_threads = len(threads)
                    lock.release()                                    
                    if active_threads < self._max_threads:
                        break                    
                    else:
                        sleep_event.wait(1)
                            
                thread = CreatureThread(creature, lock, threads, sleep_event)
                threads.append(thread)
                thread.start()
            # Wait for pending threads to complete                        
            while True:
                lock.acquire()
                active_threads = len(threads)
                lock.release()                                    
                if active_threads == 0:
                    break                    
                else:
                    sleep_event.wait(1)
                        
            sys.stdout.flush()
            
            best_creature = None
            best_avg_score = -1
            best_remove_feature = 0
            for i in range(len(current_creatures)):
                creature = current_creatures[i]
                if creature.get_best_params_result().avg_fscore > best_avg_score:
                    best_avg_score = creature.get_best_params_result().avg_fscore
                    best_creature = creature
                    best_remove_feature = current_removedfeature[i]
            best_creatures.append(best_creature)
            best_remove_features.append(best_remove_feature)
            print("Removed: "+repr(best_remove_feature)+"\t"+repr(best_avg_score))
        
        for i in range(len(best_creatures)):
            creature = best_creatures[i]
            remove_feature = best_remove_features[i]
            if remove_feature != -1:
                print(self._column_labels[remove_feature]+"\t"+repr(creature.get_best_params_result().avg_fscore)+"\t"+self._column_descriptions[self._column_labels[remove_feature]]+"\t"+creature.get_best_params_result().get_features_string())
            else:
                print("-\t"+repr(creature.get_best_params_result().avg_fscore)+"\t-\t"+creature.get_best_params_result().get_features_string())