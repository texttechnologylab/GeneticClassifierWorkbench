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

class MaximalLossFeatureRankingStudy(object):

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
        creatures = []                       

        root_creature = self._creature_type(self._data_matrix, self._labels, IOUtil.get_full_features(len(self._data_matrix[0])), self._params_list)
        root_creature.compute()
        
        for feature_id in range(self._feature_count):
            feature = ""
            for i in range(self._feature_count):
                if i == feature_id:
                    feature = feature + "0"
                else:
                    feature = feature + "1"
            creature = self._creature_type(self._data_matrix, self._labels, IOUtil.get_feature_from_string(feature), self._params_list)
            creatures.append(creature)
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

        # Now take all Features which if removed lower the f-score and compute a creature based on them
        synth_maxloss_feature = ""
        for i in range(len(creatures)):
            if creatures[i].get_best_params_result().avg_fscore < root_creature.get_best_params_result().avg_fscore:
                synth_maxloss_feature = synth_maxloss_feature + "1"
            else:
                synth_maxloss_feature = synth_maxloss_feature + "0"
        
        synth_maxloss_creature = self._creature_type(self._data_matrix, self._labels, IOUtil.get_feature_from_string(synth_maxloss_feature), self._params_list)
        synth_maxloss_creature.compute()
            
        # Other variant: Remove all which removal had a positive impact (thus keeping all the others including the "irrelevant") 
        synth_minloss_feature = ""
        for i in range(len(creatures)):
            if creatures[i].get_best_params_result().avg_fscore <= root_creature.get_best_params_result().avg_fscore:
                synth_minloss_feature = synth_minloss_feature + "1"
            else:
                synth_minloss_feature = synth_minloss_feature + "0"
        
        synth_minloss_creature = self._creature_type(self._data_matrix, self._labels, IOUtil.get_feature_from_string(synth_minloss_feature), self._params_list)
        synth_minloss_creature.compute()

        # To complete the picture: Only take the worst features
        synth_worst_feature = ""
        for i in range(len(creatures)):
            if creatures[i].get_best_params_result().avg_fscore > root_creature.get_best_params_result().avg_fscore:
                synth_worst_feature = synth_worst_feature + "1"
            else:
                synth_worst_feature = synth_worst_feature + "0"
        
        synth_worst_creature = self._creature_type(self._data_matrix, self._labels, IOUtil.get_feature_from_string(synth_worst_feature), self._params_list)
        synth_worst_creature.compute()

        print("Reference\t"+repr(root_creature.get_best_params_result().avg_fscore)+"\t"+root_creature.get_best_params_result().get_features_string())            
        print("SynthMaxLoss\t"+repr(synth_maxloss_creature.get_best_params_result().avg_fscore)+"\t"+synth_maxloss_creature.get_best_params_result().get_features_string())
        print("SynthMinLoss\t"+repr(synth_minloss_creature.get_best_params_result().avg_fscore)+"\t"+synth_minloss_creature.get_best_params_result().get_features_string())
        print("SynthWorst\t"+repr(synth_worst_creature.get_best_params_result().avg_fscore)+"\t"+synth_worst_creature.get_best_params_result().get_features_string())
        for i in range(len(creatures)):
            creature = creatures[i]
            print(repr(i)+"\t"+self._column_labels[i]+"\t"+repr(creature.get_best_params_result().avg_fscore)+"\t"+self._column_descriptions[self._column_labels[i]]+"\t"+creature.get_best_params_result().get_features_string())
