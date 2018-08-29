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

class TupleFeatureStudyResultEntry(object):
    def __init__(self, feature_label, avg_fscore):
        self.feature_label = feature_label
        self.avg_fscore = avg_fscore

class TupleFeatureStudy(object):

    def __init__(self, data_matrix, labels, params_list, column_labels, column_descriptions = None, creature_type=SVMCreature, tuple_size=2, max_threads=8):
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
        self._tuple_size = tuple_size
        self._max_threads = max_threads
        
    def _create_tuples(self):
        feature_list = []
        # Lazy... we should do it better at some time
        if self._tuple_size == 1:
            for i in range(self._feature_count):
                features = ""
                for k in range(self._feature_count):
                    if i == k:
                        features = features + "1"
                    else:
                        features = features + "0"
                feature_list.append(features)            
        elif self._tuple_size == 2:
            for i1 in range(self._feature_count-1):
                for i2 in range(i1+1, self._feature_count):
                    features = ""
                    for k in range(self._feature_count):
                        if (i1 == k) or (i2 == k):
                            features = features + "1"
                        else:
                            features = features + "0"
                    feature_list.append(features)
        return feature_list
                    
    def compute(self):
        feature_list = self._create_tuples()
        lock = threading.Lock()
        threads = []   
        sleep_event = threading.Event()   
        creatures = [];
        counter = 0
        for features in feature_list:
            counter = counter + 1
            print(repr(counter)+"/"+repr(len(feature_list)))
            # Make sure we have room for another thread
            while True:
                lock.acquire()
                active_threads = len(threads)
                lock.release()                                    
                if active_threads < self._max_threads:
                    break                    
                else:
                    sleep_event.wait(1)
            creature = self._creature_type(self._data_matrix, self._labels, IOUtil.get_feature_from_string(features), self._params_list)
            creatures.append(creature);
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
        avg_feature_score = [0 for i in range(self._feature_count)]
        avg_feature_score_div = [0 for i in range(self._feature_count)]
        for creature in creatures:
            for i in range(len(creature.get_best_params_result().features)):
                if creature.get_best_params_result().features[i] == True:
                    avg_feature_score[i] = avg_feature_score[i] + creature.get_best_params_result().avg_fscore
                    avg_feature_score_div[i] = avg_feature_score_div[i] + 1
        feature_results = []
        for i in range(self._feature_count):
            avg_feature_score[i] = avg_feature_score[i] / avg_feature_score_div[i]
            res = TupleFeatureStudyResultEntry(self._column_labels[i], avg_feature_score[i])
            feature_results.append(res)
        feature_results = sorted(feature_results, key=lambda entry: entry.avg_fscore, reverse=True)
        if self._column_descriptions is None:
            for res in feature_results:
                print(res.feature_label+"\t"+repr(res.avg_fscore))
        else:
            for res in feature_results:
                print(res.feature_label+"\t"+repr(res.avg_fscore)+"\t"+self._column_descriptions[res.feature_label])
        