from GeneticClassifier.NeuralNetworkCreature import NeuralNetworkCreature
from GeneticClassifier.IOUtil import IOUtil
import random
import codecs
import sys
import copy
import functools
from GeneticClassifier.CreatureThread import CreatureThread
import threading
from asyncio.tasks import sleep

class GeneticFeatureStudy(object):
    
    def __init__(self, data_matrix, labels, params_list, creature_type=NeuralNetworkCreature, max_population=10, max_turns=10, max_threads=16, keep_best_n = 3, mutation_rate=0.1, absolute_feature_toggle_count = None, feature_preset_list=[], optimize_features='min'):
        """
        :data_matrix: 2D data float or integer matrix containing vectors in rows and features in columns
        :labels: 1D array of integer labels matching the feature vectors of the data matrix
        :params_list: list of dictionaries representing parameters to be evaluated by the classifiers    
        :creature_type: Class to be used as classifier
        :max_population: Maximal population for each turn. Must be at least 2*keep_best_n. Defaults to 10
        :max_turns: Number of turns to compute. Defaults to 10.
        :max_threads: Maximal number of threads to use. Default 16
        :keep_best_n: Keep top n creatures of each turn unchanged. They will also be included in addition as a mutated variant.
        :mutation_rate: When iterating over each feature for mutation, each feature will be flipped with a probability of mutation_rate
        :absolute_feature_toggle_count: Absolute number of features to change. Set to an int value and mutation_rate to None to use this mode.
        :feature_preset_list: List of 1011110... Strings which shall be used as preset. For example from previous runs. Defaults to empty list
        :optimize_features: set to 'min' to optimize for minimum number of features or 'max' for maximum number of features. Defaults to 'min' 
        """
        self._data_matrix = data_matrix
        self._labels = labels
        self._params_list = params_list
        self._creature_type = creature_type
        self._max_population = max_population
        self._max_turns = max_turns
        self._max_threads = max_threads
        self._keep_best_n = keep_best_n
        self._mutation_rate = mutation_rate  
        self._absolute_feature_toggle_count = absolute_feature_toggle_count
        self._feature_preset_list = feature_preset_list      
        self._optimize_features = optimize_features
        self._best_performer_params_result_history = []
        self._best_performer_feature_usage = [0 for i in range(len(data_matrix[0]))]
        
        # Do some sanity Checks
        if (mutation_rate is None) and (absolute_feature_toggle_count is None):
            raise Exception("mutation_rate and absolute_feature_toggle_count are None")
        if (mutation_rate is not None) and (absolute_feature_toggle_count is not None):
            raise Exception("mutation_rate and absolute_feature_toggle_count are both not None")
        if (data_matrix is None) or (len(data_matrix) == 0):
            raise Exception("data_matrix is empty")
        if labels is None:
            raise Exception("labels is None")
        if len(data_matrix) != len(labels):
            raise Exception("length mismatch of data_matrix and labels")
        if (params_list is None) or (len(params_list) == 0):
            raise Exception("params_list is empty")
        labels_set = set(labels)
        labels_freq = dict()
        for i in labels_set:
            labels_freq[i] = 0
        for i in labels:
            labels_freq[i] = labels_freq[i] + 1
        for i in labels_set:
            if labels_freq[i] < 2:
                raise Exception("label "+repr(i)+" appears less than 2 times")
        if max_population < keep_best_n*2:
            raise Exception("max_population must be at least keep_best_n * 2")
                  
    def get_best_creature(self):
        return self._population[len(self._population)-1]
                  
    def get_best_performer_params_result_history(self):
        return self._best_performer_params_result_history
    
    def save_best_performer_params_result_history(self, result_file_name, append_category_details=False):
        file = codecs.open(result_file_name, "w", "utf-8")
        file.write("Turn\tFScore\tPrecision\tRecall\tActiveFeatureCount\tActiveFeatureRatio\tFeatures\tParams\n")
        counter = 0
        label_set = set(self._labels)
        for res in self._best_performer_params_result_history:
            counter = counter + 1
            line = repr(counter)
            line = line + "\t" + repr(res.avg_fscore)
            line = line + "\t" + repr(res.avg_precision)
            line = line + "\t" + repr(res.avg_recall)
            line = line + "\t" + repr(res.active_feature_count)
            line = line + "\t" + repr(res.get_active_feature_ratio())
            line = line + "\t" + repr(res.get_features_string())
            for key in res.params:
                line = line + "\t" + key + "="+repr(res.params[key])
            file.write(line+"\n")
        if append_category_details == True:
            file.write("Type=General\n")
            file.write("Category\tPrecision\tRecall\tFScore\n")
            study_genetic_best_array = self.get_best_performer_params_result_history()
            study_genetic_params_result = study_genetic_best_array[len(study_genetic_best_array) - 1]
            for cat in range(0, len(label_set)):
                file.write(str(cat) + "\t" + str(study_genetic_params_result.precisions[cat]) + "\t" + str(study_genetic_params_result.recalls[cat]) + "\t" + str(study_genetic_params_result.fscores[cat]) + "\n")
        file.close()
                  
    def get_best_performer_feature_usage(self):
        return self._best_performer_feature_usage
                  
    @staticmethod
    def compare_creatures_minimize_features(x, y):
        res = 0;
        if x.get_best_params_result().avg_fscore < y.get_best_params_result().avg_fscore:
            res = -1
        elif x.get_best_params_result().avg_fscore > y.get_best_params_result().avg_fscore:
            res = 1
        else:
            if x.get_best_params_result().active_feature_count < y.get_best_params_result().active_feature_count:
                res = 1
            elif x.get_best_params_result().active_feature_count > y.get_best_params_result().active_feature_count:
                res = -1
            else:
                res = 0
        return res
                
    @staticmethod
    def compare_creatures_maximize_features(x, y):
        res = 0;
        if x.get_best_params_result().avg_fscore < y.get_best_params_result().avg_fscore:
            res = -1
        elif x.get_best_params_result().avg_fscore > y.get_best_params_result().avg_fscore:
            res = 1
        else:
            if x.get_best_params_result().active_feature_count < y.get_best_params_result().active_feature_count:
                res = -1
            elif x.get_best_params_result().active_feature_count > y.get_best_params_result().active_feature_count:
                res = 1
            else:
                res = 0
        return res
                                              
    def compute(self):
        # Initialize First population
        self._population = [None for x in range(self._max_population)]
        # Always include a creature with all features
        self._population[0] = self._creature_type(self._data_matrix, self._labels, IOUtil.get_full_features(len(self._data_matrix[0])), self._params_list)
        # Include preset
        max_preset = len(self._feature_preset_list)
        if (max_preset > self._max_population-1):
            raise Exception("Preset plus complete 1111-vector exceeds max_population")
        for i in range(1, 1+max_preset):
            self._population[i] = self._creature_type(self._data_matrix, self._labels, IOUtil.get_feature_from_string(self._feature_preset_list[i-1]), self._params_list)
        # Fill up with random creatures
        for i in range(1+max_preset, self._max_population):                        
            self._population[i] = self._creature_type(self._data_matrix, self._labels, self._create_random_features(density=random.uniform(0,1)), self._params_list)
                
        for turn in range(0, self._max_turns):
            print("Turn "+repr(turn+1)+"/"+repr(self._max_turns))
            lock = threading.Lock()
            threads = []   
            sleep_event = threading.Event()                     
            for creature in self._population:                
                if not creature.is_computed():
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
            # Sort and handle population
            if self._optimize_features == 'min':
                self._population = sorted(self._population, key=functools.cmp_to_key(self.compare_creatures_minimize_features))
            else:
                self._population = sorted(self._population, key=functools.cmp_to_key(self.compare_creatures_maximize_features))
            self._population[len(self._population)-1].pretty_print()
            self._best_performer_params_result_history.append(self._population[len(self._population)-1].get_best_params_result())
            for i in range(len(self._population[len(self._population)-1].get_best_params_result().features)):
                if (i == True):
                    self._best_performer_feature_usage[i] = self._best_performer_feature_usage[i] + 1
            if turn < (self._max_turns -1): 
                population_new = []
                # Keep top n unchanged
                for i in range(self._max_population - self._keep_best_n, self._max_population):
                    population_new.append(self._population[i])
                # Randomize the ones we want to keep
                for i in range(self._keep_best_n, self._max_population):
                    # Make a copy of the features since we are going to manipulate them
                    features = copy.deepcopy(self._population[i].get_features())
                    if (self._mutation_rate is not None):
                        while (True):
                            active_count = 0
                            for k in range(len(features)):                    
                                if random.uniform(0,1) < self._mutation_rate:
                                    features[k] = not features[k]
                                    if features[k] == True:
                                        active_count = active_count + 1
                            if (active_count > 1):
                                break
                    if (self._absolute_feature_toggle_count is not None):
                        toggle_set = set()
                        for k in range(0, self._absolute_feature_toggle_count):
                            while True:                                
                                while True:
                                    m = random.randint(0, len(features)-1)
                                    if not m in toggle_set:
                                        toggle_set.add(m)
                                        features[m] = not features[m]
                                        break 
                                active_count = 0
                                for m in range(len(features)):
                                    if features[m] == True:
                                        active_count = active_count + 1
                                if (active_count > 1):
                                    break
                    population_new.append(self._creature_type(self._data_matrix, self._labels, features, self._params_list))             
                self._population = population_new    
    
    def _create_random_features(self, density=0.5):
        res = [False for x in range(len(self._data_matrix[0]))]
        # Ensure that at least one feature is set
        while True:
            active_count = 0
            for i in range(len(res)):
                if random.uniform(0,1) >= 0.5:
                    res[i] = True
                    active_count = active_count + 1
            if active_count > 0:
                break
        return res