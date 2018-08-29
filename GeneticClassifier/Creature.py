import threading
from GeneticClassifier.ParamsResult import ParamsResult
from GeneticClassifier.EvalUtil import EvalUtil

class Creature(object):
    
    def __init__(self, data_matrix, labels, features, params_list):
        """
        :data_matrix: 2D data float or integer matrix containing vectors in rows and features in columns
        :labels: 1D array of integer labels matching the feature vectors of the data matrix
        :features: Boolean Features vector specifying which features should actually be used. The data_matrix must be complete. Filtering will be done by Creature.
        :params_list: list of dictionaries representing parameters to be evaluated by the classifiers
        """
        self._data_matrix = data_matrix
        self._labels = labels
        self._features = features
        self._computed = False        
        self._params_list = params_list
        self._params_result_list = [ParamsResult(labels, features, p) for p in params_list]
        self._best_params_result = None
        self.__actual_feature_count = 0
        self.__actual_data_matrix = None
        self.__compute_actual_data_matrix()        
        
        # Do some sanity Checks
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
        
    def _commit_params_computation(self, params_list_index):
        """
        Compute ML-Measures for specified params_list_index
        :params_list_index: Index of params_result_list to process
        """
        current_params_result = self._params_result_list[params_list_index]
        current_params_result.active_feature_count = 0
        for f in current_params_result.features:
            if f == True:
                current_params_result.active_feature_count = current_params_result.active_feature_count + 1 
        #
        EvalUtil.process_params_result(current_params_result)
        #label_set = set(current_params_result.labels)
        #relevant = dict()
        #retrieved = dict()
        #retrieved_relevant = dict()
        #current_params_result.precisions = dict()
        #current_params_result.recalls = dict()
        #current_params_result.fscores = dict()
        #for x in label_set:
        #    relevant[x] = 0
        #    retrieved[x] = 0
        #    retrieved_relevant[x] = 0     
        #    current_params_result.precisions[x] = 0.0
        #    current_params_result.recalls[x] = 0.0
        #    current_params_result.fscores[x] = 0.0        
        #current_params_result.avg_fscore = 0.0
        #current_params_result.avg_precision = 0.0
        #current_params_result.avg_recall = 0.0    
        ## Sum up values    
        #for label in current_params_result.labels:
        #    relevant[label] = relevant[label] + 1
        #for i in range(len(current_params_result.predicted_labels)):
        #    # Sum up retrieved
        #    retrieved[current_params_result.predicted_labels[i]] = retrieved[current_params_result.predicted_labels[i]] + 1
        #    # Sum up retrieved and relevant
        #    if (current_params_result.predicted_labels[i] == current_params_result.labels[i]):
        #        retrieved_relevant[current_params_result.predicted_labels[i]] = retrieved_relevant[current_params_result.predicted_labels[i]] + 1        
        #for label in label_set:
        #    if retrieved[label] > 0:
        #        current_params_result.precisions[label] = retrieved_relevant[label] / retrieved[label]
        #    current_params_result.recalls[label] = retrieved_relevant[label] / relevant[label]
        #    if (current_params_result.precisions[label] + current_params_result.recalls[label] > 0) :
        #        current_params_result.fscores[label] = (2 * current_params_result.precisions[label] * current_params_result.recalls[label]) / (current_params_result.precisions[label] + current_params_result.recalls[label])
        #    current_params_result.avg_fscore = current_params_result.avg_fscore + current_params_result.fscores[label]
        #    current_params_result.avg_precision = current_params_result.avg_precision + current_params_result.precisions[label]
        #    current_params_result.avg_recall = current_params_result.avg_recall + current_params_result.recalls[label]
        #current_params_result.avg_fscore = current_params_result.avg_fscore / len(label_set)
        #current_params_result.avg_precision = current_params_result.avg_precision / len(label_set)
        #current_params_result.avg_recall = current_params_result.avg_recall / len(label_set)                    
    
    def _complete_computation(self):
        self._best_params_result = None
        best_avg_fscore = -1
        for p in self._params_result_list:
            if p.avg_fscore > best_avg_fscore:
                best_avg_fscore = p.avg_fscore
                self._best_params_result = p
        self._computed = True
        
    def get_params_list(self):
        return self._params_list
    
    def get_best_params_result(self):
        return self._best_params_result
    
    def get_data_matrix(self):
        return self._data_matrix
    
    def get_actual_data_matrix(self):
        return self.__actual_data_matrix
    
    def get_labels(self):
        return self._labels
    
    def get_features(self):
        return self._features
    
    def get_vector_count(self):
        return len(self._data_matrix)
            
    def is_computed(self):
        return self._computed
    
    def get_feature_count(self):
        return len(self._features)
    
    def get_actual_feature_count(self):
        return self.__actual_feature_count
    
    def __compute_actual_data_matrix(self):
        """ Compute the actual_data_matrix from the data_matrix based on the features"""
        self.__actual_feature_count = 0
        if len(self._features) != len(self._data_matrix[0]):
            raise Exception("Feature vector length mismatch")
        for i in self._features:
            if i == True:
                self.__actual_feature_count = self.__actual_feature_count + 1
        self.__actual_data_matrix = [[0.0 for x in range(self.__actual_feature_count)] for y in range(self.get_vector_count())]
        for y in range(self.get_vector_count()):
            x1 = 0
            for x in range(0, self.get_feature_count()):
                if (self._features[x] == True):
                    self.__actual_data_matrix[y][x1] = self._data_matrix[y][x]
                    x1 = x1 + 1    

    def pretty_print(self, print_labels=False):
        print("Computed\t"+repr(self._computed))    
        feature_string = ""
        feat_perc = 0
        for i in self._best_params_result.features:
            if i == True:
                feature_string = feature_string + "1"
                feat_perc = feat_perc + 1
            else:
                feature_string = feature_string + "0"
        feat_perc = (feat_perc*100.0)/len(feature_string)
        print("Features\t"+feature_string+"\t"+repr(feat_perc)+"%")
        print("Parameters:")
        for key in sorted(self._best_params_result.params):
            print("\t"+key+"\t"+repr(self._best_params_result.params[key]))
        label_set = set(self._best_params_result.labels)
        print("Label\tPrecision\tRecall\tFScore")
        for label in sorted(label_set):
            print(repr(label)+"\t"+repr(self._best_params_result.precisions[label])+"\t"+repr(self._best_params_result.recalls[label])+"\t"+repr(self._best_params_result.fscores[label]))
        print("Avg Precision\t"+repr(self._best_params_result.avg_precision))
        print("Avg Recall\t"+repr(self._best_params_result.avg_recall))
        print("Avg FScore\t"+repr(self._best_params_result.avg_fscore))
        if print_labels == True:
            print("Gold\tPredicted")
            for i in range(len(self._best_params_result.labels)):
                print(repr(self._best_params_result.labels[i])+"\t"+repr(self._best_params_result.predicted_labels[i]))
            
    def compute(self):      
        pass  