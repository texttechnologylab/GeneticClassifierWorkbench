from GeneticClassifier.Creature import Creature
from sklearn import svm
import copy
from sklearn.neural_network.multilayer_perceptron import MLPClassifier

class NeuralNetworkCreature(Creature):
    
    def __init__(self, data_matrix, labels, features, params_list):       
        super().__init__(data_matrix, labels, features, params_list)
    
    def compute(self):  
         # Iterate Leave-One-Out Index over all vectors
        actual_matrix = self.get_actual_data_matrix()                   
        for params_list_index in range(len(self._params_list)):
            params = self._params_list[params_list_index]
            current_params_result = self._params_result_list[params_list_index]                    
            for loo_index in range(self.get_vector_count()):
                # Prepare data and labels for current leave one out
                train_data = [[0 for x in range(self.get_actual_feature_count())] for y in range(self.get_vector_count()-1)]
                train_labels = [0 for x in range(0, self.get_vector_count()-1)]
                test_data = [[0 for x in range(0, self.get_actual_feature_count())] for y in range(1)]
                test_labels = [0 for x in range(1)]
                y1 = 0
                for y in range(self.get_vector_count()):
                    if (y != loo_index):                
                        for x in range(self.get_actual_feature_count()):
                            train_data[y1][x] = actual_matrix[y][x]
                        train_labels[y1] = self._labels[y]
                        y1 = y1 + 1
                for x in range(self.get_actual_feature_count()):
                    test_data[0][x] = actual_matrix[loo_index][x]
                    test_labels[0] = self._labels[loo_index] 
                            
                #clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
                clf = MLPClassifier(**params)
                clf.fit(train_data, train_labels)
                res = clf.predict(test_data)
                current_params_result.predicted_labels[loo_index] = res[0]            
                #print(repr(self.get_labels()[loo_index])+"\t"+repr(res[0]))
                
            self._commit_params_computation(params_list_index)
        self._complete_computation()                                                    