from GeneticClassifier.ParamsResult import ParamsResult

class EvalUtil(object):
    
    @staticmethod
    def process_params_result(params_result):
        # Get set of labels
        label_set = set(params_result.labels)
        # Init Maps to count relevant, retrieved and retrieved_relevant
        relevant = dict()
        retrieved = dict()
        retrieved_relevant = dict()
        # Init maps from label to their respective precision, recall and fscore
        params_result.precisions = dict()
        params_result.recalls = dict()
        params_result.fscores = dict()
        # Init values
        for x in label_set:
            relevant[x] = 0
            retrieved[x] = 0
            retrieved_relevant[x] = 0     
            params_result.precisions[x] = 0.0
            params_result.recalls[x] = 0.0
            params_result.fscores[x] = 0.0        
        params_result.avg_fscore = 0.0
        params_result.avg_precision = 0.0
        params_result.avg_recall = 0.0    
        # Sum up values    
        for label in params_result.labels:
            relevant[label] = relevant[label] + 1
        for i in range(len(params_result.predicted_labels)):
            # Sum up retrieved
            retrieved[params_result.predicted_labels[i]] = retrieved[params_result.predicted_labels[i]] + 1
            # Sum up retrieved and relevant
            if (params_result.predicted_labels[i] == params_result.labels[i]):
                retrieved_relevant[params_result.predicted_labels[i]] = retrieved_relevant[params_result.predicted_labels[i]] + 1        
        for label in label_set:
            if retrieved[label] > 0:
                params_result.precisions[label] = retrieved_relevant[label] / retrieved[label]
            params_result.recalls[label] = retrieved_relevant[label] / relevant[label]
            if (params_result.precisions[label] + params_result.recalls[label] > 0) :
                params_result.fscores[label] = (2 * params_result.precisions[label] * params_result.recalls[label]) / (params_result.precisions[label] + params_result.recalls[label])
            params_result.avg_fscore = params_result.avg_fscore + params_result.fscores[label]
            params_result.avg_precision = params_result.avg_precision + params_result.precisions[label]
            params_result.avg_recall = params_result.avg_recall + params_result.recalls[label]
        params_result.avg_fscore = params_result.avg_fscore / len(label_set)
        params_result.avg_precision = params_result.avg_precision / len(label_set)
        params_result.avg_recall = params_result.avg_recall / len(label_set)        
    
    @staticmethod
    def get_classification_performance(labels_gold, labels_predicted):
        result = ParamsResult(labels_gold, [], [])
        result.predicted_labels = labels_predicted
        EvalUtil.process_params_result(result)
        return result
        