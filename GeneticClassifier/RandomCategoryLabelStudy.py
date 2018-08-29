import random
import copy
from GeneticClassifier.EvalUtil import EvalUtil
from GeneticClassifier.DistributionStats import DistributionStats
from GeneticClassifier.BoxPlot import BoxPlot
import codecs

class RandomCategoryLabelStudy(object):
    """
    Assign random labels to objects. Either uniform random distribution or by random assignment keeping
    the original cardinality of the target classes  
    """

    def __init__(self, labels, iterations=100, keep_cardinality=True, save_file=None, restore_from_file=None):
        """    
        :labels: 1D array of integer labels matching the feature vectors of the data matrix
        :iterations: number of iterations to compute
        :keep_cardinality: True to keep cardinality of target classes. False for random distribution        
        """        
        self._labels = labels
        self._iterations = iterations
        self._keep_cardinality = keep_cardinality
        self._params_results = []
        self._save_file = save_file
        self._restore_from_file = restore_from_file

    def get_params_results(self):
        return self._params_results
    
    def get_avg_fscore(self):
        return self._avg_fscore
    
    def get_avg_precision(self):
        return self._avg_precision
    
    def get_avg_recall(self):
        return self._avg_recall

    def compute(self):
        f_scores = []
        precisions = []
        recalls = []
        label_set = set(self._labels)
        label_list = [x for x in label_set]
        file = None
        if not (self._save_file is None):
            file = codecs.open(self._save_file, "w", "utf-8")
        if self._restore_from_file is None:
            for i in range(self._iterations):
                prediction = []
                if self._keep_cardinality:
                    prediction = copy.deepcopy(self._labels)
                    random.shuffle(prediction)
                else:
                    for m in range(len(self._labels)):
                        prediction.append(label_list[random.randint(0, len(label_list)-1)])
                if not (self._save_file is None):
                    for p in range(len(prediction)):
                        if p > 0:
                            file.write("\t")
                        file.write(repr(prediction[p]))
                    file.write("\n")
                res = EvalUtil.get_classification_performance(self._labels, prediction)
                self._params_results.append(res)
                f_scores.append(res.avg_fscore)
                precisions.append(res.avg_precision)
                recalls.append(res.avg_recall)
        else:
            lines = None
            with open(self._restore_from_file) as f:
                lines = [line.rstrip() for line in f]
            self._iterations = 0
            for line in lines:
                if len(line) > 0:
                    self._iterations = self._iterations + 1
                    prediction = []
                    fields = line.split("\t")
                    for field in fields:
                        prediction.append(int(field))
                    res = EvalUtil.get_classification_performance(self._labels, prediction)
                    self._params_results.append(res)
                    f_scores.append(res.avg_fscore)
                    precisions.append(res.avg_precision)
                    recalls.append(res.avg_recall)

        if not (self._save_file is None):
            file.close

        # F-Score-Avg	F-Score-Min	F-Score-Max	F-Score-StdDev	Precision-Avg	Precision-Min	Precision-Max	Precision-StdDev	Recall-Avg	Recall-Min	Recall-Max	Recall-StdDev
        print("F-Score-Avg\tF-Score-Min\tF-Score-Max\tF-Score-StdDev\tPrecision-Avg\tPrecision-Min\tPrecision-Max\tPrecision-StdDev\tRecall-Avg\tRecall-Min\tRecall-Max\tRecall-StdDev")
        f_scores_dist = DistributionStats(f_scores)
        precisions_dist = DistributionStats(precisions)
        recalls_dist = DistributionStats(recalls)
        f_score_boxplot = BoxPlot(f_scores, "")

        print(repr(f_scores_dist.avg) + "\t" + repr(f_scores_dist.min) + "\t" + repr(f_scores_dist.max) + "\t" + repr(
            f_scores_dist.stddev) + "\t" + repr(precisions_dist.avg) + "\t" + repr(precisions_dist.min) + "\t" + repr(
            precisions_dist.max) + "\t" + repr(precisions_dist.stddev) + "\t" + repr(recalls_dist.avg) + "\t" + repr(
            recalls_dist.min) + "\t" + repr(recalls_dist.max) + "\t" + repr(recalls_dist.stddev))

        print(f_score_boxplot.get_tikz())