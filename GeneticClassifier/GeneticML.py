from GeneticClassifier.IOUtil import IOUtil
from GeneticClassifier.NeuralNetworkCreature import NeuralNetworkCreature 
from GeneticClassifier.SVMCreature import SVMCreature
from GeneticClassifier.GeneticFeatureStudy import GeneticFeatureStudy
from GeneticClassifier.TupleFeatureStudy import TupleFeatureStudy
from GeneticClassifier.MinimizeLossStudy import MinimizeLossStudy
from GeneticClassifier.MaximalLossFeatureRankingStudy import MaximalLossFeatureRankingStudy
from GeneticClassifier.CategorySensitivityStudy import CategorySensitivityStudy
from GeneticClassifier.RandomCategoryLabelStudy import RandomCategoryLabelStudy
from GeneticClassifier.BoxPlot import BoxPlot
import sys
import os
import copy
import random
import codecs
from os import listdir
from os.path import isfile, join, isdir

def main(argv=None):            
    result_lines = []

    mode = None
    data = None    
    labels = None
    params_list = []
    result_filename = None
    data_filename = None

    ic = 1
    while ic < len(sys.argv):
        if sys.argv[ic] == '-m':
            ic = ic + 1
            mode = sys.argv[ic]
        elif sys.argv[ic] == '-d':
            ic = ic + 1
            data_filename = sys.argv[ic]
            data = IOUtil.load_matrix(data_filename)
        elif sys.argv[ic] == '-l':
            ic = ic + 1
            labels = IOUtil.load_labels(sys.argv[ic])
        elif sys.argv[ic] == '-pl':
            ic = ic + 1
            params_list = IOUtil.load_params_list(sys.argv[ic])
        elif sys.argv[ic] == '-cl':
            ic = ic + 1
            column_labels = IOUtil.load_column_labels(sys.argv[ic])
        elif sys.argv[ic] == '-cd':
            ic = ic + 1
            column_descriptions = IOUtil.load_column_descriptions(sys.argv[ic])
        elif sys.argv[ic] == '-r':
            ic = ic + 1
            result_filename = sys.argv[ic]
        ic = ic + 1

    if mode == "graphsimsvmwalktrough":
        extended_mode = True
        label_set = set(labels)
        # Base - Get best Parameters for SVM
        study_base = GeneticFeatureStudy(data, labels, params_list, creature_type=SVMCreature, max_population=1, max_turns=1, keep_best_n=0, max_threads=20, mutation_rate=0.1, absolute_feature_toggle_count=None)
        study_base.compute()
        study_base_best_array = study_base.get_best_performer_params_result_history()
        study_base_params_result = study_base_best_array[len(study_base_best_array)-1]
        best_C = study_base_params_result.params["C"]
        best_gamma = study_base_params_result.params["gamma"]

        # Perform genetic Study
        params_list = []
        param = dict()
        param["C"] = best_C;
        param["gamma"] = best_gamma;
        params_list.append(param);
        study_genetic = None
        if extended_mode == True:
            study_genetic = GeneticFeatureStudy(data, labels, params_list, creature_type=SVMCreature, max_population=20, max_turns=500, keep_best_n=3, max_threads=20, mutation_rate=0.1, absolute_feature_toggle_count=None)
        else:
            study_genetic = GeneticFeatureStudy(data, labels, params_list, creature_type=SVMCreature, max_population=20, max_turns=50, keep_best_n=3, max_threads=20, mutation_rate=0.1, absolute_feature_toggle_count=None)
        study_genetic.compute()
        study_genetic_best_array = study_genetic.get_best_performer_params_result_history()
        study_genetic_params_result = study_genetic_best_array[len(study_genetic_best_array)-1]
        study_genetic_features = study_genetic_params_result.features

        # Perform feature reduction optimization
        study_final = None
        study_final_best_array = None
        study_final_params_result = None
        if extended_mode == True:
            feature_preset_list = []
            feature_string = ""
            for b in study_genetic_features:
                if b == True:
                    feature_string = feature_string + "1"
                else:
                    feature_string = feature_string + "0"
            feature_preset_list.append(feature_string)
            study_final = GeneticFeatureStudy(data, labels, params_list, creature_type=SVMCreature, max_population=20, max_turns=500, keep_best_n=3, max_threads=20, mutation_rate=None, absolute_feature_toggle_count=1, feature_preset_list=feature_preset_list)
            study_final.compute()
            study_final_best_array = study_final.get_best_performer_params_result_history()
            study_final_params_result = study_final_best_array[len(study_final_best_array)-1]

        # Write Results
        result_file = open(result_filename, "w")
        result_file.write(repr(best_C)+"\t"+repr(best_gamma)+"\t"+repr(study_base_params_result.avg_fscore)+"\t"+repr(study_base_params_result.avg_precision)+"\t"+repr(study_base_params_result.avg_recall))
        result_file.write("\t"+repr(study_genetic_params_result.avg_fscore)+"\t"+repr(study_genetic_params_result.avg_precision)+"\t"+repr(study_genetic_params_result.avg_recall)+"\t")
        for b in study_genetic_params_result.features:
            if b == True:
                result_file.write("1")
            else:
                result_file.write("0")
        result_file.write("\t")
        result_file.write(repr((study_genetic_params_result.active_feature_count/len(study_genetic_params_result.features))*100))
        if extended_mode == True:
            result_file.write("\t" + repr(study_final_params_result.avg_fscore) + "\t" + repr(study_final_params_result.avg_precision) + "\t" + repr(study_final_params_result.avg_recall) + "\t")
            for b in study_final_params_result.features:
                if b == True:
                    result_file.write("1")
                else:
                    result_file.write("0")
            result_file.write("\t")
            result_file.write(repr((study_final_params_result.active_feature_count / len(study_final_params_result.features)) * 100))
        result_file.write("\n")
        #
        result_file.write("Type=AllFeatures\n")
        result_file.write("Category\tPrecision\tRecall\tFScore\n")
        for cat in range(0, len(label_set)):
            result_file.write(str(cat)+"\t"+str(study_base_params_result.precisions[cat])+"\t"+str(study_base_params_result.recalls[cat])+"\t"+str(study_base_params_result.fscores[cat])+"\n")
        #
        result_file.write("Type=Optimized\n")
        result_file.write("Category\tPrecision\tRecall\tFScore\n")
        for cat in range(0, len(label_set)):
            result_file.write(repr(cat) + "\t" + repr(study_genetic_params_result.precisions[cat]) + "\t" + repr(study_genetic_params_result.recalls[cat]) + "\t" + repr(study_genetic_params_result.fscores[cat]) + "\n")
        #
        if extended_mode == True:
            result_file.write("Type=OptimizedMinimized\n")
            result_file.write("Category\tPrecision\tRecall\tFScore\n")
            for cat in range(0, len(label_set)):
                result_file.write(repr(cat) + "\t" + repr(study_final_params_result.precisions[cat]) + "\t" + repr(study_final_params_result.recalls[cat]) + "\t" + repr(study_final_params_result.fscores[cat]) + "\n")
        result_file.close()

    for line in result_lines:
        print(line)

if __name__ == '__main__':
    main()    
