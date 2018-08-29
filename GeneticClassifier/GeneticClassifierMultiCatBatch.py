from __future__ import division
import sys

from sklearn import svm
from sklearn.model_selection import RepeatedKFold

def print_syntax():
    print("Syntax SciKitSVMMultiCatBatch <-data <DataFile>> <-label <LabelFile>> <-param <ParamFile>> <-feat <features>")    
    sys.exit()

def main(argv=None):    
    if (len(sys.argv) < 2):
        print_syntax()
    
    data_file = None
    label_file = None
    param_file = None
    features_string = None
    
    # Parse Params
    param_index = 1
    while param_index < len(sys.argv):
        if (sys.argv[param_index] == "-data"):
            param_index = param_index+1
            data_file = sys.argv[param_index]
        elif (sys.argv[param_index] == "-label"):
            param_index = param_index+1
            label_file = sys.argv[param_index]
        elif (sys.argv[param_index] == "-param"):
            param_index = param_index+1
            param_file = sys.argv[param_index]
        elif (sys.argv[param_index] == "-feat"):
            param_index = param_index+1
            features_string = sys.argv[param_index]
        param_index = param_index + 1

    # Process feature-selection
    features = [0 for x in range(len(features_string))]
    active_feature_count = 0
    for i in range(0, len(features_string)):
        if (features_string[i] == '1'):
            features[i] = 1
            active_feature_count = active_feature_count + 1
    
    # Load Data File
    lines = None
    with open(data_file) as f:
        lines = [line.rstrip() for line in f]
    feature_count = len(lines[0].split("\t"))
    vectors = []
    for line in lines:
        fields = line.split("\t") 
        if len(fields) ==  feature_count:
            vector = [0 for x in range(active_feature_count)]
            k = 0            
            for i in range(0, feature_count):
                if (features[i] == 1):
                    vector[k] = float(fields[i])
                    k = k + 1
            vectors.append(vector)
    vector_count = len(vectors)    
        
    # Load Label File
    with open(label_file) as f:
        lines = [line.rstrip() for line in f]
    labels = []
    for line in lines:
        if len(line) > 0:
            labels.append(int(line))
    label_set = set(labels)
            
    # Load Param File
    parameter_sets = dict()
    parameter_strings = []
    with open(param_file) as f:
        lines = [line.rstrip() for line in f]
    for line in lines:
        if ((len(line) > 0) and not line.startswith("#")):
            fields = line.split(" ")
            parameter = dict()
            i = 0
            while i < len(fields):
                if (fields[i] == "-t"):
                    i = i+1
                    if (fields[i] == "0"):
                        parameter["kernel"] = "linear"
                    elif (fields[i] == "1"):
                        parameter["kernel"] = "poly"
                    elif (fields[i] == "2"):
                        parameter["kernel"] = "rbf"
                elif (fields[i] == "-c"):
                    i = i+1
                    parameter["C"] = float(fields[i])
                elif (fields[i] == "-g"):
                    i = i+1
                    parameter["gamma"] = float(fields[i])
                elif (fields[i] == "-d"):
                    i = i+1
                    parameter["degree"] = int(fields[i])
                i = i + 1
            parameter_sets[line] = parameter
            parameter_strings.append(line)
    
    random_state = 12883824
    for parameter_string in parameter_strings:
        parameter = parameter_sets[parameter_string]     
        for label in label_set:                
            model = None
            if (parameter["kernel"] == "linear"):
                model = svm.SVC(kernel=parameter["kernel"], C=parameter["C"])
            elif (parameter["kernel"] == "poly"):
                model = svm.SVC(kernel=parameter["kernel"], C=parameter["C"], gamma=parameter["gamma"], degree=parameter["degree"])
            elif (parameter["kernel"] == "rbf"):
                model = svm.SVC(kernel=parameter["kernel"], C=parameter["C"], gamma=parameter["gamma"])
                
            rkf = RepeatedKFold(n_splits=vector_count, n_repeats=1, random_state=random_state)
            
            pred = [0 for x in range(vector_count)]
            
            for train_index, test_index in rkf.split(vectors):
                X_train = [[0 for x in range(active_feature_count)] for y in range(vector_count-1)]
                X_test = [[0 for x in range(active_feature_count)] for y in range(1)]
                label_train = [0 for x in range(vector_count-1)]
                label_test = [0 for x in range(1)]
                
                y = 0
                for i in train_index:
                    for x in range(active_feature_count):
                        X_train[y][x] = vectors[i][x]
                    label_train[y] = -1
                    if labels[i] == label:
                        label_train[y] = 1
                    y = y + 1
                y = 0
                for i in test_index:
                    for x in range(active_feature_count):
                        X_test[y][x] = vectors[i][x]
                    label_test[y] = -1
                    if labels[i] == label:
                        label_test[y] = 1
                    y = y + 1;                                                
                model.fit(X_train, label_train)
                res = model.predict(X_test)
                pred[test_index[0]] = res[0]        
            
            relevant = 0
            for i in range(len(labels)):
                if (labels[i] == label):
                    relevant = relevant + 1
            relevant_and_retrieved = 0
            retrieved = 0
            for i in range(len(pred)):
                if (pred[i] == 1):
                    retrieved = retrieved + 1
                if ((pred[i] == 1) and (labels[i] == label)):
                    relevant_and_retrieved = relevant_and_retrieved + 1
            recall = relevant_and_retrieved/relevant
            precision = 0;
            if (retrieved > 0):        
                precision = relevant_and_retrieved/retrieved;
            print(repr(label)+"\t"+parameter_string+"\t"+repr(recall)+"\t"+repr(precision))            

if __name__ == '__main__':
    main()            