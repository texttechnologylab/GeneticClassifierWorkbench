from __future__ import division
import sys

from sklearn import svm
from sklearn.model_selection import RepeatedKFold

def print_syntax():
    print("Syntax SciKitSVM <-i <InputFile>> [-g <gamma>] [-c <c-param>] [-d <degree>] [-t <kernel>]")
    print("-i <input-file>")
    print("-g gamma (default 0.1")
    print("-c c-param (default 1.0")
    print("-d degree (default 1")
    print("-t kernel (default 0)")
    print("   0 = linear")
    print("   1 = polynomial")
    print("   2 = rbf")
    sys.exit()

def main(argv=None):    
    if (len(sys.argv) < 2):
        print_syntax()
    inputFile = None
    modelFile = None
    gamma = 0.1
    c = 1.0
    degree = 1
    kernel = "linear"
    param_index = 1
    while param_index < len(sys.argv):
        if (sys.argv[param_index] == "-i"):
            param_index = param_index+1
            inputFile = sys.argv[param_index]
        elif (sys.argv[param_index] == "-o"):
            param_index = param_index+1
            modelFile = sys.argv[param_index]
        elif (sys.argv[param_index] == "-m"):
            param_index = param_index+1
            modelFile = sys.argv[param_index]
        elif (sys.argv[param_index] == "-g"):
            param_index = param_index+1
            gamma = float(sys.argv[param_index])
        elif (sys.argv[param_index] == "-c"):
            param_index = param_index+1
            c = float(sys.argv[param_index])
        elif (sys.argv[param_index] == "-d"):
            param_index = param_index+1
            degree = int(sys.argv[param_index])
        elif (sys.argv[param_index] == "-t"):
            param_index = param_index+1
            if sys.argv[param_index] == "0":
                kernel = "linear"
            elif sys.argv[param_index] == "1":
                kernel = "poly"
            elif sys.argv[param_index] == "2":
                kernel = "rbf"
            else:
                print_syntax()
        else:
            print("Unknown parameter: ", sys.argv[param_index])
            print_syntax()
        param_index = param_index + 1
    lines = None
    with open(inputFile) as f:
        lines = [line.rstrip() for line in f]
    split_lines = [None] * len(lines)
    max_index = 0
    valid_lines = 0
    for i in range(0, len(lines)):                
        fields = lines[i].split(" ")
        split_lines[i] = fields
        if (len(fields) > 1):
            current_maxindex = int(fields[len(fields)-1][:fields[len(fields)-1].find(":")])
            if (current_maxindex > max_index):
                max_index = current_maxindex
            valid_lines = valid_lines + 1
        
    matrix = []
    label = []
    for fields in split_lines:
        if (len(fields) > 1):
            data = [0.0 for x in range(max_index)]
            label.append(int(fields[0]))
            for i in range(1, len(fields)):
                data[int(fields[i][:fields[i].find(":")])-1] = float(fields[i][(fields[i].find(":"))+1:])
            matrix.append(data) 
    
    model = svm.SVC(kernel=kernel, gamma=gamma, C=c)
        
    random_state = 12883824
    rkf = RepeatedKFold(n_splits=len(matrix), n_repeats=1, random_state=random_state)
    
    pred = [0 for x in range(len(matrix))]
    
    for train_index, test_index in rkf.split(matrix):
        X_train = [[0 for x in range(max_index)] for y in range(len(matrix)-1)]
        X_test = [[0 for x in range(max_index)] for y in range(1)]
        label_train = [0 for x in range(len(matrix)-1)]
        label_test = [0 for x in range(1)]
        
        y = 0
        for i in train_index:
            for x in range(max_index):
                X_train[y][x] = matrix[i][x]
            label_train[y] = label[i]
            y = y + 1
        y = 0
        for i in test_index:
            for x in range(max_index):
                X_test[y][x] = matrix[i][x]
            label_test[y] = label[i]
            y = y + 1;                                                
        model.fit(X_train, label_train)
        res = model.predict(X_test)
        pred[test_index[0]] = res[0]        
    
    relevant = 0
    for i in range(len(label)):
        if (label[i] == 1):
            relevant = relevant + 1
    relevant_and_retrieved = 0
    retrieved = 0
    for i in range(len(pred)):
        if (pred[i] == 1):
            retrieved = retrieved + 1
        if ((pred[i] == 1) and (label[i] == 1)):
            relevant_and_retrieved = relevant_and_retrieved + 1
    recall = relevant_and_retrieved/relevant
    precision = 0;
    if (retrieved > 0):        
        precision = relevant_and_retrieved/retrieved;
    
    print("Mean-Recall "+repr(recall))
    print("Mean-Precision "+repr(precision))
    
if __name__ == '__main__':
    main()    
    