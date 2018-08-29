import codecs
from os import listdir
from os.path import isfile, join

class IOUtil(object):
    
    @staticmethod
    def load_matrix(filename):
        """ Load float value matrix from tab separated text file """        
        lines = None
        with open(filename) as f:
            lines = [line.rstrip() for line in f]
        feature_count = len(lines[0].split("\t"))
        vectors = []
        for line in lines:
            fields = line.split("\t") 
            if len(fields) ==  feature_count:
                vector = [0 for x in range(feature_count)]                
                for i in range(0, feature_count):                    
                    vector[i] = float(fields[i])                        
                vectors.append(vector)
        vector_count = len(vectors)
        return [[vectors[y][x] for x in range(feature_count)] for y in range(vector_count)]
         
    @staticmethod
    def load_labels(filename):
        """ Load int Label vector from file. Each label is expected to be in one line. 
        The order of labels must correspond to the order of vectors in the matrix
        """
        lines = None
        with open(filename) as f:
            lines = [line.rstrip() for line in f]
        labels = []
        for line in lines:
            if len(line) > 0:                
                labels.append(int(line))
        return [labels[x] for x in range(len(labels))]
    
    @staticmethod
    def load_document_labels(filename):
        """ Load Document Label vector from file. Each label is expected to be in one line. 
        The order of labels must correspond to the order of vectors in the matrix
        """
        lines = None
        f = codecs.open(filename, "r", "utf-8")
        lines = [line.rstrip() for line in f]
        f.close()
        labels = []
        for line in lines:
            if len(line) > 0:                
                labels.append(line)
        return [labels[x] for x in range(len(labels))]
    
    @staticmethod
    def load_category_label_map(filename):        
        res = dict()
        with open(filename) as f:
            lines = [line.rstrip() for line in f]        
        for line in lines:
            if len(line) > 0:                
                fields = line.split("\t")
                res[int(fields[0])] = fields[1]
        return res
    
    @staticmethod
    def load_column_labels(filename):
        lines = None
        with open(filename) as f:
            lines = [line.rstrip() for line in f]
        fields = lines[0].split("\t")
        return fields
    
    @staticmethod
    def load_column_descriptions(filename):
        lines = None
        with open(filename) as f:
            lines = [line.rstrip() for line in f]
        res = dict()
        for line in lines:
            if len(line) > 0:
                fields = line.split("\t")
                if len(fields) == 2:
                    res[fields[0]] = fields[1]
        return res
    
    @staticmethod
    def load_features_preset(filename):
        """ Load feature preset from file. These are multiple lines of 10001011111 etc """
        lines = None
        with open(filename) as f:
            lines = [line.rstrip() for line in f]
        features = []
        for line in lines:
            if len(line) > 0:                
                features.append(line)
        return [features[x] for x in range(len(features))]
    
    @staticmethod
    def load_params_list(filename):
        """ Load list of dictionaries representing parameter sets for classifiers
        Each non empty line line represents a parameter set.
        Lines may be commented out by a leading #.
        Each line is read as key-value pairs, separated by whitespaces
        """
        lines = None
        with open(filename) as f:
            lines = [line.rstrip() for line in f]
        result = []
        for line in lines:
            if len(line) > 0:
                if (not line.startswith("#")):
                    params = dict()
                    fields = line.split(" ")
                    if len(fields) > 0:
                        if len(fields) % 2 == 0:
                            i = 0
                            while i<len(fields):
                                value = None
                                try:
                                    value = int(fields[i+1])
                                except:
                                    try:
                                        value = float(fields[i+1])
                                    except:
                                        value = fields[i+1]
                                params[fields[i]] = value
                                i = i + 2
                            result.append(params)
                        else:
                             raise Exception("Invalid parameter line: "+line)
        return result
    
    @staticmethod
    def get_feature_from_string(feature_string):        
        """ Parses a string consisting of 1 and 0 and converts it into a boolean vector """
        return [feature_string[x] == "1" for x in range(len(feature_string))]
    
    @staticmethod
    def get_full_features(count):
        """ Get a boolean feature vector consistingn only of True-values """
        return [True for x in range(count)]

    @staticmethod
    def get_float_column(filename, column = 0, sep = "\t", ignore_first_line = False):
        lines = None
        with open(filename) as f:
            lines = [line.rstrip() for line in f]
        result = []
        for i in range(len(lines)):
            line = lines[i]
            if (i > 0) or not ignore_first_line:
                fields = line.split(sep)
                if len(fields) > column:
                    result.append(float(fields[column]))
        return result

    @staticmethod
    def convert_tex_2_matrix(source_filename, target_filename):
        output = codecs.open(target_filename, "w", "utf-8")
        with open(source_filename) as f:
            lines = [line.rstrip() for line in f]
        result = []
        for i in range(len(lines)):
            line = lines[i]
            line = line.strip()
            if line.startswith("\mchori{") and not(line.startswith("\mchori{}")):
                line = line[line.index("&")+1:line.index("\\\\")].replace("&","\t").replace(" ", "").strip()
                output.write(line+"\n")
        output.close()

    @staticmethod
    def get_latest_dataline(source_filename):
        input = codecs.open(source_filename, "r", "utf-8")
        lines = [line.rstrip() for line in input]
        prev_line = ""
        for line in lines:
            if line.startswith("Type="):
                break
            prev_line = line
        input.close()
        return prev_line

    @staticmethod
    def get_directory_latest_datalines(source_directory):
        files = [f for f in listdir(source_directory) if f.endswith(".txt")]
        result = []
        for file in files:
            result.append(IOUtil.get_latest_dataline(join(source_directory, file)))
        return result

    @staticmethod
    def get_directory_first_lines(source_directory):
        files = [f for f in listdir(source_directory) if f.endswith(".txt")]
        result = []
        for file in files:
            input = codecs.open(join(source_directory, file), "r", "utf-8")
            lines = [line.rstrip() for line in input]
            input.close()
            result.append(lines[0])
        return result

    @staticmethod
    def get_directory_second_lines(source_directory):
        files = [f for f in listdir(source_directory) if f.endswith(".txt")]
        result = []
        for file in files:
            input = codecs.open(join(source_directory, file), "r", "utf-8")
            lines = [line.rstrip() for line in input]
            input.close()
            result.append(lines[1])
        return result