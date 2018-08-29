import codecs

class ParamsResult(object):
    
    def __init__(self, labels, features, params):
        self.labels = labels
        self.features = features
        self.active_feature_count = 0
        self.params = params        
        self.predicted_labels = [0 for x in range(len(labels))]        
        self.fscores = dict()
        self.precisions = dict()
        self.recalls = dict()        
        self.avg_fscore = 0.0
        self.avg_precision = 0.0
        self.avg_recall = 0.0    
    
    def get_features_string(self):
        res = ""
        for i in self.features:
            if (i == True):
                res = res + "1"
            else:
                res = res + "0"
        return res
    
    def get_active_feature_ratio(self):
        return self.active_feature_count/len(self.features)
    
    def print_errors(self, document_labels=None, category_label_map=None, save_file=None, plot_file=None, min_edge_width=1, max_edge_width=20):
        file = None
        if not (save_file is None):
            file = codecs.open(save_file, "w", "utf-8")
                
        print("Vector-ID\tGold\tPredicted\tLabel (Optional)")
        mismatch_count_map = dict()
        for i in range(len(self.labels)):
            if self.labels[i] != self.predicted_labels[i]:
                key = repr(self.labels[i])+"\t"+repr(self.predicted_labels[i])
                if not (key in mismatch_count_map):
                    mismatch_count_map[key] = 1
                else:
                    mismatch_count_map[key] = mismatch_count_map[key] + 1
                line = ""
                if not(document_labels is None):
                    if not (category_label_map is None):
                        line = repr(i)+"\t"+category_label_map[self.labels[i]]+"\t"+category_label_map[self.predicted_labels[i]]+"\t"+document_labels[i]
                    else:
                        line = repr(i)+"\t"+repr(self.labels[i])+"\t"+repr(self.predicted_labels[i])+"\t"+document_labels[i]
                else:
                    if not (category_label_map is None):
                        line = repr(i)+"\t"+category_label_map[self.labels[i]]+"\t"+category_label_map[self.predicted_labels[i]]+"\t"+repr(i)
                    else:
                        line = repr(i)+"\t"+repr(self.labels[i])+"\t"+repr(self.predicted_labels[i])+"\t"+repr(i)
                print(line)
                if not (save_file is None):
                    file.write(line+"\n")
                    
        if not (save_file is None):
            file.close()
            
        if not (plot_file is None):
            gold_freq_map = dict()
            pred_freq_map = dict()
            for i in self.labels:
                if not (i in gold_freq_map):
                    gold_freq_map[i] = 1
                else:
                    gold_freq_map[i] = gold_freq_map[i] + 1                    
            for i in self.predicted_labels:
                if not (i in pred_freq_map):
                    pred_freq_map[i] = 1
                else:
                    pred_freq_map[i] = pred_freq_map[i] + 1
            file = codecs.open(plot_file, "w", "utf-8")
            file.write("% arara: xelatex: { shell: yes }\n\\XeTeXinputencoding cp1252\n\\documentclass{article}\n\\input{C:/_Template/template-XE.tex}\n\n\\renewcommand{\\familydefault}{\\sfdefault}\n\n\\AtBeginDocument{\\raggedright}\n\n\\usepackage{tikz}\n\\usetikzlibrary{babel}\n\\usepackage{pgfplots}\n\\pgfplotsset{compat=newest}\n\\usepgfplotslibrary{dateplot}\n\n\\usepackage[active,tightpage]{preview}\n\\PreviewEnvironment{tikzpicture}\n\n\\begin{document}\n\n\\newcommand*{\\Scaling}{0.2}\n\\begin{tikzpicture}[Vertex/.style={circle, line width=1.0, fill=white, minimum size=\\Scaling*4.5cm},\nSeed/.style={circle, line width=1.0, fill=SeminarRot, minimum size=\\Scaling*2cm},\nevery label/.style={rectangle, align=center, minimum width=0.7cm, inner sep=1, font=\\tiny\\ttfamily},\nSeedText/.style={rectangle, fill=SeminarRot, align=center, minimum width=1cm, font=\\ttfamily, text=white},\n% every to/.style={bend left},\nx=\\Scaling,y=\\Scaling,\n>=latex]\n\\pgftransformyscale{-1}\n")
            counter = 0
            for gold in gold_freq_map:
                if not (category_label_map is None):                    
                    file.write("\\node (g"+repr(gold)+") [at={("+repr(counter * 140)+", 200)},  Vertex, fill=SeminarHellGruen, label={center:"+category_label_map[gold]+"\\\\"+"{:1.1f}".format((gold_freq_map[gold]*100)/len(self.labels))+"\\%}]{};\n")
                else:
                    file.write("\\node (g"+repr(gold)+") [at={("+repr(counter * 140)+", 200)},  Vertex, fill=SeminarHellGruen, label={center:"+repr(gold)+"\\\\"+"{:1.1f}".format((gold_freq_map[gold]*100)/len(self.labels))+"\\%}]{};\n")
                counter = counter + 1
            counter = 0
            for pred in pred_freq_map:
                if not (category_label_map is None):                    
                    file.write("\\node (p"+repr(pred)+") [at={("+repr(counter * 140)+", 400)},  Vertex, fill=SeminarHellRot, label={center:"+category_label_map[pred]+"\\\\"+"{:1.1f}".format((pred_freq_map[pred]*100)/len(self.labels))+"\\%}]{};\n")
                else:
                    file.write("\\node (p"+repr(pred)+") [at={("+repr(counter * 140)+", 400)},  Vertex, fill=SeminarHellRot, label={center:"+repr(pred)+"\\\\"+"{:1.1f}".format((pred_freq_map[pred]*100)/len(self.labels))+"\\%}]{};\n")
                counter = counter + 1
            file.write("\\begin{scope}[on background layer]\n")
            max_possible_mismatch = 0
            for gold in gold_freq_map:
                if len(self.labels) - gold_freq_map[gold] > max_possible_mismatch:
                    max_possible_mismatch = len(self.labels) - gold_freq_map[gold] 
            for key in mismatch_count_map:
                source_id = int(key.split("\t")[0])
                target_id = int(key.split("\t")[1])
                value = mismatch_count_map[key]
                value = value/max_possible_mismatch
                value = min_edge_width + (value * (max_edge_width - min_edge_width))
                file.write("\path [->,line width=\Scaling*"+repr(value)+",draw,](g"+repr(source_id)+") to (p"+repr(target_id)+");\n")
            file.write("\\end{scope}\n\\end{tikzpicture}\n\\end{document}\n")
            file.close()                    