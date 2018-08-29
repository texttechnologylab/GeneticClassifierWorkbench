import math

from GeneticClassifier.IOUtil import IOUtil
from os import listdir
from os.path import isfile, isdir, join

class BoxPlot(object):

    def __init__(self, values, label, minlabel=None, maxlabel=None):
        self._values = sorted(values)
        self._label = label
        self._median = self._get_quartil(0.5)
        self._min = self._values[0]
        self._max = self._values[len(values)-1]
        self._lower_quartile = self._get_quartil(0.25)
        self._upper_quartile = self._get_quartil(0.75)
        self._minlabel = minlabel
        self._maxlabel = maxlabel

    def _get_quartil(self, fraction):
        if (len(self._values)-1) * fraction == math.floor((len(self._values)-1) * fraction):
            return self._values[int((len(self._values)-1) * fraction)]
        else:
            return (self._values[int(math.floor((len(self._values)-1) * fraction))] + self._values[int(math.ceil((len(self._values)-1) * fraction))]) / 2



    def get_values(self):
        return self._values

    def get_label(self):
        return self._label

    def get_median(self):
        return self._median

    def get_min(self):
        return self._min

    def get_max(self):
        return self._max

    def get_lower_quartile(self):
        return self._lower_quartile

    def get_upper_quartile(self):
        return self._upper_quartile

    def get_tikz(self):
        base = "% "+(self._label.replace("_", "\\_"))+"\n\\addplot+[\n  boxplot prepared={\n    median="+repr(self._median)+",\n    upper quartile="+repr(self._upper_quartile)+",\n    lower quartile="+repr(self._lower_quartile)+",\n    upper whisker="+repr(self._max)+",\n    lower whisker="+repr(self._min)+"\n  }\n] coordinates {}"
        if not self._maxlabel is None:
            base = base + " node[align=left,black,rotate=45,anchor=west] at (boxplot box cs: \\boxplotvalue{upper whisker}, 0) {\\tiny "+self._maxlabel.replace("_", "\\_")+"} "
        if not self._minlabel is None:
            base = base + " node[align=left,black,rotate=45,anchor=west] at (boxplot box cs: \\boxplotvalue{lower whisker}, 0) {\\tiny "+self._minlabel.replace("_", "\\_")+"}"
        base = base + ";"
        return base

def main(argv=None):
    fsc_all = IOUtil.get_float_column("k:/Wiki/StadtWikis/DDC/Classification/SimilarityGraph/Results/Summary.txt", column=2, ignore_first_line=False)
    fsc_opt = IOUtil.get_float_column("k:/Wiki/StadtWikis/DDC/Classification/SimilarityGraph/Results/Summary.txt", column=5, ignore_first_line=False)
    fsc_min = IOUtil.get_float_column("k:/Wiki/StadtWikis/DDC/Classification/SimilarityGraph/Results/Summary.txt", column=10, ignore_first_line=False)
    #
    f_scores_b2_all = []
    f_scores_b2_all_minval = 1
    f_scores_b2_all_maxval = 0
    f_scores_b2_all_minlabel = ""
    f_scores_b2_all_maxlabel= ""
    for d in listdir("k:/Wiki/StadtWikis/DDC/Classification/SimilarityGraph/RandomGraph/StatsAllFeatures"):
        if isdir("k:/Wiki/StadtWikis/DDC/Classification/SimilarityGraph/RandomGraph/StatsAllFeatures/"+d):
            for f in listdir("k:/Wiki/StadtWikis/DDC/Classification/SimilarityGraph/RandomGraph/StatsAllFeatures/"+d):
                if f.endswith(".txt"):
                    vals = IOUtil.get_float_column("k:/Wiki/StadtWikis/DDC/Classification/SimilarityGraph/RandomGraph/StatsAllFeatures/"+d+"/"+f, 1, ignore_first_line=True)
                    val = vals[len(vals)-1]
                    if val < f_scores_b2_all_minval:
                        f_scores_b2_all_minval = val
                        f_scores_b2_all_minlabel = d+"/"+f
                    if val > f_scores_b2_all_maxval:
                        f_scores_b2_all_maxval = val
                        f_scores_b2_all_maxlabel = d+"/"+f
                    f_scores_b2_all.append(val)
    #
    f_scores_b2_min = []
    f_scores_b2_min_minval = 1
    f_scores_b2_min_maxval = 0
    f_scores_b2_min_minlabel = ""
    f_scores_b2_min_maxlabel = ""
    for d in listdir("k:/Wiki/StadtWikis/DDC/Classification/SimilarityGraph/RandomGraph/StatsOptimizedMinimized"):
        if isdir("k:/Wiki/StadtWikis/DDC/Classification/SimilarityGraph/RandomGraph/StatsOptimizedMinimized/" + d):
            for f in listdir("k:/Wiki/StadtWikis/DDC/Classification/SimilarityGraph/RandomGraph/StatsOptimizedMinimized/" + d):
                if f.endswith(".txt"):
                    vals = IOUtil.get_float_column("k:/Wiki/StadtWikis/DDC/Classification/SimilarityGraph/RandomGraph/StatsOptimizedMinimized/" + d + "/" + f, 1, ignore_first_line=True)
                    val = vals[len(vals)-1]
                    if val < f_scores_b2_min_minval:
                        f_scores_b2_min_minval = val
                        f_scores_b2_min_minlabel = d + "/" + f
                    if val > f_scores_b2_min_maxval:
                        f_scores_b2_min_maxval = val
                        f_scores_b2_min_maxlabel = d + "/" + f
                    f_scores_b2_min.append(val)
    #
    f_scores_b3_all = []
    f_scores_b3_all_minval = 1
    f_scores_b3_all_maxval = 0
    f_scores_b3_all_minlabel = ""
    f_scores_b3_all_maxlabel = ""
    for f in listdir("k:/Wiki/StadtWikis/DDC/Classification/SimilarityGraph/Random_43x43/StatsAllFeatures"):
        if f.endswith(".txt"):
            vals = IOUtil.get_float_column("k:/Wiki/StadtWikis/DDC/Classification/SimilarityGraph/Random_43x43/StatsAllFeatures/"+f, 1, ignore_first_line=True)
            val = vals[len(vals)-1]
            if val < f_scores_b3_all_minval:
                f_scores_b3_all_minval = val
                f_scores_b3_all_minlabel = f
            if val > f_scores_b3_all_maxval:
                f_scores_b3_all_maxval = val
                f_scores_b3_all_maxlabel = f
            f_scores_b3_all.append(val)
    #
    f_scores_b3_min = []
    f_scores_b3_min_minval = 1
    f_scores_b3_min_maxval = 0
    f_scores_b3_min_minlabel = ""
    f_scores_b3_min_maxlabel = ""
    for f in listdir("k:/Wiki/StadtWikis/DDC/Classification/SimilarityGraph/Random_43x43/StatsOptimizedMinimized"):
        if f.endswith(".txt"):
            vals = IOUtil.get_float_column("k:/Wiki/StadtWikis/DDC/Classification/SimilarityGraph/Random_43x43/StatsOptimizedMinimized/" + f, 1, ignore_first_line=True)
            val = vals[len(vals)-1]
            if val < f_scores_b3_min_minval:
                f_scores_b3_min_minval = val
                f_scores_b3_min_minlabel = f
            if val > f_scores_b3_min_maxval:
                f_scores_b3_min_maxval = val
                f_scores_b3_min_maxlabel = f
            f_scores_b3_min.append(val)
    #
    f_scores_b4_all = []
    f_scores_b4_all_minval = 1
    f_scores_b4_all_maxval = 0
    f_scores_b4_all_minlabel = ""
    f_scores_b4_all_maxlabel = ""
    for d in listdir("k:/Wiki/StadtWikis/DDC/Classification/SimilarityGraph/RandomGoldRandomClasses/StatsAllFeatures"):
        if isdir("k:/Wiki/StadtWikis/DDC/Classification/SimilarityGraph/RandomGoldRandomClasses/StatsAllFeatures/" + d):
            for f in listdir("k:/Wiki/StadtWikis/DDC/Classification/SimilarityGraph/RandomGoldRandomClasses/StatsAllFeatures/" + d):
                if f.endswith(".txt"):
                    vals = IOUtil.get_float_column("k:/Wiki/StadtWikis/DDC/Classification/SimilarityGraph/RandomGoldRandomClasses/StatsAllFeatures/" + d + "/" + f, 1, ignore_first_line=True)
                    val = vals[len(vals)-1]
                    if val < f_scores_b4_all_minval:
                        f_scores_b4_all_minval = val
                        f_scores_b4_all_minlabel = d + "/" + f
                    if val > f_scores_b4_all_maxval:
                        f_scores_b4_all_maxval = val
                        f_scores_b4_all_maxlabel = d + "/" + f
                    f_scores_b4_all.append(val)
    #
    f_scores_b4_min = []
    f_scores_b4_min_minval = 1
    f_scores_b4_min_maxval = 0
    f_scores_b4_min_minlabel = ""
    f_scores_b4_min_maxlabel = ""
    for d in listdir("k:/Wiki/StadtWikis/DDC/Classification/SimilarityGraph/RandomGoldRandomClasses/StatsOptimizedMinimized"):
        if isdir("k:/Wiki/StadtWikis/DDC/Classification/SimilarityGraph/RandomGoldRandomClasses/StatsOptimizedMinimized/" + d):
            for f in listdir("k:/Wiki/StadtWikis/DDC/Classification/SimilarityGraph/RandomGoldRandomClasses/StatsOptimizedMinimized/" + d):
                if f.endswith(".txt"):
                    vals = IOUtil.get_float_column("k:/Wiki/StadtWikis/DDC/Classification/SimilarityGraph/RandomGoldRandomClasses/StatsOptimizedMinimized/" + d + "/" + f, 1, ignore_first_line=True)
                    val = vals[len(vals)-1]
                    if val < f_scores_b4_min_minval:
                        f_scores_b4_min_minval = val
                        f_scores_b4_min_minlabel = d + "/" + f
                    if val > f_scores_b4_min_maxval:
                        f_scores_b4_min_maxval = val
                        f_scores_b4_min_maxlabel = d + "/" + f
                    f_scores_b4_min.append(val)

    fsc_all_boxplot = BoxPlot(fsc_all, "fsc_all_boxplot")
    fsc_opt_boxplot = BoxPlot(fsc_opt, "fsc_opt")
    fsc_min_boxplot = BoxPlot(fsc_min, "fsc_min")
    f_scores_b2_all_boxplot = BoxPlot(f_scores_b2_all, "f_scores_b2_all", minlabel = f_scores_b2_all_minlabel, maxlabel = f_scores_b2_all_maxlabel)
    f_scores_b2_min_boxplot = BoxPlot(f_scores_b2_min, "f_scores_b2_min", minlabel = f_scores_b2_min_minlabel, maxlabel = f_scores_b2_min_maxlabel)
    f_scores_b3_all_boxplot = BoxPlot(f_scores_b3_all, "f_scores_b3_all", minlabel = f_scores_b3_all_minlabel, maxlabel = f_scores_b3_all_maxlabel)
    f_scores_b3_min_boxplot = BoxPlot(f_scores_b3_min, "f_scores_b3_min", minlabel = f_scores_b3_min_minlabel, maxlabel = f_scores_b3_min_maxlabel)
    f_scores_b4_all_boxplot = BoxPlot(f_scores_b4_all, "f_scores_b4_all", minlabel = f_scores_b4_all_minlabel, maxlabel = f_scores_b4_all_maxlabel)
    f_scores_b4_min_boxplot = BoxPlot(f_scores_b4_min, "f_scores_b4_min", minlabel = f_scores_b4_min_minlabel, maxlabel = f_scores_b4_min_maxlabel)

    print(fsc_all_boxplot.get_tikz())
    print(fsc_opt_boxplot.get_tikz())
    print(fsc_min_boxplot.get_tikz())
    print(f_scores_b2_all_boxplot.get_tikz())
    print(f_scores_b2_min_boxplot.get_tikz())
    print(f_scores_b3_all_boxplot.get_tikz())
    print(f_scores_b3_min_boxplot.get_tikz())
    print(f_scores_b4_all_boxplot.get_tikz())
    print(f_scores_b4_min_boxplot.get_tikz())

def ddcplots(argv=None):
    fsc_all = IOUtil.get_float_column("k:/Wiki/StadtWikis/DDC/Classification/DDCResults/Summary.txt", column=3, ignore_first_line=False)
    fsc_opt = IOUtil.get_float_column("k:/Wiki/StadtWikis/DDC/Classification/SimilarityGraph/Results/Summary.txt", column=6, ignore_first_line=False)
    fsc_min = IOUtil.get_float_column("k:/Wiki/StadtWikis/DDC/Classification/SimilarityGraph/Results/Summary.txt", column=11, ignore_first_line=False)
    #
    f_scores_b3_all = []
    for f in listdir("k:/Wiki/StadtWikis/DDC/Classification/Random_43x98/StatsAllFeatures"):
        if f.endswith(".txt"):
            vals = IOUtil.get_float_column("k:/Wiki/StadtWikis/DDC/Classification/Random_43x98/StatsAllFeatures/"+f, 1, ignore_first_line=True)
            f_scores_b3_all.append(vals[len(vals)-1])
    #
    f_scores_b3_min = []
    for f in listdir("k:/Wiki/StadtWikis/DDC/Classification/Random_43x98/StatsOptimizedMinimized"):
        if f.endswith(".txt"):
            vals = IOUtil.get_float_column("k:/Wiki/StadtWikis/DDC/Classification/Random_43x98/StatsOptimizedMinimized/" + f, 1, ignore_first_line=True)
            f_scores_b3_min.append(vals[len(vals)-1])
    #
    f_scores_b4_all = []
    for d in listdir("k:/Wiki/StadtWikis/DDC/Classification/DDCRandomGoldRandomClasses/StatsAllFeatures"):
        if isdir("k:/Wiki/StadtWikis/DDC/Classification/DDCRandomGoldRandomClasses/StatsAllFeatures/" + d):
            for f in listdir("k:/Wiki/StadtWikis/DDC/Classification/DDCRandomGoldRandomClasses/StatsAllFeatures/" + d):
                if f.endswith(".txt"):
                    vals = IOUtil.get_float_column("k:/Wiki/StadtWikis/DDC/Classification/DDCRandomGoldRandomClasses/StatsAllFeatures/" + d + "/" + f, 1, ignore_first_line=True)
                    f_scores_b4_all.append(vals[len(vals)-1])
    #
    f_scores_b4_min = []
    for d in listdir("k:/Wiki/StadtWikis/DDC/Classification/DDCRandomGoldRandomClasses/StatsOptimizedMinimized"):
        if isdir("k:/Wiki/StadtWikis/DDC/Classification/DDCRandomGoldRandomClasses/StatsOptimizedMinimized/" + d):
            for f in listdir("k:/Wiki/StadtWikis/DDC/Classification/DDCRandomGoldRandomClasses/StatsOptimizedMinimized/" + d):
                if f.endswith(".txt"):
                    vals = IOUtil.get_float_column("k:/Wiki/StadtWikis/DDC/Classification/DDCRandomGoldRandomClasses/StatsOptimizedMinimized/" + d + "/" + f, 1, ignore_first_line=True)
                    f_scores_b4_min.append(vals[len(vals)-1])
    fsc_all_boxplot = BoxPlot(fsc_all, "fsc_all_boxplot")
    fsc_opt_boxplot = BoxPlot(fsc_opt, "fsc_opt")
    fsc_min_boxplot = BoxPlot(fsc_min, "fsc_min")
    f_scores_b3_all_boxplot = BoxPlot(f_scores_b3_all, "f_scores_b3_all")
    f_scores_b3_min_boxplot = BoxPlot(f_scores_b3_min, "f_scores_b3_min")
    f_scores_b4_all_boxplot = BoxPlot(f_scores_b4_all, "f_scores_b4_all")
    f_scores_b4_min_boxplot = BoxPlot(f_scores_b4_min, "f_scores_b4_min")

    print(fsc_all_boxplot.get_tikz())
    print(fsc_opt_boxplot.get_tikz())
    print(fsc_min_boxplot.get_tikz())
    print(f_scores_b3_all_boxplot.get_tikz())
    print(f_scores_b3_min_boxplot.get_tikz())
    print(f_scores_b4_all_boxplot.get_tikz())
    print(f_scores_b4_min_boxplot.get_tikz())

if __name__ == '__main__':
    main()