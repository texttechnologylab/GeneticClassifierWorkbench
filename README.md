[![version](https://img.shields.io/github/license/texttechnologylab/reCAPTCHA)]()

# GeneticClassifierWorkbench
```
GeneticClassifier.bat -m graphsimsvmwalktrough

-m execution-mode
-d data matrix file
-l category label file
-pl parameter study file
-r result file
```

# Example
```
GeneticClassifier.bat -m graphsimsvmwalktrough -d data/object.matrix -l data/object.labels -pl data/parameter_svm.model -r data/object.result.txt
```

Result File for execution mode graphsimsvmwalktrough

Header:
C	gamma	F-Score All	Precision All	Recall All	F-Score Optimized	Precision Optimized	Recall Optimized	Features Optimized	Ratio Optimized	F-Score Extended	Precision Extended	Recall Extended	Features Extended	Ratio Extended
