--- OVERVIEW ---
This project contains the datasets submitted during Phase I and all the code
from preprocessing the data, to model tuning, and reproducing performance
metrics, tables, and confusion matrices produced in the report.

Note: Execution may take 25+ minutes as the SVM model in particular takes a
very long time to run with the large dataset.

I tried running my code on my Macbook, and I get some different output and charts
than on my Windows desktop. I get the exact same results on my Windows
Laptop as I get on my Windows desktop however, so this appears to be
platform specific for some reason.

--- REQUIREMENTS ---
The code was developed on Windows using Python 3.13 with the following libraries and versions:
contourpy==1.3.3
cycler==0.12.1
fonttools==4.62.1
joblib==1.5.3
kiwisolver==1.5.0
matplotlib==3.10.9
numpy==2.4.4
packaging==26.2
pandas==3.0.2
pillow==12.2.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
scikit-learn==1.8.0
scipy==1.17.1
seaborn==0.13.2
six==1.17.0
threadpoolctl==3.6.0
tzdata==2026.2


--- DATA ---
The '../data/' directory should contain:
- animal_test.csv
- animal_train.csv
- baseball_test.csv
- baseball_train.csv

These are the datasets exactly as provided in Phase I.

--- EXECUTION ---
All data processing, model training, and figure generation is automated
via main.py. To reproduce the results:

1. To install required packages:
     pip install -r requirements.txt

2. Run main.py

main.py first executes 'baseball.py' and then 'animal.py',
saving all resulting plots to the '../plots/' directory as specified
in the source code. The first line in both baseball.py and animal.py
loads the data, and within that load function, the preprocessing is done with
a single function (contained in utils.py). Each model is given its own .py file
where training and tuning occur, outputting a final optimal model determined by
GridSearchCV and used in baseball.py/animal.py for model evaluation. Learning
curves produced in the model .py files as well. Baseball.py and animal.py produce
the confusion matrices, and tables with training, validation, and test results.

--- FILE DESCRIPTIONS ---
- main.py: Central hub for running both baseball.py and animal.py.
- baseball.py: Handles training and evaluation for the baseball dataset.
- animal.py: Handles training and evaluation for the animal dataset.
- utils.py: Holds util functions used across both datasets
========================================================================