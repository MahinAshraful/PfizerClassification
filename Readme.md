This is the classification models

Each folder is named for the configurations

eg:

n_estimators=100 (number of trees)
max_depth=10 (tree depth limit)
min_samples_split=2 (default - not explicitly set)
min_samples_leaf=1 (default - not explicitly set)
random_state=42 (random seed)
class_weight='balanced' (class balancing)

this will be in folder "est100_d10_3cv_s2_l1_rs42"

## HOW TO RUN
1-> pip install -r requirements.txt (pip3 install -r requirements.txt for Linux)
2 -> navigate through directory & once in run: python main.py

### References

If you used the dataset from the following paper, please cite it as:

Little MA, McSharry PE, Roberts SJ, Costello DAE, Moroz IM.  
"Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection."  
*BioMedical Engineering OnLine*, 2007, 6:23 (26 June 2007).  
DOI: [10.1186/1475-925X-6-23](https://doi.org/10.1186/1475-925X-6-23)