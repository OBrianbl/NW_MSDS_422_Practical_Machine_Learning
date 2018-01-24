## Evaluating Classification Models

For this assignment you are asked to fit classification models to data from the Bank Marketing Study.

### Background Materials

* Use three binary explanatory variables relating to client banking history: default, housing, and loan. 
* Predict the binary response variable: 
* Has the client subscribed to a term deposit? 
* Data is presented as csv <bank.csv> 
### Employ two classification methods: 
* logistic regression 
* naïve Bayes classification
Methods were evaluated with a 10 fold cross-validation design, using the area under the receiver operating characteristic (ROC) curve as an index of classification performance.<br> 
Python Scikit Learn is the primary environment for conducting this research.

## Management Questions to Answer

Advising the bank about machine learning methods to guide telephone marketing campaigns.
* Which of the two modeling methods would you recommend, and why?
* which group of banking clients appears to be the best target for direct marketing efforts 
(similar to those used with previous telephone campaigns)? 


## Results and Recommendations
Examining the average AUROC for the logistic regression classification method, 0.6079, and the 
naives Bayes classification method, 0.6081, we see that the naives Bayes model performs 
slightly better for predicting customers that will participate in term deposits when using three 
explanatory variables, default, loan, and housing. Those most likely to participate also have no 
defaults. Thus, I recommend the naives Bayes method to direct telephone direct marketing 
campaigns towards those with no defaults. Also, I recommend further models to be explored 
with possible more explanatory variables being introduced to help better explain an individual’s 
response.<br><br> 
The final analysis and recommendations can be found here: [Final_Analysis_Management_Recommendation.pdf](Final_Analysis_Management_Recommendation.pdf)

### Python Programming
**Python packages: matplotlib, numpy, os, pandas, and seaborn  
[Jupyter Notebook](Bank_Markerting_Campaign_Classifier_Comparision.ipynb)<br>
[Python Code](Bank_Markerting_Campaign_Classifier_Comparision.py)

### Prepare for Python version 3x features and functions
** These two lines of code are needed for Python 2.7 only
* from __future__ import division, print_function
* from future_builtins import ascii, filter, hex, map, oct, zip




