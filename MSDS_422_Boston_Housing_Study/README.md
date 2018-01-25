## Boston Housing Study

This project involves fitting regression models with Python.
The regression models are to be fit to data from the Boston Housing Study <boston.csv>.

### Background Materials
All explanaroty variables are used, expect <em>neighborhood</em>, along with 506 census tract observations
from the Boston Housing Study.

* The log median values of homes in thousands of 1970 dollars.
### The following regression methods were used for comparision. 
* linear regression
* ridge regression
* lasso regression
* elastic net 
* Evaluate these methods within a cross-validation design, using root mean-squared error (RMSE) as an index of prediction error. 
* Python Scikit Learn should be your primary environment for conducting this research. 

 

### Management Questions to Answer
We are advising a real estate brokerage firm in its attempt to employ machine learning methods.
The firm wants to use machine learning to complement 
conventional methods for assessing the market value of residential real estate. 
* Of the modeling methods
 examined in your study, which would you recommend to management, and why?

### Results and Recommendations
The results from the 10-fold cross-validation in standardized units informs us that both the 
Linear and Ridge regression models performed the best with 0.50 and 0.50, respectively, under 
the root mean-squared error metric. Elastic Net regression model coming in next with a reporting 
0.52 root mean-squared error metric and Lasso regression model taking last place with a root 
mean-squared error of 0.54. Therefore, I recommend from the regression models presented the 
Linear regression model using the log mean home market valuation for further valuation 
endeavors until other methods can be explored and compared.<br><br> 
The final analysis and recommendations can be found here: [Final_Analysis_Management_Recommendation.pdf](Final_Analysis_Management_Recommendation.pdf)

#### Python Programming
* Python packages: matplotlib, numpy, os, pandas, and seaborn  
[Jupyter Notebook](MSDS_422_Boston_Housing_Study.ipynb)<br>
[Python Code](MSDS_422_Boston_Housing_Study.py)<br><br>

#### prepare for Python version 3x features and functions, thus for Python 2x put the following in the preamble
* from __future__ import division, print_function
* from future_builtins import ascii, filter, hex, map, oct, zip
<br>
<a href= "https://obrianbl.github.io/NW_MSDS_422_Practical_Machine_Learning/">NW_MSDS_422_Practical_Machine_Learning</a>
<br>
<a href="https://obrianbl.github.io/">Return to Main Page</a>


