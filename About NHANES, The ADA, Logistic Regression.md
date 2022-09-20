    Gabriel Castellanos
0MBS Analytics
Evaluating the American Diabetes Association guidelines Using Machine Learning Methods
Introduction
	The American Diabetes Association (ADA) bi-yearly publishes new guidelines on diagnostic criteria for physicians to use to diagnose patients with diabetes. More specifically, they primarily use three diagnostic criteria: HGBA1C (A1C), fasting blood glucose level (FBG), and an oral glucose tolerance test. These criteria determine who is classified as optimistic and pessimistic, who has diabetes, and who does not have diabetes. Note that there is a distinction between those diagnosed as pre-diabetic and diabetic, but for this project, I lumped this distinction into one class; those diagnosed as pre-diabetic based on laboratory markers are classified as diabetic. 
	The A1C level measures one's average blood sugar levels over a two to three-month period, and a value of 5.7% or greater indicates a person has diabetes. In reality, this would suggest that a person is pre-diabetic, but it means that a person has clinical diabetes for this project. Another measure of clinical diabetes is a patient's FBG levels, which indicate someones glucose level after not having eaten anything for at least 8 hours. A measure of less than 100 mg/dl (for this project) would indicate a person without diabetes, and greater than 100 mg/dl would tell someone who does. The last measure clinicians use to measure diabetes status is an oral glucose tolerance test which is (in lamens terms) a measure of how your body responds to glucose. In regards to this project, this last biomarker—while important for clinicians—was not used because I could not find appropriate data on this variable. Therefore, only HGB!AC levels and FBG levels were used to group people into either having diabetes or not having diabetes. I will talk more about this in the approach section. This project aims to assess how well these two criteria distinguish these two class attributes--- 'yes' and 'no' for if a person has diabetes. I originally wanted to do something similar with unhealthy eating, but all my data came from the National Health and Nutrition Examination Survey (NHANES), data regarding this topic was much easier to work with and obtain.  
Motivation:
	The motivation for this project was complex. First, the goal was to get practice working with the NHANES datasets. These datasets come from the National Center for Health Statistics (NCHS), which seeks to provide health and nutritional information for US citizens. The NHANES collects health, nutrition, and demographics data by conducting personal interviews, physical examinations, and laboratory tests to detect disease prevalence and common health conditions throughout the US. Each set of variables is located in different datasets depending on how those variables are categorized. For example, data about Body Mass Index (BMI) or height would be found in the physical examination portion of the NHANES data set, while information about laboratory biomarkers (like Vitamin D levels) would be found in the laboratory examination section. There are five sections: demographic, questionnaire, laboratory, nutrition, and physical examination data. New data for each unit is collected every year, dating back to 2002. Each team has many datasets depending on the topic of interest, and I wanted to get some practice cleaning these datasets and merging them to create a target class, with features of interest extracted from different datasets.  
	Another motivation for this project was to appreciate logistic regression better. I will discuss this more in the results/approach section, but needless to say, I desired to have a binary classification variable as my target class—in this case, it consists of 'yes' or 'no' depending on whether or not the participant has clinical diabetes. This is contingent on what their blood work said in the NHANES data. In addition to one binary dependent variable, I also wanted to play with multiple independent variables to see if I could use different statistical criteria to evaluate different sub-models and pick the best one. One other motivation would be to assess the diagnostic criteria for diabetes; are the cutoff-values fine at their current values. This criterion is constantly changed on a year-to-year basis, so I hope to use machine learning methods to evaluate whether or not the requirements posted by the American Diabetes Association are good for diagnosing patients. Given a set 
Literature Review 
The following papers were reviewed in preparation for this proposal: An Unsupervised Machine-Learning Approach to Characterize Ultra-Processed Food Consumers: Results from the NHANES 2015–"2015 Survey Cycle"  , "Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records",  "Predicting youth diabetes risk using NHANES data and machine learning," "Improving Healthcare Cost, Quality, and Access Through Artificial Intelligence and Machine  Applications,"  and "Logistic Regression: A Brief Primer."  
The first article I wish to discuss is Logistic Regression: A Brief Primer.  The authors discuss the pros and cons of different generalized linear models and how appropriate other models. Medical outcomes are typically suitable for logistic regression as a binary outcome generally is measured; mortality, for example. What is being measured is not a continuous outcome, as such is the case with linear regression, and logistic regression is much more flexible with what types of independent variables can be inputted into the model. The authors also point out the assumptions that must be met for logistic regression as well as other generalized linear models, such as multicollinearity of variables (being avoided), normality of variables, and linearity between continuous variables and the logit of the response. Different model-building approaches were also covered in this paper. 
I want to look at the following article: "Predicting youth diabetes risk using NHANES data and machine learning."   This article uses NHANES datasets to create a dataset on youth diagnosed with diabetes. Then, they used machine learning methods to validate and assess these different models, using sensitivity, recall, and F measures. Their goal was to determine how well the ADA criteria predicted diabetes for youth, given certain features, and I used these same features and cutoff values for my project. While they focused on the youth population, I chose to focus on NHANES data, excluding youth—anyone older than 16 years old I did not use in my dataset. This paper also gave me the motivation to try and build my dataset from individual NHANES datasets.  
The last article we wish to discuss is "Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records."  The authors looked at patient data collected over a decade from 1999-2008 and fitted a logistic regression model to predict the probability of hospital readmission given numerous attributes. Each patient in the dataset was admitted to the hospital and had at least one pre-existing diagnosis of diabetes, with common attributes such as age, weight, other pre-existing medical conditions, and more. The authors were able to find out that HbA1c measurement, along with any co-morbidities, was the essential factor in determining if someone was going to be re-admitted into the hospital within the next 30 days (from their initial admission into the hospital). They built multiple models with different variables to see which variables had the most effective and to what extent. Ultimately, they concluded that clinicians might have prevented readmission if more HbA1c measurements had been made.    

Approach:
The approach was to try and build my own 'custom' dataset using the NHANES datasets online for the year of 2017-2018. All analysis and tasks were performed using R, and the individual datasets were imported using the NHANES library (the package's exact name is NHANESA). I believe Python has a similar module, but I found this process much more intuitive with the R version of it. I took data sets from the demographics, questionnaire, physical examination, and laboratory sections for the 2017-2018 year. I wanted to use other years, but merging datasets from different years proved to be beyond my means. Combining datasets within the same year is feasible because every participant from that year has a corresponding SEQN primary key number, which is a number tied to every response for each unique participant in that year's survey. I chose my feature variables from my literature review and variables that one would intuitively think to play a role in diabetes incidence. I also included variables that are not directly related to diabetes prevalence, such as vitamin D levels. 
My main algorithm for this project was a logistic regression model that typically has a binary dependent variable; my binary dependent variable is a clinical diabetes diagnosis, a factor variable with two levels: yes or no. Creating this target variable took quite a bit of data wrangling, as I used both fasting blood glucose levels and hemoglobin A1C levels to classify participants as either having a positive clinical diagnosis of diabetes or not having a clinical diagnosis of diabetes. I used the cutoff values provided by the ADA for the diagnosis of diabetes; anybody with an HGBA1C level of equal to 5.7% or greater had a yes under the dx (diagnosis) of diabetes feature or a no for dx of diabetes if their measurement was less the 5.7%. Alternatively, if the participant had a fasting blood glucose of greater than or equal to 100 mg/dl, they would have a label of yes for dx of diabetes and a designation of no if it is lower than 100 mg/dl. Unfortunately, not every participant in the survey for the selected year (2017 – 2018). While some study participants had one or both of these measures recorded, a large number of participants did not report this measure. Therefore, participants with at least one of these measures— blood glucose or HGB1C—were merged into a single target variable of diabetes. 
For feature variables, I started with wrangling the race/ethnicity variable. This variable came from the demographics dataset. Instead of including all the surveyed races that were present in the survey— Mexican American, Other Hispanic, Non-Hispanic White, Non-Hispanic Black, Non-Hispanic Asian, and, Other Race - Including Multi-Racial— I chose instead to look only at Non-Hispanic White and those that were not Non-Hispanic White. That is any participant who was not Non-Hispanic White was lumped into a race labeled Other. People who are white are typically at the highest risk for diabetes incidence, which is why this wrangling was done for this variable. Once completed, the race variable was a factor variable with two levels: Non-Hispanic White and Other. 
Another independent variable used for analysis was age. While this is, in theory, a continuous variable—we continue to age at any given time interval—age is typically reported as a discrete variable, which is how it was used in the analysis. Note that some participants had a recorded age of 0, which is technically impossible. Thus, anyone whose age was 0 was converted to an NA value, which is how NA values for age were recorded in the survey. Gender was also used in my analysis, but not much is noteworthy about this variable. It is considered a factor/categorical variable with two levels: Male and Female. A variable from the questionnaire dataset was taken to establish some indication of family history, which entails whether the respondent has a family history of high blood pressure. The is also a categorical variable/factor variable with two levels: yes or no. Those who had a record of 9 under this attribute did not answer this question and thus were recoded to NA. 
The model was also used to measure body mass index (BMI), blood pressure (BP, Systolic), Total cholesterol, and Vitamin D levels. A box plot of each of these individual continuous variables revealed that each one of these variables has clear outliers. However, instead then deleting these outliers—which would result in less overall data for the model— I used min/max scaling to mitigate the effect outliers have on the logistic regression model I built. So, again, the more instances I have for the model, the better. Moreover, age was also converted using min/max scaling. Finally, z score standardization was not used because many of these variables were not normally distributed, as revealed by their respective histogram plots.   	
Noticeably, many patient instances did not have the target variable recorded—dx of diabetes. Accordingly, instances without these measures received a value of NA. While we could (in theory) remove those instances with NA for the target variable, that would reduce our total instance count from ~ 11,000 to ~ 2,000. Not ideal as (generally speaking) the more instances you have to work with, the better. Instead of taking this approach, I used the Multivariate Imputation by Chained Equations (MICE )package in R to impute missing values for the target variable and any other variables that had missing values. The method is based on Fully Conditional Specification, where a separate model imputes each incomplete variable.  The MICE algorithm can attribute mixes of continuous, binary, unordered categorical and ordered categorical data. The parameters that go into this MICE function that determines how the missing values are imputed are pretty complex; the parameter method accepts a vector of strings where each string is used to assign missing values. Different parameters can be cross-validated with any given fitness metric; in theory, a grid search of all possible parameter permutations could be used to search for the permutation that yields the highest accuracy or any other fitness metric. However, converging on a single answer was difficult, given my limited computing resources. If I had the money, I would definitely try and use cloud resources (as opposed to local), but I was able to find a permutation of parameters that yielded an accuracy of >= 90%. Accuracy was used as the classes of the target variable (yes and no) were relatively balanced. Given that the target class variable was not balanced, another fitness metric (such as AUC ROC) would be used. 
A sample of the first three instances is shown below, along with a scatterplot matrix of the independent numeric variables used in the model:
 
 


	Subsequent exploratory data analysis was done to study our dataset. The following plots show how the four continuous vary by diabetes state (either yes or no):

    
Clearly, there are too many outliers to consider removing those instances from the data set. Thus min/max scaling was used to minimize the effect of statistical outliers. Vitamin D levels also do not appear to vary based on status, at least when it comes to the spread of the data. We can also look at how diabetes status varies based on race using a dual-bar plot. The overall instances with the race ‘Other.’ 
 
Note that the blue bar represents the race of Other while the orange one shows how diabetes counts for the white race. 
We can also see how the target variable differs by sex, as shown in the following graph, which shows that males are more likely to be diagnosed with diabetes than females. Note that the dataset's balance between females and males is more or less equal. 
 
Note that instead of using age as a discrete variable, in addition to min/max scaling, I used minimum description length (MDL) binning based on the target variable using the discretization package in R. Discretization was done because the age variable, in particular, seemed to be excessively skewed. (as shown in the scatterplot matrix). MDL binning of the age variable also yielded a greater accuracy than not using it. MDL was also used on other numeric variables, but these proved to show little improvement in model fit and accuracy.. Moreover, using MICE enhanced the available data to model dramatically; if we drop incomplete instances with NA values, we have 2653 obs. of  10 variables, while with MICE, we have 15998 obs. of  9 variables. Finally, note that at one point, I experimented with using BMI as a factor variable based on percentile rank, but this proved to yield any fruitful results. 


Evaluation Metrics/Results  
	Once the dataset was cleaned, variables were wrangled, and data was augmented, I was ready to assess how well a logistic regression model would fit the model. The three basic assumptions of logistic regression are: The relationship between continuous independent variables and logit function is linear; Lack of multicollinearity; Independence of residuals; Absence of powerfully influential outliers. For starters, a graphical representation that depicts the relationship between continuous variables and the logit of the response variable (diabetes_status) is shown below:
 
As we can see, these smooth curves show a linear relationship between some variables, while others—specifically total cholesterol and vitamin D levels show a flat level, constant response. In theory, this would be reflected in the p values when the model is discussed later during analysis. The next plot depicts the residual deviance for each observation which is based on maximum likelihood estimation.: 
A handful of observations (~60) are => 3 standardized residuals. Those observations were removed. We can also look at Cook's distance to quantify observations that have a large degree of influence:  
The top 3 observations with the highest degree of influence are shown based on Cooks distance. Of course, Cook distance is not the only metric that can tell which points have a significant degree of leverage, but this is enough evidence to remove these observation numbers shown in the graph. Lastly, we have to look at multicollinearity between features to determine the aptitude for logistic regression. Our chart on the previous page five shows quantifies this metric, with 1 or -1 being a clear sign of this violation. Clearly, no variable fits these criteria, so we can say that the numeric variables are not correlated with each other. 
	Now that the basic assumption of logistic regression has been met, we can look at statistics showing how well the model fits the data. The following statistical table shows how well the GLM fits the data:

Call:
glm(formula = diabetes_status ~ ., family = binomial, data = fullmodel)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-4.3744  -0.1952  -0.0001   0.3766   3.1246  

Coefficients:
                         Estimate Std. Error z value Pr(>|z|)    
(Intercept)              -2.16171  104.77881  -0.021    0.984    
Age.L                    42.16174  400.98585   0.105    0.916    
Age.Q                     7.70394  553.42335   0.014    0.989    
Age.C                   -22.66657  396.61894  -0.057    0.954    
Age^4                    -2.85580  552.71873  -0.005    0.996    
Age^5                     3.58907  406.06189   0.009    0.993    
Age^6                     3.92484  576.39856   0.007    0.995    
Age^7                   -10.59264  385.44533  -0.027    0.978    
Age^8                    -7.14407  609.11554  -0.012    0.991    
Age^9                     1.20932  448.60344   0.003    0.998    
Age^10                   -1.88506  611.01972  -0.003    0.998    
Age^11                   -3.44309  476.57729  -0.007    0.994    
Age^12                  -10.88769  601.92045  -0.018    0.986    
Age^13                    1.43725  439.12644   0.003    0.997    
Age^14                   -6.11698  576.64822  -0.011    0.992    
Age^15                   -1.11275  510.90713  -0.002    0.998    
Age^16                  -11.14418  628.06848  -0.018    0.986    
Age^17                    0.86692  514.90310   0.002    0.999    
Age^18                   -9.19225  574.80100  -0.016    0.987    
Age^19                   -0.18149  564.22931   0.000    1.000    
Age^20                  -12.89398  603.54149  -0.021    0.983    
Age^21                    0.59949  584.13253   0.001    0.999    
Age^22                  -12.73163  559.09042  -0.023    0.982    
Age^23                   -2.12864  608.46872  -0.003    0.997    
Age^24                  -16.10520  540.49410  -0.030    0.976    
Age^25                  -15.82058  915.31828  -0.017    0.986    
Age^26                  -27.31635  533.67700  -0.051    0.959    
Age^27                    7.64229 1011.76493   0.008    0.994    
Age^28                   -1.56899  460.60234  -0.003    0.997    
Age^29                   -7.39693  554.21015  -0.013    0.989    
GenderFemale             -0.92208    0.06197 -14.879  < 2e-16 ***
RaceNon-Hispanic White    0.06598    0.06843   0.964    0.335    
BMI                      11.11518    0.41958  26.491  < 2e-16 ***
BP_mmHG                   0.57605    0.36534   1.577    0.115    
Total_Cholesterol        -0.16791    0.33375  -0.503    0.615    
Vit_D_nmol_L             -2.17988    0.51494  -4.233  2.3e-05 ***
Dx_of_high_BPNo           0.03226    0.08147   0.396    0.692    
---
Signif. codes:  0 â€˜***â€™ 0.001 â€˜**â€™ 0.01 â€˜*â€™ 0.05 â€˜.â€™ 0.1 â€˜ â€™ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 21797.5  on 15997  degrees of freedom
Residual deviance:  7319.6  on 15961  degrees of freedom


As we can see, the use of MDL binning for the age variable vastly increases the complexity of our statistical output. Each age group is treated as a separate categorical variable instead of one group for the entire variable. The nominal p values associated with each age group are expected, they were grouped in such a way that minimizes entropy concerning the target variable.
The deviance measures are calculated using maximum liklehood. Null deviance can be thought of as how well the model predicts the target variable with only the slope parameter, while residual deviance can be considered as how well the model predicts the outcome (diabetes status) with the given variables/parameters. Subtracting these two values not only tells yields a chi-squared statistic and can be used to assess how well the model fits the data based on the associated p-value, but the difference also tells us that the fit is significantly better than simply using the slope parameter to estimate the model. We could also use Mcfadden pseudo R squared to measure fit, but this is not as universally accepted as a viable metric as Pearson R squared is.
In contrast, let's take a look at the models without MDL binning for the age variable and simply min/max scaling applied to all numeric variables.:
Call:
glm(formula = diabetes_status ~ ., family = binomial, data = impdata1)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-5.2586  -0.3651  -0.1367   0.3467   3.5714  

Coefficients:
                         Estimate Std. Error z value Pr(>|z|)    
(Intercept)            -8.4243614  0.3731305 -22.578  < 2e-16 ***
Age                     0.0918601  0.0024152  38.034  < 2e-16 ***
GenderFemale           -1.1135123  0.0588817 -18.911  < 2e-16 ***
RaceNon-Hispanic White -0.2213063  0.0658622  -3.360 0.000779 ***
BMI                     0.2203877  0.0060740  36.284  < 2e-16 ***
BP_mmHG                 0.0019108  0.0027082   0.706 0.480445    
Total_Cholesterol      -0.0009784  0.0010037  -0.975 0.329657    
Vit_D_nmol_L           -0.0102784  0.0014023  -7.330 2.31e-13 ***
Dx_of_high_BPNo         0.2964856  0.0841875   3.522 0.000429 ***
---
Signif. codes:  0 â€˜***â€™ 0.001 â€˜**â€™ 0.01 â€˜*â€™ 0.05 â€˜.â€™ 0.1 â€˜ â€™ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 21786.3  on 15997  degrees of freedom
Residual deviance:  8668.3  on 15989  degrees of freedom
AIC: 8686.3

Number of Fisher Scoring iterations: 6

In contrast, we can see that the difference between the two deviances is less, indicating an inferior fit, but the importance of the individual variables is much easier to interpret. A p-value of <=.05 suggests that the inclusion of the variable is warranted, while a p-value  > .05 would mean the model would be better if we simply removed the variable outright. The beta coefficients can be used to calculate the odds ratio for each variable, which can be used to quantify how much more likely a person is to have diabetes—given the inputs shown in the model.
Lastly, let's look at how the model fits the data without MICE. Note that the total number of instances is significantly less, as, without MICE, we have to drop instances with missing values.:

Call:
glm(formula = diabetes_status ~ ., family = binomial, data = analytic.data2)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-2.3300  -1.0103   0.5750   0.8735   2.0656  

Coefficients:
                         Estimate Std. Error z value Pr(>|z|)    
(Intercept)            -2.777e+00  4.818e-01  -5.763 8.25e-09 ***
Age                     3.811e-02  3.346e-03  11.389  < 2e-16 ***
GenderFemale           -6.272e-01  9.797e-02  -6.402 1.53e-10 ***
RaceNon-Hispanic White  2.720e-02  1.079e-01   0.252    0.801    
BMI                     6.392e-02  7.425e-03   8.609  < 2e-16 ***
BP_mmHG                 1.882e-03  3.129e-03   0.602    0.547    
Total_Cholesterol       9.914e-05  1.174e-03   0.084    0.933    
Vit_D_nmol_L           -2.140e-03  1.744e-03  -1.227    0.220    
Dx_of_high_BPNo        -1.900e-01  1.201e-01  -1.582    0.114    
---
Signif. codes:  0 â€˜***â€™ 0.001 â€˜**â€™ 0.01 â€˜*â€™ 0.05 â€˜.â€™ 0.1 â€˜ â€™ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 3009.0  on 2273  degrees of freedom
Residual deviance: 2584.3  on 2265  degrees of freedom
  (13724 observations deleted due to missingness)
AIC: 2602.3

Number of Fisher Scoring iterations: 4

Note the difference in B coefficients and p values and the drastic drop in the difference between the residual and null deviances. In summary, min-max scaling without discretization of the age variable provides a much more interpretable model with a slightly inferior fit.
	With an appropriate model worked out, the last step is to validate such a model using the desired fitness metric. In this case, accuracy— is defined as ((TN + TP)/(FP + FN +TN +TP)) – I believe it is appropriate, given that the target class is balanced in terms of class instances. Specifically, the ratio of the cases with label no to those with the label yes is 9230:6768. While this ratio is not perfectly balanced, it is not egregious enough to consider other fitness metrics such as precision, recall, or even ROC AUC. Finally, we will produce a ROC curve to see how the model performs at different thresholds. 
	In addition to the logistic model, we will use an ensemble machine learning method (random forest) and compare the reported accuracy using training and test data sets. Furthermore, in addition to corresponding with random forest, we will compare the GLM model with MDL binning to the GLM model without MDL binning and without MICE imputation to see which model attains the highest. Note that both GLMs and RF machine learning methods have associated hyperparameters that affect how well each method performs; the regularization rate for GLMs and n_estimaters for the random forest method. Each unique set of parameters is subsequently used during k-fold cross validation where k = 3 folds of data. A split ratio of .75 was used to split the data between training and testing. Note that parameter tuning was done using the caret package in R, which only allows for tuning the mtry parameter. A random search for the optimal value of the mtry parameter was used. Mtry is the number of variables that are sampled at each split of a tree. The desired fitness metric used was accuracy. The results are displayed below, showing the algorithm's performance at each value. 
 
The graph shows that the model performs best when mtry = 9—just over 90%. Not bad. Next, we will compare the accuracy metric with that of the model used for logistic regression. 
The following shows the confusion matrix for our random forest model:
Cross-Validated (3 fold) Confusion Matrix 

(entries are percentual average cell counts across resamples)
 
          Reference
Prediction   No  Yes
       No  51.7  3.8
       Yes  6.0 38.5
                            
 Accuracy (average) : 0.9024
Here is how the model performed using MICE with logistic regression:
Cross-Validated (3 fold) Confusion Matrix 

(entries are percentual average cell counts across resamples)
 
          Reference
Prediction   No  Yes
       No  51.9  4.1
       Yes  5.8 38.2
                            
 Accuracy (average) : 0.9016
And this is how our model performed without age discretization, MICE, with logistic regression :
Cross-Validated (3 fold) Confusion Matrix 

(entries are percentual average cell counts across resamples)
 
          Reference
Prediction   No  Yes
       No  52.8  3.9
       Yes  5.1 38.2
                            
 Accuracy (average) : 0.9103

And this is how the model performed without MICE, min/max scaling—using logistic regression:

Cross-Validated (3 fold) Confusion Matrix 

(entries are percentual average cell counts across resamples)
 
          Reference
Prediction   No  Yes
       No  18.7 10.2
       Yes 19.4 51.7
                            
 Accuracy (average) : 0.7039



It seems like the model without age discretization, logistic regression, and min/max scaling for the age variable attained the highest accuracy of 91%. Not bad.
The following shows the ROC AUC as well as the precision recall curves for the model using logistic regression and age discretization:
 
 

Given that there is a slight imbalance in the proportion of class instances, it is essential to look at the PR curve to see how the model's performance changes at different thresholds. In addition, the ROC AUC of .97 indicates that there is quite a separability between the two classes. 
Discussion/Future Endeavors 
	In conclusion, we obtained a model accuracy of 91% using a combination of min/max scaling, MICE imputation of missing values, and residual analysis. The GLM model outperformed the random forest model in training and testing by 1% point, which is a lot. Moreover, the implementation of these methods dramatically increased the performance of our model 
	It is essential to mention that not only min/max scaling was applied in this case, but also z-score normalization and as well as binning of all numerical variables. Min/max scaling appeared to yield the model with the best accuracy with no binning/discretization applied, but any other number of transformations might have resulted in higher accuracy; log transformations and squared transformations were examined but with limited success. In addition, the chosen variables were based on a literature review, but any number of variables could have yielded a better model; we used a diagnosis of high blood pressure and gender in the model, but it was clear that those variables had a minimal impact. There are many variables to choose from during each year, but the goal of this project was (in part) to look at the ADA guidelines for diabetes diagnosis and build a model around those variables. In the future, I desire to look at items taken in the survey that are outside the guidelines but are still valuable for determining a diagnosis. The inclusion of vitamin D levels in the model was to, in part, achieve this goal. While in this instance, we filled imputed missing data to augment the dataset, hopefully (hopefully in the future) we can combine data taken from multiple years (not just one year). In addition, I would hope to gain a better understanding of the MICE package in the future, which was used to impute missing values; there are numerous algorithms you can use to assign missing values, and trying to find the best one, be tricky. Initially, the KNN algorithm was used to impute missing values, but the MICE package in R offered a lot more flexibility, and the MICE proved superior to the KNN algorithm in terms of accuracy.  
	While the logistic regression model outperformed the random forest model during training, remember that this was done using the caret package in R. With the caret package, due to the way the algorithm was written (done by Max Kuhn), only one hyperparameter is modifiable during testing. For example, the random forest done using scikit-learn does not have this limitation as there is an enormous amount of hyperparameters to choose from. However, despite only being able to choose one hyperparameter, the training of the random forest model still took a while to converge, even with doing a random grid search. In the future, it might be more feasible to use cloud computing resources that scale, so training does not take so long. Moreover, we could use cloud resources to automate the training process. That way, we can pass in different parameters into the model—not just hyperparameters but additional data transformations retrain the model and see how our fitness metric changes.    
	In closing, we were able to augment, fit and build a dataset from the NHANES dataset from the year 2017-2018 that had an excessive amount of missing values. Data augmentation and data transformations were applied to try and create a logistic regression model that could predict the probability of diabetes, given risk factors similar to those used by the ADA. These methods resulted in a vastly superior model in terms of how well the model fit the data and in training and testing. The GLM also proved to be better than one fit using the random forest method, as the GLM had a higher accuracy during training and testing.   
	


 								
								
Company Pitch	Integrity 	Social Benefit	IP Balance	IP Portfolio			
Sunset(AP1) 	20; Nothing outwardly egregious, average score	50; the social benefit this provides is important, can potentially prevent cancer	15; average score as ip portfolio is empty at this point	15, Great licensing, product but nothing beyond that
Avec beaute(AP2)	15; Nothing otwardly wrong, average score	15 low score due to the fact that this is a luxury product	20; below average due to the fact that at this point I hav already invested in the personal care industry	3ips I counted for Glow$go system-- which is a lot. The more Ips the higher the score
CustoMed solutions(AP 3)	25; average score, anything related to medicine always has high integrity	25; high social benefit, but needs to be explained to outside investors	35 Yet to include biotech company	15 seems like a licensing product, but regularoy issues make it difficult to see IP potential beyond that 
Ionic Ear (Shark Tank 1)	10; does not seem to serve moral purpose	10; almost none to speak of, luxury item	70; yet to include a tech company	10; very little potential Ips from what is presented. Also a whole host of regulatory issues 
3d food printing( (Dragons den 2	15; No unsrupulous buisness pactices	25; always need for healthier foods	35; have not invested in food industry	25; Endless different Ips for different foods
I want to draw a cat for you (Shark Tank 3)	0; seves no real higher purpose	0; no real benefit to society 	100; very niche product	0; virtually nothing patentable-- it is a drawing
								










 
