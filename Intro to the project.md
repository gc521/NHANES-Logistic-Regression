    The American Diabetes Association (ADA) bi-yearly publishes new guidelines on diagnostic 
criteria for physicians to use to diagnose patients with diabetes. More specifically, they primarily use three 
diagnostic criteria: HGBA1C (A1C), fasting blood glucose level (FBG), and an oral glucose tolerance test. These criteria determine who 
is classified as optimistic and pessimistic, who has diabetes, and who does not have diabetes. 
Note that there is a distinction between those diagnosed as pre-diabetic and diabetic, but for this project, 
I lumped this distinction into one class; those diagnosed as pre-diabetic based on laboratory markers are classified as diabetic.
Data from the NHANES datasets were used from the 2017-2018 year that include such independent variables such as blood pressure, BMI, Total Cholesterol, Gender, Race, and a history of being diagnosed with high blood pressure. Other independent variables were also used in addition to the target variable which is clincal diabetes status as already mentioned. Datasetes were queried using NHANES A package in R. All plots and visualizations were done using plotly in R.
   Many missing values are a problem when dealing with these datasets, so rather then removing instances with NA values, Multivariate Imputation by Chained Equations (MICE) R package was used with shown parameters in code. Ideally, a grid search can be done to find the most optimal set of parameters through cross validation given a chosen fitness metric (we will use accuracy). 

  After the variables were wrangled appropriately, exploratory data analysis was preformed in oreder to determine how apropriate logisitc regression would be in determing a predictive model for diabetes status. In addition to visualizing how the target variable differes by independent variable, the deviance residuals were also analyzed, in addition to influential outlieres based on cooks distance. Continuouse variables were relationship to the logit resopnse curve was also visualized. A Logistic Regression model was then fitted to the data. The area infromation criterion (AIC) is indicative of a good model fit (based on likelehood). 
  
  Model was then compared to random forest algorithim to see how well the logistic regression model. Both models achieved a 90% accuracy when split into training and testing sets and cross validitated using k = 3 folds. Random forest was slightly superior with an accuracy of 91%. Random forest was computed using the caret package, which only allows for the tuning of the the mtry parameter. A graph of the accuracy at different values of theis parameter are shown using a random grid search. 
  
   As the target class is slightly imbalanced (9230No:6768Yes),  a precision-recall curve and ROC curve is warrented. The AUC of .97 shows a good seperability between the two classes.    


  
