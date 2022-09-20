library(psych)
library(usdm)
library(nhanesA)
library(labelled)
library(car)
library(plyr)
library(dplyr)
library(effects)
library(trafo)
library(ggplot2)
library(DescTools)
library(lmtest)
library(caret)
library(ROCR)
library(DMwR2)
library(patc)
library(ggeffects) 
library(flexmix)
library(LambertW)
library(questionr)
library(qwraps2)
library(VIM)
library(moments)
library(graphics)
library(DMwR)
library(mice)
library(bruceR)
library(kableExtra)
#info on different datasets from NHNAES website
demo=nhanes('DEMO_J')#Store desired dataset into object(demographic for 2017-2018)
names(demo)## Variable names ## Begin subsetting the data
#based on lit. review/variable which are important
demo1=demo[c('SEQN', #Respondent sequence number, used to merge datasets
             'RIDRETH1', #Race/Hispanic origin
             'RIDAGEYR',#Age at years of screening
             'RIAGENDR'#Gender of participant
             )]
demovars=names(demo1)
demo2=nhanesTranslate('DEMO_J', demovars, data = demo1)
head(demo2)##subsetting of demographic dataset for 2017-2018 yr is done
#Next is BMI variable, which comes from lit review
bmx=nhanes('BMX_J') #next dataset includes body measures 
#which we want to extract variable BMI from
names(bmx)#All variables in this dataset 
bmx1=bmx[c('SEQN',
           'BMXBMI' )]
bmxvars=names(bmx1)

bmx2=nhanesTranslate('BMX_J', bmxvars, data = bmx1)
head(bmx2)#Need to remove NA value instnaces for the BMI variable,
#cannot preform analysis if this variable is missing
##Next dataset includes total cholesterol value, found in 
#lab data, total cholesterol 

tchol=nhanes('TCHOL_J')
names(tchol)
tchol1=tchol[c('SEQN',
               'LBXTC')]     
tcholvars=names(tchol1)
tchol2=nhanesTranslate('TCHOL_J', tcholvars, data = tchol1)
demo3=nhanesTranslate('DEMO_I', demovars, data = demo3)




Glyco2017=nhanes('GHB_I') #HGB1C for 2015-2018
Glyco2017=Glyco2017[c('SEQN',
             'LBXGH')]
bpq17=nhanes('bpq_J')
bpq17=bpq17[c('SEQN',
              'BPQ020'
              )] # ever told you had high blood pressure (Y/N)

bpx17=nhanes('BPX_J')
bpx17=bpx17[c('SEQN', 'BPXSY1')] # Blood pressure reading, systolic measure of 1st read

glu2017=nhanes('GLU_J')
glu2017=glu2017[c('SEQN',
                  'LBXGLU')] #fastin glucose mg/dl 2017

vit_d=nhanes('VID_J')
vit_d=vit_d[c('SEQN',
              'LBXVIDMS')] # Vit_D levels for respondents 2017

analytic.data1=join_all(list(demo2, bmx2, tchol2,
                            Glyco2017, glu2017, bpx17, bpq17, vit_d), by = 'SEQN', type = 'full') 

##Full join on variables of interest from all datasets by SEQN, a unique identifier for each participant in the survey. 

names(analytic.data1)
names(analytic.data1) <- c('Patient_ID#', 'Race', 'Age', 'Gender', 
                      'BMI', 'Total_Cholesterol', 
                       'HGBA1C',
                       'Fastomg_Blood_Glucose', 'BP_mmHG',
                       'Dx_of_high_BP',
                      'Vit_D_nmol_L')## Change all variable names



#Cut HGBA1C according to diagnosisng criteria for diabetes promulgated by the ADA 

analytic.data1$HGBA1C.cut=cut(analytic.data1$HGBA1C, breaks = c(-Inf, 5.7, +Inf), right=FALSE)
#These cuts indicate our target class of diagnosed DM-- anybody >= 5.7% HGBA1C has a diagnosis (dx) of diabetes 

 table(analytic.data1$HGBA1C.cut)
analytic.data1$Fastomg_Blood_Glucose.cut=cut(analytic.data1$Fastomg_Blood_Glucose, c(-Inf, 100, +Inf), right = FALSE) 
#These cuts of FBG are also used to diagnose diabetes as per the ADA (in additon to HGBA1C)
table(analytic.data1$Fastomg_Blood_Glucose.cut)
#We can 'add' these two target vectors together to get a target class(FBG.cut and HGB1C.cut)

analytic.data1$Fastomg_Blood_Glucose.cut=Recode(analytic.data1$Fastomg_Blood_Glucose.cut,"c('[100, Inf)') = 'Yes'")
analytic.data1$Fastomg_Blood_Glucose.cut=Recode(analytic.data1$Fastomg_Blood_Glucose.cut,"c('[-Inf,100)') = 'No'")
#Code data so that anybody who meets the ADA diagnosis has an indication of yes
#for diabetes status

analytic.data1$HGBA1C.cut=Recode(analytic.data1$HGBA1C.cut,"c('[-Inf,5.7)')= 'No'")
analytic.data1$HGBA1C.cut=Recode(analytic.data1$HGBA1C.cut,"c('[5.7, Inf)')= 'Yes'")
analytic.data1$diabetes_status=paste(analytic.data1$Fastomg_Blood_Glucose.cut, analytic.data1$HGBA1C.cut) #This code allows us to wrangle out two target classes into one


analytic.data1$diabetes_status=as.factor(analytic.data1$diabetes_status)
analytic.data1$diabetes_status=Recode(analytic.data1$diabetes_status,"c('NA No', 'No NA')= 'No'")
analytic.data1$diabetes_status=Recode(analytic.data1$diabetes_status,"c('NA Yes', 'Yes NA')= 'Yes'")
str(analytic.data1$diabetes_status)



analytic.data1$diabetes_status=droplevels.factor(analytic.data1$diabetes_status, exclude = 'NA NA')
drop_na(analytic.data1$diabetes_status)



as.factor(analytic.data1$Dx_of_high_BP)# Convert BP to factor 
remove_labels(analytic.data1$Dx_of_high_BP)
analytic.data1$Dx_of_high_BP=as.factor(analytic.data1$Dx_of_high_BP)
summary(analytic.data1$Dx_of_high_BP)

analytic.data1$Dx_of_high_BP[analytic.data1$Dx_of_high_BP == 9] <- NA #Convert to 9 to NA, will deal with it later 
droplevels(analytic.data1$Dx_of_high_BP)
levels(analytic.data1$Dx_of_high_BP)= c("Yes", "No", NA) "Rename Factor with two levels: Yes and No. We will deal with NA values later

remove_labels(analytic.data1)


summary(analytic.data1)


summary(analytic.data1$Age)
hist(analytic.data1$Age) # Histogram shows age variable is skewed
analytic.data1_age1=log(analytic.data1$Age) #Experiment with different transformations 


#Drop unwanted variables 
analytic.data1$Fastomg_Blood_Glucose.cut=NULL
analytic.data1$HGBA1C.cut=NULL
analytic.data1$HGBA1C=NULL
analytic.data1$Fastomg_Blood_Glucose=NULL


#Attach appropriate labels to factors of target variable
analytic.data1$diabetes_status <- revalue(analytic.data1$diabetes_status, c('NA [-Inf,5.7)'='No', '[-Inf,100) NA'='No', '[100, Inf) NA'= 'Yes', 'NA [5.7, Inf)'='Yes') 
                               )
#Wrangle race variable, white are at a higher risk for diabetes statuse comared to other ethinicities, group races in two groups: Non Hispanic White or other
analytic.data1$Race <- revalue(analytic.data1$Race, c('Mexican American'='Other', 'Other Hispanic'='Other', 'Non-Hispanic Black'='Other', 'Other Race - Including Multi-Rac'='Other'))

summary(analytic.data3)
#Convert age = 0 to NA
analytic.data1$Age[analytic.data1$Age == 0] <- NA

analytic.data2=subset(analytic.data1, select = c(diabetes_status, Age, Gender, Race, BMI, BP_mmHG, Total_Cholesterol,
                                                 Vit_D_nmol_L, Dx_of_high_BP))   #subset variables of interst 
summary(analytic.data2)
analytic.data2=subset(analytic.data2, select = diabetes_status:Dx_of_high_BP)

#Use MICE package to impute missing values, based on given parameters, You can experiment with different permutations 
#of parameters and use cross validation to see which permutation is the best
impdata=mice(analytic.data2, m = 1, method = c("logreg","norm.predict", "rf", "rf", "norm.predict", "norm.predict", "norm.predict", "norm.predict", "rf"), maxit=20)
impdata1=complete(impdata,1) 
summary(impdata1)

impdata1$Age=discretize(impdata1$Age, impdata1$diabetes_status)
#discretize age using Minimum description length (MDL) binning 




#Min/Max scalining applied to numeric variables 
impdata1$Age = scaler(impdata1$Age)
impdata1$BMI = scaler(impdata1$BMI)
impdata1$Vit_D_nmol_L = scaler(impdata1$Vit_D_nmol_L)
impdata1$BP_mmHG = scaler(impdata1$BP_mmHG)
impdata1$Total_Cholesterol = scaler(impdata1$Total_Cholesterol)

summary(impdata1)




fullmodel_BMII=impdata1
fullmodel=impdata1

fullmodelBP=impdata1

fullmodeltcho=impdata1
summary(fullmodel1)
fullmodel1=subset(fullmodel, select = c(Age, Vit_D_nmol_L, BP_mmHG, BMI, Total_Cholesterol))
predictors <- colnames(mydata)
Map logit response to the numeric variables
# Bind the logit and tidying the data for plot
mydata <- mydata %>%
  mutate(logit = log(probabilities/(1-probabilities))) %>%
  gather(key = "predictors", value = "predictor.value", -logit)

str(fullmodel)
kable(head(fullmodel), digits = 2, format = "html", row.names = TRUE)%>%
  kable_styling(full_width = T,
                font_size = 15) %>%
  column_spec(column = 2, bold = TRUE) %>%  # columns must be specified by number
  column_spec(column = 5, width = "2cm") %>%
  row_spec(row = 0, color = "#660033") %>%  # row = 0 allows us to format the header
  row_spec(row = 2, italic = TRUE) %>%
  row_spec(row = 3, color = "#104e8b", background = "#d3d3d3") %>%
  row_spec(row = 4, monospace = TRUE) %>%
  row_spec(row = 5, underline = TRUE) %>%
  row_spec(row = 6, strikeout = FALSE)

summary(fullmodel)
model3_na <- glm(diabetes_status ~., data = analytic.data2, 
             family = binomial)
# Predict the probability (p) of diabete positivity
probabilities <- predict(model, type = "response")
predicted.classes <- ifelse(probabilities > 0.5, "pos", "neg")

summary(model2_scaled)
summary(analytic.data2)
mydata <- impdata1 %>%
  dplyr::select_if(is.numeric) 
predictors <- colnames(mydata)
# Bind the logit and tidying the data for plot
mydata <- mydata %>%
  mutate(logit = log(probabilities/(1-probabilities))) %>%
  gather(key = "predictors", value = "predictor.value", -logit)
Blood_pressure = model.data$BP_mmHG


race=model.data$Race
White <- fullmodel[ which(fullmodel$Race =='Non-Hispanic White' & 
          fullmodel$diabetes_status == 'Yes'), ]


Female <- fullmodel[ which(fullmodel$Gender =='Female' & 
                            fullmodel$diabetes_status == 'Yes'), ]
summary(White)


Male  <- fullmodel[ which(fullmodel$Gender =='Male' & 
                              fullmodel$diabetes_status == 'Yes'), ]
noWhite  <- fullmodel[ which(fullmodel$Race =='Non-Hispanic White' & 
                                fullmodel$diabetes_status == 'No'), ]

noFemale  <- fullmodel[ which(fullmodel$Gender =='Female' & 
                               fullmodel$diabetes_status == 'No'), ]



noMale  <- fullmodel[ which(fullmodel$Gender =='Male' & 
                               fullmodel$diabetes_status == 'No'), ]
noOther  <- fullmodel[ which(fullmodel$Race =='Other' & 
fullmodel$diabetes_status == 'No'), ]


dat1 <- data.frame(
  sex = factor(c("Female","Female","Male","Male")),
  Status = factor(c("Yes","No","Yes","No"), levels=c("Yes","No")),
  Total_Instances = c(nrow(Female), nrow(noFemale), nrow(Male), nrow(noMale))
)

p <- ggplot(data=dat1, aes(x=Status, y=Total_Instances, fill=sex)) +
  geom_bar(stat="identity", position=position_dodge(), colour="black") +
  scale_fill_manual(values=c("#999999", "#E69F00"))

fig <- ggplotly(p)

fig
str(Male)
summary(model2_scaled)

fig <- plot_ly(ggplot2::diamonds, y = ~vitD, color = ~impdata1$diabetes_status, type = "box")

fig
ggplot(mydata, aes(logit, predictor.value))+
  geom_point(size = 0.5, alpha = 0.5) +
  geom_smooth(method = "loess") + 
  theme_bw() + 
  facet_wrap(~predictors, scales = "free_y")

plot(model, which = 4, id.n = 3)
model.data <- augment(model) %>% 
  mutate(index = 1:n()) 

model.data %>% top_n(3, .cooksd)

ggplot(model.data, aes(index, .std.resid)) + 
  geom_point(aes(color = diabetes_status), alpha = .5) +
  theme_bw()

model.data %>% 
  filter(abs(.std.resid) > 3)

car::vif(model)
summary(impdata1)
pairs.panels(fullmodel1) #scatterplot matrix to check colinearity, target variable removed, Age Group 1, 16-36


summary(fullmodel.glm1)
pchisq(13118, 1, lower.tail = FALSE)
summary(model1)
ggplot(fullmodel, aes(x=Vit_D_nmol_L, y=as.numeric(diabetes_status) - 1)) + 
  geom_point(alpha=.5) +
  stat_smooth(method="glm", se=FALSE, method.args = list(family=binomial))
s

PseudoR2(model, c("McFadden", "Nagel"))

PseudoR2(model3_na, c("McFadden", "Nagel"))

PseudoR2(fullmodel.glm3, c("McFadden", "Nagel"))

summary(model)




  
#Validate different models
set.seed(123) 
xgbGrid <- expand.grid(nrounds = c(100,200),  # this is n_estimators in the python code above
                       max_depth = c(10, 15, 20, 25),
                       colsample_bytree = seq(0.5, 0.9, length.out = 5),
                       ## The values below are default values in the sklearn-api. 
                       eta = 0.1,
                       gamma=0,
                       min_child_weight = 1,
                       subsample = 1
)


train.control <- trainControl(method = "cv", search = 'random', allowParallel = TRUE, number = 3)
model_rf <- train(diabetes_status ~., data = dresstrain, method = "rf",
               trControl = train.control, tuneLength = 20)

model_glm <- train(diabetes_status ~., data = dresstrain, method = "glm",
                   trControl = train.control, family = binomial)


model_glm1 <- train(diabetes_status ~., data = dresstrain1, method = "glm",
                   trControl = train.control, family = binomial)


model_glm2 <- train(diabetes_status ~., na.action = na.omit, data = dresstrain2, method = "glm",
                    trControl = train.control, family = binomial)
print(model_glm1)


confusionMatrix(model_glm2)
print(model_glm_BP)
print(model_glm_tcho)
confusionMatrix(model_glm1)

m <- matrix(c(51.7, 3.8, 6.0, 38.5), nrow = 2, ncol = 2)
fig <- plot_ly(
  x = c("T", "F"), y = c("A", "P"),
  z = m, type = "heatmap"
)
fig

confusionMatrix(model_glm)
print(model_rf)

plot(model_glm)
plot(model_rf)

table(model)
split1 <- sample.split(impdata1$diabetes_status, SplitRatio = 0.75)

spli2 <- sample.split(analytic.data2$diabetes_status, SplitRatio = 0.75)


split <- sample.split(fullmodel$diabetes_status, SplitRatio = 0.75)

#get training and test data
dresstrain1 <- subset(impdata1, split == TRUE)
dresstest1 <- subset(impdata1, split == TRUE)
dresstrain <- subset(fullmodel, split == TRUE)
dresstest <- subset(fullmodel, split == TRUE)
dresstrain2 <- subset(analytic.data2, split == TRUE)
dresstest2 <- subset(analytic.data2, split == TRUE)



str()


y=fullmodel$diabetes_status
db=fullmodel
test_data = subset(db,select = -c(diabetes_status))
model<- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification") %>%
  # Fit the model
  fit(diabetes_status ~., data = db)

ypred <- predict(model,
                 new_data = test_data,
                 type = "prob")

yscore <- data.frame(ypred$.pred_No)
rdb <- cbind(db$diabetes_status,yscore)
colnames(rdb) = c('diabetes_status','yscore')

summary(fullmodel$diabetes_status)
pdb <- roc_curve(rdb, diabetes_status, yscore)
pdb$specificity <- 1 - pdb$specificity
auc = roc_auc(rdb, diabetes_status, yscore)
auc = auc$.estimate 

tit = paste('ROC Curve (AUC = ',toString(round(auc,2)),')',sep = '')


fig <-  plot_ly(data = pdb ,x =  ~specificity, y = ~sensitivity, type = 'scatter', mode = 'lines', fill = 'tozeroy') %>%
  layout(title = tit,xaxis = list(title = "False Positive Rate"), yaxis = list(title = "True Positive Rate")) %>%
  add_segments(x = 0, xend = 1, y = 0, yend = 1, line = list(dash = "dash", color = 'black'),inherit = FALSE, showlegend = FALSE)
fig

fig <-  plot_ly(data = pdb ,x =  ~recall, y = ~precision, type = 'scatter', mode = 'lines', fill = 'tozeroy') %>%
  add_segments(x = 0, xend = 1, y = 1, yend = 0, line = list(dash = "dash", color = 'black'),inherit = FALSE, showlegend = FALSE) %>%
  layout(title = tit, xaxis = list(title = "Recall"), yaxis = list(title = "Precision") )

pdb <- pr_curve(rdb, y, yscore)
fig


