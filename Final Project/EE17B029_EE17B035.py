import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Reading in the files
train = pd.read_csv("train.csv") 
ratings = pd.read_csv("ratings.csv") 
remarks = pd.read_csv("remarks.csv") 
remarksso = pd.read_csv("remarks_supp_opp.csv")

# Getting effective support of remarks
remarks['supportcount'] = 0.
remarks['opposecount'] = 0.
for i in range(len(remarksso)):
    if(remarksso['support'][i]):
        remarks['supportcount'][remarks[remarks['remarkId'] == remarksso['remarkId'][i]].index.values] += 1
    elif(remarksso['oppose'][i]):
        remarks['opposecount'][remarks[remarks['remarkId'] == remarksso['remarkId'][i]].index.values] += 1
remarks['effectivesupport'] = remarks['supportcount']/(remarks['supportcount'] + remarks['opposecount'])
remarks['effectiveoppose'] = remarks['opposecount']/(remarks['supportcount'] + remarks['opposecount'])

# Getting date weighted ratings
ratings[['date','month','year']]=ratings.Date.str.split("-",expand=True)
ratings["date"]=ratings["date"].astype(int)
ratings["month"]=ratings["month"].astype(int)-1
ratings["year"]=ratings["year"].astype(int)-2014
ratings["nod"]=(366*ratings["year"]+30*ratings["month"]+ratings["date"])
ratings["prod"]=ratings["rating"]*ratings["nod"]

# Extracting features from the training data
ytrain = np.array(train["left"])
xtrain = np.zeros((len(train),32))

for i in range(len(train)):
    a = ratings.loc[(ratings['emp'] == train["emp"][i]) & (ratings['comp'] == train["comp"][i])]
                  
    f1 = len(a.loc[a["rating"] == 1])
    f2 = len(a.loc[a["rating"] == 2])
    f3 = len(a.loc[a["rating"] == 3])
    f4 = len(a.loc[a["rating"] == 4])
    f5 = (f1+2*f2+3*f3+4*f4)/(f1+f2+f3+f4) # Mean rating
    f6 = np.sum(a["prod"])/np.sum(a["nod"]) # Date weighted rating
    f7 = len(a) # Number of ratings
    x = a.loc[(a['Date'] == train["lastratingdate"][i])]
    f8 = 0
    if(len(x)!=0):
        f8 = np.sum(x["rating"])/len(x) # Average last date rating
    
    a = ratings.loc[(ratings['comp'] == train["comp"][i])]
                  
    f11 = len(a.loc[a["rating"] == 1])
    f12 = len(a.loc[a["rating"] == 2])
    f13 = len(a.loc[a["rating"] == 3])
    f14 = len(a.loc[a["rating"] == 4])
    f15 = (f1+2*f2+3*f3+4*f4)/(f1+f2+f3+f4) # Mean rating
    f16 = np.sum(a["prod"])/np.sum(a["nod"]) # Date weighted rating
    f17 = len(a) # Number of ratings
    
    b = remarks.loc[(remarks['emp'] == train["emp"][i]) & (remarks['comp'] == train["comp"][i])]
    
    e = b["txt"].str.len()
    e = e.fillna(0)
    f21 = 0
    if(len(e)!=0):
        f21 = sum(e)/len(e) # Average Remark Length
        
    f22 = len(b) # Number of Remarks
    f23 = -1
    f24 = -1
    if(len(b)!=0):
        f23 = np.mean(np.isnan(b['effectivesupport']) == False) # Average Support
        f24 = np.mean(np.isnan(b['effectiveoppose']) == False)
    
    b = remarks.loc[(remarks['comp'] == train["comp"][i])]
    
    e = b["txt"].str.len()
    e = e.fillna(0)
    f31 = 0
    if(len(e)!=0):
        f31 = sum(e)/len(e) # Average Remark Length
        
    f32 = len(b) # Number of Remarks
    f33 = -1
    f34 = -1
    if(len(b)!=0):
        f33 = np.mean(np.isnan(b['effectivesupport']) == False)
        f34 = np.mean(np.isnan(b['effectiveoppose']) == False)
    
    c = remarksso.loc[(remarksso['emp'] == train["emp"][i]) & (remarksso['comp'] == train["comp"][i])]
    f41 = len(c.loc[c["support"] == 1])
    f42 = len(c.loc[c["oppose"] == 1])
    f43 = -1
    f44 = -1
    if(len(c)!=0):
        f43 = f41/(f41+f42)
        f44 = f42/(f41+f42)
    
    c = remarksso.loc[(remarksso['comp'] == train["comp"][i])]
    f51 = len(c.loc[c["support"] == 1])
    f52 = len(c.loc[c["oppose"] == 1])
    f53 = -1
    f54 = -1
    if(len(c)!=0):
        f53 = f51/(f51+f52)
        f54 = f52/(f51+f52)
        
    dd,mm,yyyy = train["lastratingdate"][i].split("-")
    f61 = 366*(int(yyyy)-2014)+30*(int(mm)-1)+int(dd) # Date of last rating
    
    xtrain[i] = [f1,f2,f3,f4,f5,f6,f7,f8,
                 f11,f12,f13,f14,f15,f16,f17,
                 f21,f22,f23,f24,
                 f31,f32,f33,f34,
                 f41,f42,f43,f44,
                 f51,f52,f53,f54,
                 f61]

# Extracting features from the testing data
test = pd.read_csv("test.csv") 
xtest =np.zeros((len(test),32))

for i in range(len(test)):
    a = ratings.loc[(ratings['emp'] == test["emp"][i]) & (ratings['comp'] == test["comp"][i])]
                  
    f1 = len(a.loc[a["rating"] == 1])
    f2 = len(a.loc[a["rating"] == 2])
    f3 = len(a.loc[a["rating"] == 3])
    f4 = len(a.loc[a["rating"] == 4])
    f5 = (f1+2*f2+3*f3+4*f4)/(f1+f2+f3+f4) # Mean rating
    f6 = np.sum(a["prod"])/np.sum(a["nod"]) # Date weighted rating
    f7 = len(a) # Number of ratings
    x = a.loc[(a['Date'] == test["lastratingdate"][i])]
    f8 = 0
    if(len(x)!=0):
        f8 = np.sum(x["rating"])/len(x) # Average last date rating
    
    a = ratings.loc[(ratings['comp'] == test["comp"][i])]
                  
    f11 = len(a.loc[a["rating"] == 1])
    f12 = len(a.loc[a["rating"] == 2])
    f13 = len(a.loc[a["rating"] == 3])
    f14 = len(a.loc[a["rating"] == 4])
    f15 = (f1+2*f2+3*f3+4*f4)/(f1+f2+f3+f4) # Mean rating
    f16 = np.sum(a["prod"])/np.sum(a["nod"]) # Date weighted rating
    f17 = len(a) # Number of ratings
    
    b = remarks.loc[(remarks['emp'] == test["emp"][i]) & (remarks['comp'] == test["comp"][i])]
    
    e = b["txt"].str.len()
    e = e.fillna(0)
    f21 = 0
    if(len(e)!=0):
        f21 = sum(e)/len(e) # Average Remark Length
        
    f22 = len(b) # Number of Remarks
    f23 = -1
    f24 = -1
    if(len(b)!=0):
        f23 = np.mean(np.isnan(b['effectivesupport']) == False) # Average Support
        f24 = np.mean(np.isnan(b['effectiveoppose']) == False)
    
    b = remarks.loc[(remarks['comp'] == test["comp"][i])]
    
    e = b["txt"].str.len()
    e = e.fillna(0)
    f31 = 0
    if(len(e)!=0):
        f31 = sum(e)/len(e) # Average Remark Length
        
    f32 = len(b) # Number of Remarks
    f33 = -1
    f34 = -1
    if(len(b)!=0):
        f33 = np.mean(np.isnan(b['effectivesupport']) == False)
        f34 = np.mean(np.isnan(b['effectiveoppose']) == False)
    
    c = remarksso.loc[(remarksso['emp'] == test["emp"][i]) & (remarksso['comp'] == test["comp"][i])]
    f41 = len(c.loc[c["support"] == 1])
    f42 = len(c.loc[c["oppose"] == 1])
    f43 = -1
    f44 = -1
    if(len(c)!=0):
        f43 = f41/(f41+f42)
        f44 = f42/(f41+f42)
    
    c = remarksso.loc[(remarksso['comp'] == test["comp"][i])]
    f51 = len(c.loc[c["support"] == 1])
    f52 = len(c.loc[c["oppose"] == 1])
    f53 = -1
    f54 = -1
    if(len(c)!=0):
        f53 = f51/(f51+f52)
        f54 = f52/(f51+f52)
        
    dd,mm,yyyy = test["lastratingdate"][i].split("-")
    f61 = 366*(int(yyyy)-2014)+30*(int(mm)-1)+int(dd) # Date of last rating
    
    xtest[i] = [f1,f2,f3,f4,f5,f6,f7,f8,
                 f11,f12,f13,f14,f15,f16,f17,
                 f21,f22,f23,f24,
                 f31,f32,f33,f34,
                 f41,f42,f43,f44,
                 f51,f52,f53,f54,
                 f61]

# Weighted Accuracy Function
def lossfunct(output,target):
    ln=(target==0).sum()
    lp=(target==1).sum()
    fp=((output==target) & (target==1)).sum()
    fn=((output==target) & (target==0)).sum()
    loss=(5*fp+fn)/(5*lp+ln)
    return loss

# Random Forest Classifier Model Validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
accuracy = 0
for train_index, test_index in kf.split(xtrain):
    X_train, X_test = xtrain[train_index], xtrain[test_index]
    y_train, y_test = ytrain[train_index], ytrain[test_index]
    model = RandomForestClassifier(n_estimators = 1000, class_weight = {0: 1, 1: 5}, min_samples_leaf = 7)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy += lossfunct(y_pred, y_test)
print(accuracy/5)

# Running test data through the model and saving the output file for submission
model = RandomForestClassifier(n_estimators = 1000, class_weight = {0: 1, 1: 5}, min_samples_leaf = 7)
model.fit(xtrain,ytrain)
y_pred = model.predict(xtest)
df = pd.DataFrame(list(zip(test["id"], y_pred)), 
               columns =['id', 'left']) 
df.to_csv ('output.csv', index = False, header=True)