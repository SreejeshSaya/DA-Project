import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import  date
%matplotlib inline
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
plt.style.use('ggplot')
df=pd.read_csv('US_accidents_for_5_states.csv')



from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder




from sklearn.neighbors import KNeighborsClassifier

# Import DecisionTreeClassifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier

# Import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# Import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc





df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')

# Extract year, month, day, hour and weekday
df['Year']=df['Start_Time'].dt.year
df['Month']=df['Start_Time'].dt.strftime('%b')
df['Day']=df['Start_Time'].dt.day
df['Hour']=df['Start_Time'].dt.hour
df['Weekday']=df['Start_Time'].dt.strftime('%a')

# Extract the amount of time in the unit of minutes for each accident, round to the nearest integer
td='Time_Duration(min)'
df[td]=round((df['End_Time']-df['Start_Time'])/np.timedelta64(1,'m'))
df.info()


dd=df.copy()


dd=dd[dd['State']=='KY']

print(len(dd))


print(len(dd.columns))
cols = dd.columns[dd.isnull().mean()>0.5]
dd.drop(cols, axis=1,inplace=True)




print(dd.shape)








unwanted_cols=['Turning_Loop','Civil_Twilight','Nautical_Twilight','Astronomical_Twilight','Weather_Timestamp','TMC']
dd.drop(unwanted_cols, axis=1,inplace=True)



print(dd.shape)
print(dd.isnull().sum())
print(len(dd))


import math
bin_size=math.floor(1+3.322*math.log(len(dd),10))
print(bin_size)


#plotting the box plots and finding the outliers
sns.boxplot(data=dd,x=dd['Temperature(F)'])
Q1=dd['Temperature(F)'].quantile(0.25)
Q3=dd['Temperature(F)'].quantile(0.75)
IQR=Q3-Q1
print(Q1)
print(Q3)
print(IQR)
iqr_range=1.5*IQR
Lower_bound=Q1-iqr_range
Upper_bound=Q3+iqr_range
lower_outliers=dd[dd['Temperature(F)']<Lower_bound]
upper_outliers=dd[dd['Temperature(F)']>Upper_bound]
print(len(lower_outliers),len(upper_outliers))
print(dd['Temperature(F)'].skew())










#Filling the temperature column missing values with its mean
dd_temperature=dd["Temperature(F)"]
#plotting the temperature histogram
dd_temperature.hist(bins=bin_size)
#approximately follows normal distribution
print(dd["Temperature(F)"].isnull().sum())
#filling the missing values with the median
median_temperature=dd_temperature.median()
mean_temperature=dd_temperature.mean()
print(mean_temperature,median_temperature)
print(median_temperature)
#filling the median temperature into the missing values since we are keeping the outliers
dd["Temperature(F)"].fillna(median_temperature, inplace=True)
#ensuring that there are no more null values in temperature
print(dd['Temperature(F)'].isnull().sum())









#plotting the box plots and finding the outliers
sns.boxplot(data=dd,x=dd['Humidity(%)'])
Q1=dd['Humidity(%)'].quantile(0.25)
Q3=dd['Humidity(%)'].quantile(0.75)
IQR=Q3-Q1
print(Q1)
print(Q3)
print(IQR)
iqr_range=1.5*IQR
Lower_bound=Q1-iqr_range
Upper_bound=Q3+iqr_range
lower_outliers=dd[dd['Humidity(%)']<Lower_bound]
upper_outliers=dd[dd['Humidity(%)']>Upper_bound]
print(len(lower_outliers),len(upper_outliers))









#Filling the humidity column missing values with its mean
dd_humidity=dd["Humidity(%)"]
#plotting the humidity histogram
dd_humidity.hist(bins=bin_size)
#approximately follows normal distribution
print(dd["Humidity(%)"].isnull().sum())
#filling the missing values with the mean
median_humidity=dd_humidity.median()
mean_humidity=dd_humidity.mean()
print(mean_humidity,median_humidity)
#calculating the median and mean humidity and it is left-skewed ditribution
print(mean_humidity)
#filling the median temperature into the missing values since we are keeping the outliers
dd["Humidity(%)"].fillna(mean_humidity, inplace=True)
#ensuring that there are no more null values in temperature
print(dd['Humidity(%)'].isnull().sum())









#plotting the box plots and finding the outliers
sns.boxplot(data=dd,x=dd['Pressure(in)'])
Q1=dd['Pressure(in)'].quantile(0.25)
Q3=dd['Pressure(in)'].quantile(0.75)
IQR=Q3-Q1
print(Q1)
print(Q3)
print(IQR)
iqr_range=1.5*IQR
Lower_bound=Q1-iqr_range
Upper_bound=Q3+iqr_range
lower_outliers=dd[dd['Pressure(in)']<Lower_bound]
upper_outliers=dd[dd['Pressure(in)']>Upper_bound]
print(len(lower_outliers),len(upper_outliers))



#using the concept of flooring and capping
quartile_10=dd['Pressure(in)'].quantile(0.10)
print(quartile_10) #29.28
quartile_90=dd['Pressure(in)'].quantile(0.90)
print(quartile_90)
print(dd['Pressure(in)'].skew())  




#Filling the Pressure column missing values with its mean
dd_Pressure=dd["Pressure(in)"]
#plotting the Pressure histogram
dd_Pressure.hist(bins=bin_size)
#approximately follows normal distribution
print(dd["Pressure(in)"].isnull().sum())
#filling the missing values with the concept of flooring and capping
median_Pressure=dd_Pressure.median()
mean_Pressure=dd_Pressure.mean()
print(mean_Pressure,median_Pressure)
dd["Pressure(in)"] = np.where(dd["Pressure(in)"] <quartile_10, quartile_10,dd['Pressure(in)'])
dd["Pressure(in)"] = np.where(dd["Pressure(in)"] >quartile_90, quartile_90,dd['Pressure(in)'])
print(dd['Pressure(in)'].skew())
new_median_Pressure=dd_Pressure.median()
new_mean_Pressure=dd_Pressure.mean()
print(new_mean_Pressure,new_median_Pressure)
dd["Pressure(in)"].fillna(new_median_Pressure, inplace=True)
#ensuring that there are no more null values in pressure
print(dd['Pressure(in)'].isnull().sum())









#visibility

#plotting the box plots and finding the outliers
sns.boxplot(data=dd,x=dd['Visibility(mi)'])
Q1=dd['Visibility(mi)'].quantile(0.25)
Q3=dd['Visibility(mi)'].quantile(0.75)
IQR=Q3-Q1
print(Q1)
print(Q3)
print(IQR)
iqr_range=1.5*IQR
Lower_bound=Q1-iqr_range
Upper_bound=Q3+iqr_range
lower_outliers=dd[dd['Visibility(mi)']<Lower_bound]
upper_outliers=dd[dd['Visibility(mi)']>Upper_bound]
print(len(lower_outliers),len(upper_outliers))


#it is skewed


#using the concept of flooring and capping
quartile_10=dd['Visibility(mi)'].quantile(0.10) 
quartile_90=dd['Visibility(mi)'].quantile(0.90) 
print(dd['Visibility(mi)'].skew())  #4.37




#Filling the Visibility column missing values with its mean
dd_Visibility=dd["Visibility(mi)"]
#plotting the Pressure histogram
dd_Visibility.hist(bins=bin_size)
#approximately follows normal distribution
print(dd["Visibility(mi)"].isnull().sum())
#filling the missing values with the concept of flooring and capping
median_Visibility=dd_Visibility.median()
mean_Visibility=dd_Visibility.mean()
print(mean_Visibility,median_Visibility)
#calculating the median and mean pressure and it is left-skewed ditribution
dd["Visibility(mi)"] = np.where(dd["Visibility(mi)"] <quartile_10, quartile_10,dd['Visibility(mi)'])
dd["Visibility(mi)"] = np.where(dd["Visibility(mi)"] >quartile_90, quartile_90,dd['Visibility(mi)'])
print(dd['Visibility(mi)'].skew())
#new mean and median
new_median_Visibility=dd_Visibility.median()
new_mean_Visibility=dd_Visibility.mean()
print(new_mean_Visibility,new_median_Visibility)
dd["Visibility(mi)"].fillna(new_median_Visibility, inplace=True)
#ensuring that there are no more null values in visibility
print(dd['Visibility(mi)'].isnull().sum())









print(dd.isnull().sum())

















#plotting the box plots and finding the outliers
sns.boxplot(data=dd,x=dd['Wind_Speed(mph)'])
Q1=dd['Wind_Speed(mph)'].quantile(0.25)
Q3=dd['Wind_Speed(mph)'].quantile(0.75)
IQR=Q3-Q1
print(Q1)
print(Q3)
print(IQR)
iqr_range=1.5*IQR
Lower_bound=Q1-iqr_range
Upper_bound=Q3+iqr_range
lower_outliers=dd[dd['Wind_Speed(mph)']<Lower_bound]
upper_outliers=dd[dd['Wind_Speed(mph)']>Upper_bound]
print(len(lower_outliers),len(upper_outliers))


#it is skewed


#using the concept of flooring and capping
quartile_10=dd['Wind_Speed(mph)'].quantile(0.10) #0
quartile_90=dd['Wind_Speed(mph)'].quantile(0.90) #13.8
print(dd['Wind_Speed(mph)'].skew())  #38.513



#Filling the Visibility column missing values with its mean
dd_Windspeed=dd["Wind_Speed(mph)"]
#plotting the Pressure histogram
dd_Windspeed.hist(bins=bin_size)
#approximately follows normal distribution
print(dd["Wind_Speed(mph)"].isnull().sum())
#filling the missing values with the concept of flooring and capping
median_Windspeed=dd_Windspeed.median()
mean_Windspeed=dd_Windspeed.mean()
print(mean_Windspeed,median_Windspeed)
#calculating the median and mean pressure and it is left-skewed ditribution
dd["Wind_Speed(mph)"] = np.where(dd["Wind_Speed(mph)"] <quartile_10, quartile_10,dd['Wind_Speed(mph)'])
dd["Wind_Speed(mph)"] = np.where(dd["Wind_Speed(mph)"] >quartile_90, quartile_90,dd['Wind_Speed(mph)'])
print(dd['Visibility(mi)'].skew())
#improved skew value from 38 to -1.84
#new mean and median

new_median_Windspeed=dd_Windspeed.median()
new_mean_Windspeed=dd_Windspeed.mean()
print(new_mean_Windspeed,new_median_Windspeed)
dd["Wind_Speed(mph)"].fillna(new_median_Windspeed, inplace=True)
#ensuring that there are no more null values in visibility
print(dd['Wind_Speed(mph)'].isnull().sum())






#Wind_Direction

#filling the category with the mode 
mode_found=(dd['Wind_Direction'].mode())
print(mode_found[0])
dd["Wind_Direction"].fillna(mode_found[0], inplace=True)
print(dd['Wind_Direction'].isnull().sum())




print(dd.isnull().sum())



#Weather_Condition
#filling the category with the mode 
mode_found=(dd['Weather_Condition'].mode())
print(mode_found[0])
dd["Weather_Condition"].fillna(mode_found[0], inplace=True)
print(dd['Weather_Condition'].isnull().sum())





#City
#filling the category with the mode 
mode_found=(dd['City'].mode())
print(mode_found[0])
dd["City"].fillna(mode_found[0], inplace=True)
print(dd['City'].isnull().sum())




#Sunrise_sunset
#filling the category with the mode 
mode_found=(dd['Sunrise_Sunset'].mode())
print(mode_found[0])
dd["Sunrise_Sunset"].fillna(mode_found[0], inplace=True)
print(dd['Sunrise_Sunset'].isnull().sum())
#dropping the cols 'Zipcode','Timezone','Airport_Code' 
unwanted_cols_2=['Zipcode','Timezone','Airport_Code','Wind_Chill(F)','Precipitation(in)']
dd.drop(unwanted_cols_2, axis=1,inplace=True)





print(dd.isnull().mean())
print(dd.info())


#correlation plots for numerical features
numerical_features=['Start_Lat','Start_Lng','Temperature(F)','Humidity(%)','Pressure(in)','Visibility(mi)','Wind_Speed(mph)']
df_numerical=dd[numerical_features].copy()
print(df_numerical.shape)
sns.heatmap(df_numerical.corr(), annot = True)
print(df_numerical.corr())




#removing the unwanted columns
filtered_columns=['Start_Time','End_Time','Description','Street','State','Country','Unnamed: 0','ID']
dd.drop(filtered_columns, axis=1,inplace=True)
print(dd.shape)



#Performing a standardisation on numerical columns such that mean is 0 and variance is 1
# numerical features
num_cols = ['Temperature(F)','Humidity(%)','Pressure(in)','Visibility(mi)','Wind_Speed(mph)','Distance(mi)']
# apply standardization on numerical features
for i in num_cols:
    # fit on training data column
    scale = StandardScaler().fit(dd[[i]])
    # transform the training data column
    dd[i] = scale.transform(dd[[i]])




#Approach using one hot encoding/pd.get_dummies for few columns
#one hot encoding approach



#Copying the dataframe into ds
ds=dd.copy()




#using pd_getdummies i.e one hot encoding
features_converted=['Side','Bump','Crossing','Give_Way','Junction','No_Exit','Railway','Roundabout','Station','Stop','Traffic_Calming','Traffic_Signal','Sunrise_Sunset','Year','Month','Day','Hour','Weekday']
for i in features_converted:
    
    ds = pd.concat([ds,pd.get_dummies(ds[i], prefix=i)],axis=1)
    ds.drop([i],axis=1, inplace=True)




#using label encoding to the rest of the columns having object datatype
features_label_encoding=['Source','City','County','Wind_Direction','Weather_Condition']
labelencoder = LabelEncoder()
# Assigning numerical values and storing in another column

for i in features_label_encoding:
    ds[i] = labelencoder.fit_transform(ds[i])




target='Severity'
# Create arrays for the features and the response variable
# set X and y
y = ds[target]
X = ds.drop(target, axis=1)

# Split the data set into training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)
lr = LogisticRegression(random_state=0)
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)

# Get the accuracy score
acc=accuracy_score(y_test, y_pred)
print(acc)









#using PCA
from sklearn.decomposition import PCA
# put none to n_componenets to create explained variance vector 
# ( contain the percentage of variance explained by each of the principal components that we extracted here.)
pca = PCA(n_components=2) 
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
expained_variance = pca.explained_variance_ratio_

#Fitting logistic Regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train_pca, y_train)

#Prdicting the test set results
y_pred = classifier.predict(X_test_pca)
acc=accuracy_score(y_test, y_pred)
print(acc)





#Approach using labelencoding for all few columns
#labelencoding approach



#Copying the dataframe into dr

dr=dd.copy()
features_label_encoding=['Source','Side','City','County','Wind_Direction','Weather_Condition','Bump','Crossing','Give_Way','Junction','No_Exit','Railway','Roundabout','Station','Stop','Traffic_Calming','Traffic_Signal','Sunrise_Sunset','Year','Month','Day','Hour','Weekday']
labelencoder = LabelEncoder()
# Assigning numerical values and storing in another column

for i in features_label_encoding:
    dr[i] = labelencoder.fit_transform(dr[i])





target='Severity'
# Create arrays for the features and the response variable
# set X and y
y = dr[target]
X = dr.drop(target, axis=1)
# Split the data set into training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)
lr = LogisticRegression(random_state=0)
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
# Get the accuracy score
acc=accuracy_score(y_test, y_pred)
print(acc)



#using PCA
from sklearn.decomposition import PCA
# put none to n_componenets to create explained variance vector 
# ( contain the percentage of variance explained by each of the principal components that we extracted here.)
pca = PCA(n_components=2) 
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
expained_variance = pca.explained_variance_ratio_

#Fitting logistic Regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train_pca, y_train)

#Prdicting the test set results
y_pred = classifier.predict(X_test_pca)
acc=accuracy_score(y_test, y_pred)
print(acc)
















    
   







