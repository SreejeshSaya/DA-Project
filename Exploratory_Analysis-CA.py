#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from datetime import  date
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
%matplotlib inline
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
plt.style.use('ggplot')





#importing dataset
df=pd.read_csv('US_accidents_for_5_states.csv')



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
#filtering out the accident records occuring in the state of California
dd=dd[dd['State']=='CA']
# dd=dd[dd['State']=='KY']
# dd=dd[dd['State']=='MD']
# dd=dd[dd['State']=='LA']
# dd=dd[dd['State']=='MA']
#length of the new dataframe
print(len(dd))
print(len(dd.columns))




#removing the columns having more than 50 percent of missing values
cols = dd.columns[dd.isnull().mean()>0.5]
dd.drop(cols, axis=1,inplace=True)
#printing shape of dataframe
print(dd.shape)



#removing the unwanted columns
unwanted_cols=['Turning_Loop','Civil_Twilight','Nautical_Twilight','Astronomical_Twilight','Weather_Timestamp','TMC']
dd.drop(unwanted_cols, axis=1,inplace=True)
#finding the number of null values
print(dd.isnull().sum())






'''DATA CLEANING - Treating the columns by replacing the missing values with suitable outlier treatment'''

#treating temperature column
bin_size=math.floor(1+3.322*math.log(len(dd),10))
#computing suitable bin size for histogram
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








#treating humidity column 
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

dd_humidity=dd["Humidity(%)"]
#plotting the humidity histogram
dd_humidity.hist(bins=bin_size)
print(dd["Humidity(%)"].isnull().sum())
#filling the missing values with the mean
median_humidity=dd_humidity.median()
mean_humidity=dd_humidity.mean()
print(mean_humidity,median_humidity)
print(mean_humidity)
#filling the median temperature into the missing values since we are keeping the outliers
dd["Humidity(%)"].fillna(mean_humidity, inplace=True)
#ensuring that there are no more null values in temperature
print(dd['Humidity(%)'].isnull().sum())








#treating pressure column
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





#treating visibility column
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

#using the concept of flooring and capping
quartile_10=dd['Visibility(mi)'].quantile(0.10) 
quartile_90=dd['Visibility(mi)'].quantile(0.90) 
print(dd['Visibility(mi)'].skew()) 
dd_Visibility=dd["Visibility(mi)"]
#plotting the Visibility histogram
dd_Visibility.hist(bins=bin_size)
print(dd["Visibility(mi)"].isnull().sum())
#filling the missing values with the concept of flooring and capping
median_Visibility=dd_Visibility.median()
mean_Visibility=dd_Visibility.mean()
print(mean_Visibility,median_Visibility)
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




#treating wind speed column
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
#using the concept of flooring and capping
quartile_10=dd['Wind_Speed(mph)'].quantile(0.10) 
quartile_90=dd['Wind_Speed(mph)'].quantile(0.90) 
print(dd['Wind_Speed(mph)'].skew())
#Filling the wind_speed column missing values with its mean
dd_Windspeed=dd["Wind_Speed(mph)"]
#plotting the wind_speed histogram
dd_Windspeed.hist(bins=bin_size)
print(dd["Wind_Speed(mph)"].isnull().sum())
#filling the missing values with the concept of flooring and capping
median_Windspeed=dd_Windspeed.median()
mean_Windspeed=dd_Windspeed.mean()
print(mean_Windspeed,median_Windspeed)
dd["Wind_Speed(mph)"] = np.where(dd["Wind_Speed(mph)"] <quartile_10, quartile_10,dd['Wind_Speed(mph)'])
dd["Wind_Speed(mph)"] = np.where(dd["Wind_Speed(mph)"] >quartile_90, quartile_90,dd['Wind_Speed(mph)'])
print(dd['Visibility(mi)'].skew())
#new mean and median
new_median_Windspeed=dd_Windspeed.median()
new_mean_Windspeed=dd_Windspeed.mean()
print(new_mean_Windspeed,new_median_Windspeed)
dd["Wind_Speed(mph)"].fillna(new_median_Windspeed, inplace=True)
#ensuring that there are no more null values in wind_speed
print(dd['Wind_Speed(mph)'].isnull().sum())






#Wind_Direction
#filling the missing values in the column with the mode 
mode_found=(dd['Wind_Direction'].mode())
print(mode_found[0])
dd["Wind_Direction"].fillna(mode_found[0], inplace=True)
print(dd['Wind_Direction'].isnull().sum())
print(dd.isnull().sum())



#Weather_Condition
#filling the missing values in the column with the mode 
mode_found=(dd['Weather_Condition'].mode())
print(mode_found[0])
dd["Weather_Condition"].fillna(mode_found[0], inplace=True)
print(dd['Weather_Condition'].isnull().sum())



#City
#filling the missing values in the column category with the mode 
mode_found=(dd['City'].mode())
print(mode_found[0])
dd["City"].fillna(mode_found[0], inplace=True)
print(dd['City'].isnull().sum())





#Sunrise_sunset
#filling the missing values in the column with the mode 
mode_found=(dd['Sunrise_Sunset'].mode())
print(mode_found[0])
dd["Sunrise_Sunset"].fillna(mode_found[0], inplace=True)
print(dd['Sunrise_Sunset'].isnull().sum())


#dropping the cols 'Zipcode','Timezone','Airport_Code' 
unwanted_cols_2=['Zipcode','Timezone','Airport_Code']
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
#one hot encoding was used only to test logistic regression 
#other classification methods along with Logistic regression used label encoding




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



'''LOGISTIC REGRESSION'''
#creating the model, predicting the values for the test set and retrieving the accuracy score for the test set
lr = LogisticRegression(random_state=0)
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
# Get the accuracy score
acc=accuracy_score(y_test, y_pred)
print(acc)
#printing confusion matrix
print(confusion_matrix(y_test, y_pred))

#using PCA
# put none to n_componenets to create explained variance vector 
# ( contain the percentage of variance explained by each of the principal components that we extracted here.)
pca = PCA(n_components=2) 
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
expained_variance = pca.explained_variance_ratio_
#Fitting logistic Regression to the training set
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train_pca, y_train)
#Predicting the test set results
y_pred = classifier.predict(X_test_pca)
acc=accuracy_score(y_test, y_pred)
print(acc)
print(confusion_matrix(y_test, y_pred))





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




'''LOGISTIC REGRESSION'''
#creating the model, predicting the values for the test set and retrieving the accuracy score for the test set
lr = LogisticRegression(random_state=0)
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
# Get the accuracy score
acc=accuracy_score(y_test, y_pred)
print(acc)
print(confusion_matrix(y_test, y_pred))



#using PCA
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




'''k- Nearest Neighbours'''
#creating the model, predicting the values for the test set and retrieving the accuracy score for the test set
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
#Elbow method
# Loop over different values of k 
for i, n_neighbor in enumerate(neighbors):    
    # Setup a k-NN Classifier with n_neighbor
    knn = KNeighborsClassifier(n_neighbors=n_neighbor)
    # Fit the classifier to the training data
    knn.fit(X_train,y_train)    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)
# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train) 
accuracy = knn.score(X_test, y_test) 
print(accuracy)
knn_predictions = knn.predict(X_test)  
cm = confusion_matrix(y_test, knn_predictions) 
print(cm)




'''Decision Tree'''
#creating the model, predicting the values for the test set and retrieving the accuracy score for the test set
#using entropy as criterion
dtree_model = DecisionTreeClassifier(max_depth = 10, criterion='entropy').fit(X_train, y_train) 
y_pred = dtree_model.predict(X_test) 
acc=accuracy_score(y_test, y_pred)  
# creating a confusion matrix 
print(acc)
cm = confusion_matrix(y_test, y_pred) 
print(cm)

#using gini index as criterion
dtree_model = DecisionTreeClassifier(max_depth = 10, criterion='gini').fit(X_train, y_train) 
y_pred = dtree_model.predict(X_test) 
acc=accuracy_score(y_test, y_pred)  
# creating a confusion matrix 
print(acc)
cm = confusion_matrix(y_test, y_pred) 
print(cm)




'''Random Forest'''
#creating the model, predicting the values for the test set and retrieving the accuracy score for the test set
rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
acc=accuracy_score(y_test, y_pred)
print(acc)
print(confusion_matrix(y_test, y_pred))




'''Naive Bayes'''
#creating the model, predicting the values for the test set and retrieving the accuracy score for the test set
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
acc=accuracy_score(y_test, y_pred)
print(acc)
print(confusion_matrix(y_test, y_pred))
