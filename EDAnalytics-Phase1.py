#importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
%matplotlib inline
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
plt.style.use('ggplot')


def my_autopct(pct):
    return ('%1.0f%%' % pct) if pct > pct_cutoff else ''

#importing the dataset
df=pd.read_csv('US_accidents_for_5_states.csv')



#Our analysis is for the 5 states of CA, MA, KY, MD and LA
print(df.head(5))
#printing the first five entries
print(len(df))
#printing the length of the dataframe
#storing the abbrevations for the 5 states under study
#CA- California
#MA -Massachusetts
#KY -Kentucky
#MD -Maryland
#LA -Louisiana
state_abbreviation=['CA','MA','KY','MD','LA']


#plot of the locations of accidents in the 5 states
sns.regplot(x=df["Start_Lng"], y=df["Start_Lat"], fit_reg=False);
plt.show()



#number of accidents falling under each severity category of the 5 states
print(df.Severity.value_counts())



#countplot to analyse the number of accidents by severity
sns.set(style="darkgrid")
sns.countplot(x="Severity",data=df)


#state-wise number of accidents
print(df.State.value_counts())

#plot to see the distribution of accidents by severity in each state
for i in state_abbreviation:
    filtered_df = df[(df['State'] == i)]
    sns.countplot(x="Severity",data=filtered_df,palette = "Set2").set_title(i)
    plt.show()






severity=[2,3,4]
# Set a list of colors, markers and linestyles for plotting
color_lst=['r','b','k','g','m']
marker_lst=['D','o','*','+','s']
linestyle_lst=['dashed','dashdot','solid','dashdot','solid']

# Set a list of month, weekday, hour for reindex purpose and time_duration to clear the accident
month_lst = [ 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul','Aug','Sep','Oct','Nov','Dec']
weekday_lst = [ 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
weekday_lst_full = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
hour_lst= np.arange(24)
td='Time_Duration(min)'

df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')

# Extract year, month, day, hour, weekday and time_duration information
df['Start_Year']=df['Start_Time'].dt.year
df['Start_Month']=df['Start_Time'].dt.strftime('%b')
df['Start_Day']=df['Start_Time'].dt.day
df['Start_Hour']=df['Start_Time'].dt.hour
df['Start_Weekday']=df['Start_Time'].dt.strftime('%a')

# Extract the amount of time in the unit of minutes for each accident, round to the nearest integer
td='Time_Duration(min)'
df[td]=round((df['End_Time']-df['Start_Time'])/np.timedelta64(1,'m'))

allday_lst=df.Start_Time.astype(str).str.split(' ')
allday_lst2=[item[0] for item in allday_lst]
print('For the 5 states')
print('There are {} total accidents.'.format(df.shape[0]))
print('There are {} total days.'.format(len(allday_lst2)))
print('There are {} unique days.'.format(len(set(allday_lst2))))
print('On average, there are {} accidents per day.'.format(round(df.shape[0]/len(set(allday_lst2)))))




#day wise number of accidents
sns.countplot(x="Start_Weekday",data=df)



# Find out how many days (Monday-Sunday) between the beginning and end of this dataset.
calendar_weekday_num=[]
d1=df.Start_Time.min()
d2=df.Start_Time.max()
for i in range(7):
    count = 0
    for d_ord in range(d1.toordinal(), d2.toordinal()+1):
        d = date.fromordinal(d_ord)
        if (d.weekday() == i):
            count += 1
    calendar_weekday_num.append(count)

print('Number of days for Monday-Sunday: {}.'.format(calendar_weekday_num))
print('Total number of days between {} and {}: {} days.'.format(d1,d2,sum(calendar_weekday_num)))




state_lst=['CA','MA','KY','MD','LA']
# For each state, find out how many unique days for each weekday/weekend
# Initialize an empty list to hold the number of days for each weekday/weekend for the three states
weekday_num_state=[]
for state in state_lst:
    weekday_num=[]    
    for weekday in weekday_lst:
        # Slice the dataframe for specific state & weekday
        df_weekday=df[(df['State']==state) & (df.Start_Weekday==weekday)]
        # For each weekday, extract the day information from the Start_Time column
        day_lst1=df_weekday.Start_Time.astype(str).str.split(' ')
        # Extract the day information
        day_lst2=[item[0] for item in day_lst1]
        # Append the day into the list weekday_num
        weekday_num.append(len(set(day_lst2)))
        # Append the day with state information encoded into the list weekday_num_state
    weekday_num_state.append(weekday_num)
print('For the states of {}, here is the list of numbers of weekdays (Mon-Sun): {}.'.format(state_lst,weekday_num_state))




#percentage of days included for each state
day_pct_lst=[]
for i,state in enumerate(state_lst):
    day_pct=[round(int(item1)/int(item2),2)*100 for item1,item2 in zip(weekday_num_state[i],calendar_weekday_num)]
    day_pct_lst.append(day_pct)
    print('For the state of {}, the percentage of days with accident during this period in the data set: {}%.'.format(state_lst[i], day_pct))
print(day_pct_lst)





#TIME SERIES ANALYSIS resampled by months
# Set the start_time as the index for resampling purpose
df.set_index('Start_Time',drop=True,inplace=True)
fig= plt.figure(figsize=(15,6))
for i,state in enumerate(state_lst):
    plt.subplot(1, 6, 1+i)
     # Slice the dataframe for the specific state and weekday
    df[df['State']==state].resample('M').count()['ID'].plot(linestyle=linestyle_lst[i], color=color_lst[i])
    plt.xlim('2016','2019-Mar')
    plt.xlabel('Year')
    plt.title('{}'.format(state))
plt.show()





#day wise countplot of accidents for each state
for i in state_abbreviation:    
    filtered_df = df[(df['State'] == i)]
    sns.countplot(x="Start_Weekday",data=filtered_df).set_title(i)
    plt.show()





#PIE chart distribution of accidents day-wise for each state
feature='Start_Weekday'
fig_x=len(state_lst)
# Divide the total number of accidents by the number of unique days
fig= plt.figure(figsize=(5*fig_x,6))
pct_cutoff=2
for i,state in enumerate(state_lst):
    plt.subplot(1, 6, 1+i)    
    # Slice the dataframe for the specific state and weekday
    df_temp=df[df['State']==state].groupby('Start_Weekday').count()['ID'].reindex(weekday_lst)    
    df_temp2=[round(int(item1)/int(item2)) for item1,item2 in zip(df_temp,weekday_num_state[i])]    
    df_temp2=pd.Series(df_temp2)    
    labels = [n if v > pct_cutoff/100 else '' for n, v in zip(df_temp.index, df_temp)]     
    plt.pie(df_temp2, labels=labels, autopct=my_autopct, shadow=True)        
    plt.axis('equal')
    plt.xlabel('Weekday/Weekend')
    plt.title(state)
plt.tight_layout()
plt.show()










#TIME SERIES analysis for states resampled by hour of day -all days
fig= plt.figure(figsize=(18,6))
plt.subplot(1, 3, 1)
df[df['State']=='CA'].groupby('Start_Hour').count()['ID'].reindex(hour_lst).plot(linestyle='dashed',color='r')
df[df['State']=='MA'].groupby('Start_Hour').count()['ID'].reindex(hour_lst).plot(linestyle='dashdot',color='b')
df[df['State']=='KY'].groupby('Start_Hour').count()['ID'].reindex(hour_lst).plot(linestyle='solid',color='k')
df[df['State']=='MD'].groupby('Start_Hour').count()['ID'].reindex(hour_lst).plot(linestyle='dashed',color='g')
df[df['State']=='LA'].groupby('Start_Hour').count()['ID'].reindex(hour_lst).plot(linestyle='dashdot',color='m')
plt.ylabel('Number of accidents')
plt.xlabel('Hour')
plt.legend(['CA','MA','KY','MD','LA'])
plt.title('All days')
plt.xticks(np.arange(0, 24, step=2))



#TIME SERIES analysis for states resampled by hour of day -weekdays
# Weekdays
plt.subplot(1, 3, 2)
df[(df['State']=='CA') & (df['Start_Weekday'].isin(weekday_lst[:5]))].groupby('Start_Hour').count()['ID'].reindex(hour_lst).plot(linestyle='dashed',color='r')
df[(df['State']=='MA') & (df['Start_Weekday'].isin(weekday_lst[:5]))].groupby('Start_Hour').count()['ID'].reindex(hour_lst).plot(linestyle='dashdot',color='b')
df[(df['State']=='KY') & (df['Start_Weekday'].isin(weekday_lst[:5]))].groupby('Start_Hour').count()['ID'].reindex(hour_lst).plot(linestyle='solid',color='k')
df[(df['State']=='MD') & (df['Start_Weekday'].isin(weekday_lst[:5]))].groupby('Start_Hour').count()['ID'].reindex(hour_lst).plot(linestyle='dashed',color='g')
df[(df['State']=='LA') & (df['Start_Weekday'].isin(weekday_lst[:5]))].groupby('Start_Hour').count()['ID'].reindex(hour_lst).plot(linestyle='dashdot',color='m')
plt.xlabel('Hour')
plt.legend(['CA','MA','KY','MD','LA'])
plt.title('Weekdays')
plt.xticks(np.arange(0, 24, step=2))





#TIME SERIES analysis for states resampled by hour of day -weekends
# Weekends
plt.subplot(1, 3, 3)
df[(df['State']=='CA') & (df['Start_Weekday'].isin(weekday_lst[5:]))].groupby('Start_Hour').count()['ID'].reindex(hour_lst).plot(linestyle='dashed',color='r')
df[(df['State']=='MA') & (df['Start_Weekday'].isin(weekday_lst[5:]))].groupby('Start_Hour').count()['ID'].reindex(hour_lst).plot(linestyle='dashdot',color='b')
df[(df['State']=='KY') & (df['Start_Weekday'].isin(weekday_lst[5:]))].groupby('Start_Hour').count()['ID'].reindex(hour_lst).plot(linestyle='solid',color='k')
df[(df['State']=='MD') & (df['Start_Weekday'].isin(weekday_lst[5:]))].groupby('Start_Hour').count()['ID'].reindex(hour_lst).plot(linestyle='dashed',color='g')
df[(df['State']=='LA') & (df['Start_Weekday'].isin(weekday_lst[5:]))].groupby('Start_Hour').count()['ID'].reindex(hour_lst).plot(linestyle='dashdot',color='m')
plt.xlabel('Hour')
plt.legend(['CA','MA','KY','MD','LA'])
plt.title('Weekend')
plt.xticks(np.arange(0, 24, step=2))
plt.tight_layout()
plt.show()








#state wise PIE Chart indicating the percentage of accidents occuring in each city of the corresponding state
feature='City'
fig= plt.figure(figsize=(15,6))
pct_cutoff=2.5
for i,state in enumerate(state_lst):    
    plt.subplot(1, 6, 1+i)
    # Slice the dataframe for the specific state and feature
    df_temp=df[df['State']==state][feature].value_counts(normalize=True).round(8)
    labels = [n if v > pct_cutoff/100 else ''
              for n, v in zip(df_temp.index, df_temp)]     
    plt.pie(df_temp, labels=labels, autopct=my_autopct, shadow=True)    
    plt.axis('equal')
    plt.xlabel(feature)
    plt.title(state)
plt.xlabel(feature)
plt.show()





#state wise PIE Chart indicating the percentage of accidents occuring in each location of the corresponding state
feature='Accident location'
df.set_index('State',drop=True,inplace=True)
# State is the index when selecting bool type data as df_bool
df_bool=df.select_dtypes(include=['bool'])
df.reset_index(inplace=True)
fig= plt.figure(figsize=(15,6))
pct_cutoff=2.5
for i,state in enumerate(state_lst):    
    plt.subplot(1, 6, 1+i)
    # Slice the dataframe for the specific state and feature
    df_temp=df_bool[df_bool.index==state]
    df_temp=(df_temp.sum(axis=0)/df_temp.sum(axis=0).sum()).sort_values()    
    labels = [n if v > pct_cutoff/100 else ''
              for n, v in zip(df_temp.index, df_temp)] 
    plt.pie(df_temp, labels=labels, autopct=my_autopct, shadow=True)    
    plt.axis('equal')
    plt.xlabel(feature)
    plt.title(state)
plt.xlabel(feature)
plt.show()









#state wise PIE Chart indicating the percentage of accidents occuring during each weather condition of the corresponding state
df[df['State']==state]['Weather_Condition'].value_counts(normalize=True).round(5)
# The weather condition for each state
feature='Weather_Condition'
fig= plt.figure(figsize=(15,6))
pct_cutoff=2
for i,state in enumerate(state_lst):    
    plt.subplot(1, 6, 1+i)
    # Slice the dataframe for the specific state and feature
    df_temp=df[df['State']==state][feature].value_counts(normalize=True).round(2)
    labels = [n if v > pct_cutoff/100 else ''
              for n, v in zip(df_temp.index, df_temp)]     
    plt.pie(df_temp, labels=labels, autopct=my_autopct, shadow=True)    
    plt.axis('equal')
    plt.xlabel(feature)
    plt.title(state)
plt.tight_layout()
plt.show()





#boxplot of temperatures for each state
sns.boxplot(data=df,x="State",y="Temperature(F)")
plt.show()


#boxplot of windchill for each state
sns.boxplot(data=df,x="State",y="Wind_Chill(F)")
plt.show()


#boxplot of humidity for each state
sns.boxplot(data=df,x="State",y="Humidity(%)")
plt.show()



#boxplot of pressure for each state
sns.boxplot(data=df,x="State",y="Pressure(in)")
plt.show()


#boxplot of visibility for each state
sns.boxplot(data=df,x="State",y="Visibility(mi)")
plt.show()


#boxplot of windspeed for each state
sns.boxplot(data=df,x="State",y="Wind_Speed(mph)")
plt.show()


#boxplot of precipitation for each state
sns.boxplot(data=df,x="State",y="Precipitation(in)")
plt.show()