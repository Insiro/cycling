import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,RobustScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from matplotlib import pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows',None)

#clustering

#Data Definition
df_station = pd.read_csv("station.csv")
df_trip = pd.read_csv("trip.csv",parse_dates=['starttime','stoptime'])
df_station_GCS = pd.DataFrame(df_station.iloc[:,2:4],columns = ['lat','long']);

#feature selection
df_trip.drop(['trip_id', 'bikeid', 'tripduration','from_station_name', 'to_station_name', 'usertype', 'gender', 'birthyear'],axis=1,inplace=True)
df_station.drop(['name', 'lat', 'long', 'install_date','install_dockcount', 'modification_date', 'current_dockcount','decommission_date'],axis=1,inplace=True)

#feature creation
df_trip['day']=df_trip['stoptime'].dt.day.astype('category')
df_trip['starttime']=df_trip['stoptime'].dt.hour.astype('category')
df_trip['stoptime']=df_trip['stoptime'].dt.hour.astype('category')

#scaling data
standard_scaler = StandardScaler().fit(df_station_GCS)
df_station_GCS = standard_scaler.transform(df_station_GCS)
scaled_df = pd.DataFrame(df_station_GCS,columns=['lat','long'])

model = KMeans(init="k-means++", n_clusters=4, n_init=12)
model.fit(scaled_df)

#save cluseter id
df_station['cluster'] = model.labels_

#replace cluster id and station id
len = len(df_station['station_id'])
for i in range(0,len):
    df_trip.replace(df_station['station_id'][i],df_station['cluster'][i],inplace=True)

df_trip['from_station_id']=df_trip['from_station_id'].astype('category')
df_trip['to_station_id']=df_trip['to_station_id'].astype('category')

grouped_df=df_trip.groupby(['to_station_id','stoptime']).count().reset_index()

#visualize clustering results
cluster_id = ['A','B','C','D']
color = ['r','g','b','violet']

for i in range(0,4):
    group = grouped_df.loc[grouped_df['to_station_id']==i]
    plt.subplot(2,2,i+1)
    plt.bar(group['stoptime'],group['starttime'],color=color[i],alpha=0.5)
    plt.title(cluster_id[i])
    plt.xlabel('time')
    plt.ylabel('count')
    
plt.show()

#regression

#feature creation&selection
df=df_trip.drop(['starttime'],axis=1)
df=df.groupby(['to_station_id','stoptime','day']).count().reset_index()
df.drop(['day'],axis=1,inplace=True)

#repeated regression by cluster
for i in range(0,4):
    group = df.loc[df['to_station_id']==i]
    print("\ncluster"+cluster_id[i])

    #scaling data
    standard_scaler = RobustScaler().fit(group)
    group = standard_scaler.transform(group)
    scaled_df = pd.DataFrame(group,columns=['to_station_id','stoptime','from_station_id'])

    #target Data and Learning Data
    df_x=scaled_df.drop(['to_station_id','from_station_id'],axis=1)
    df_y=scaled_df['from_station_id']
    
    #set degree for polynomial regression
    poly_features = PolynomialFeatures(degree=9, include_bias=True)

    #repeated regression 3 times
    for j in range(0,3):
        X_train, X_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,shuffle=True)

        model = LinearRegression()
        x_poly = poly_features.fit_transform(X_train.values)
        model.fit(x_poly, y_train.values)

        X_new_poly = poly_features.transform(X_test.values)
        y_pred = model.predict(X_new_poly)

        print("R2 score of take ",str(i+1),":",r2_score(y_test, y_pred))
        print("MSE of take ",str(i+1),"     :",mean_squared_error(y_test, y_pred))
    #visualize regression results
    plt.subplot(1,4,i+1)
    plt.title("cluster"+cluster_id[i])
    plt.scatter(df_x, df_y, s=5,color=color[i])

    xx = np.linspace(-1, 1, 100).reshape(100, 1)
    X_new_poly = poly_features.transform(xx)
    y_pred = model.predict(X_new_poly)
    plt.plot(xx, y_pred, color='black')

plt.show()