
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,accuracy_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split



# Grading script can be found from line 115 to 135
# File path can be found on line 25 and 142
# Grading file path can be found on line 120

#########################################################################################################
#####################################SECTION 1 - CLASSIFICATION##########################################
#########################################################################################################

print("reading input dataset for classification")

df = pd.read_excel('Kickstarter.xlsx')

def pre_processing_classification(df):

    df = df[(df['state'] == 'successful') | (df['state'] == 'failed')]

    df = df.dropna()

    # drop all columns that does not exist at the launch of a project
    df.drop(['pledged','backers_count','state_changed_at','staff_pick',
            'backers_count','spotlight',
            'usd_pledged',
            'state_changed_at_weekday',
            'state_changed_at_month',
            'state_changed_at_day','state_changed_at_yr',
            'state_changed_at_hr','launch_to_state_change_days'
            ],axis=1,inplace=True)

    # convert goal to usd
    df['goal'] = df['goal']*df['static_usd_rate']

    # drop columns for curerncy conversion
    df.drop(['static_usd_rate','currency'],axis=1,inplace=True)

    # drop all columns that have the same value
    df.drop(['disable_communication'],axis=1,inplace=True)

    # drop deadline_at, state_change_at, created_at, launched_at since the pre-processing is already done in other columns
    df.drop(['deadline','created_at','launched_at'], axis=1, inplace=True)

    # Assuming df is your DataFrame and 'column' is the column from which you want to remove outliers
    z_scores = np.abs(stats.zscore(df['goal']))
    df = df[(z_scores < 3)]

    df.drop(['id','name'],axis=1,inplace=True)

    # hours seem to be too granular, thus is dropped
    df.drop(['created_at_hr','launched_at_hr','deadline_hr'],axis=1,inplace=True)

    # drop some dates
    df.drop(['launched_at_day','launched_at_month','launched_at_yr'],axis=1,inplace=True)
    df.drop(['created_at_weekday','launched_at_weekday','deadline_weekday'],axis=1,inplace=True)

    # country turn out to not be very useful
    df.drop(['country'],axis=1,inplace=True)

    # there are ~1,300 projects that has null values for category, fill with unknown
    df['category'].fillna('unknown',inplace=True)

    # dummify categorical variables
    df = pd.get_dummies(data=df,columns=['state'],drop_first=True)
    df = pd.get_dummies(data=df,columns=['category'],drop_first=True)
    df.rename({"state_successful":"state"},axis=1,inplace=True)

    print("finished pre-processing for classification")
    return df

df = pre_processing_classification(df)

# train test split
print("train test split for classification")
X = df.drop(['state'],axis=1)
y = df['state']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)

# hyperparameter tuning
# gb = GradientBoostingClassifier()
# gb_params = {'n_estimators':[400,425,450,475,500],
#              'learning_rate':[0.001,0.05,0.1],
#              'max_depth':[1,2],
#              'warm_start':[True,False]
#              }
# gb_grid = GridSearchCV(gb,gb_params,cv=10)
# gb_grid.fit(X_train,y_train)

# print(gb_grid.best_params_)
# print(gb_grid.best_score_)
# print(gb_grid.best_estimator_)

print("training model")
gb = GradientBoostingClassifier(learning_rate=0.1,max_depth=2,n_estimators=450)

gb.fit(X_train,y_train)

# evaluate model
print(classification_report(y_test,gb.predict(X_test)))



#########################################################################################################
## Grading ##

# Import Grading Data
print("reading grading dataset")
kickstarter_grading_df = pd.read_excel("Kickstarter-Grading.xlsx")

# Pre-Process Grading Data
kickstarter_grading_df = pre_processing_classification(kickstarter_grading_df.dropna())

# Setup the variables
X_grading = kickstarter_grading_df.drop(["state"], axis=1)
y_grading = kickstarter_grading_df["state"]

# Apply the model previously trained to the grading data
y_grading_pred = gb.predict(X_grading)

# Calculate the accuracy score
print(accuracy_score(y_grading, y_grading_pred))

#########################################################################################################
#####################################SECTION 2 - CLUSTERING##############################################
#########################################################################################################
numerical_columns = ['goal','backers_count','usd_pledged','name_len_clean','blurb_len_clean','create_to_launch_days','launch_to_deadline_days','launch_to_state_change_days']
scaler = StandardScaler()

print("reading input dataset for clustering")
df = pd.read_excel('Kickstarter.xlsx')

def pre_processing_clustering(df, numerical, scaler):
    df = df[(df['state'] == 'successful') | (df['state'] == 'failed')]
    df = df.dropna()

    df.drop(['pledged'], axis=1, inplace=True)

    # convert goal to usd
    df['goal'] = df['goal']*df['static_usd_rate']

    # drop columns for curerncy conversion
    df.drop(['static_usd_rate', 'currency'], axis=1, inplace=True)

    # drop all columns that have the same value
    df.drop(['disable_communication'], axis=1, inplace=True)

    # drop deadline_at, state_change_at, created_at, launched_at since the pre-processing is already done in other columns
    df.drop(['deadline', 'created_at', 'launched_at','state_changed_at'], axis=1, inplace=True)

    # remove outliers
    z_scores = np.abs(stats.zscore(df['goal']))
    df = df[(z_scores < 3)]
    z_scores = np.abs(stats.zscore(df['usd_pledged']))
    df = df[(z_scores < 3)]

    df.drop(['id', 'name'], axis=1, inplace=True)

    # drop time information that is too granular
    df.drop(['created_at_hr', 'launched_at_hr',
            'deadline_hr','state_changed_at_hr'], axis=1, inplace=True)
    df.drop(['launched_at_day', 'launched_at_month', 'launched_at_yr',
            'launched_at_weekday'], axis=1, inplace=True)
    df.drop(['created_at_weekday','state_changed_at_weekday','deadline_weekday'],axis=1,inplace=True)
    df.drop(['created_at_day','state_changed_at_day','deadline_day'],axis=1,inplace=True)

    # these are highly correlated with state_changed_at_yr
    df.drop(['created_at_yr','deadline_yr'], axis=1, inplace=True)

    # highly correlated with name_len_clean and blurb_len_clean
    df.drop(['name_len', 'blurb_len'], axis=1, inplace=True)

    df = pd.get_dummies(data=df, columns=['state'], drop_first=True)
    df = pd.get_dummies(data=df, columns=['country'], drop_first=True)
    df = pd.get_dummies(
        data=df, columns=['created_at_month','state_changed_at_month','deadline_month'], drop_first=True)


    df[numerical] = scaler.fit_transform(df[numerical])
    df = pd.get_dummies(data=df,columns=['category'],drop_first=True)
    df = pd.get_dummies(data=df,columns=['state_changed_at_yr'],drop_first=True)
    df.rename({"state_successful":"state"},axis=1,inplace=True)

    print("finished pre-processing for clustering")
    return df

df2 = pre_processing_clustering(df,numerical_columns,scaler)

###### Model Selection ######
# inertia = []
# silouette = []

# # Finding optimal K
# for i in range (2,10):    
#     kmeans = KMeans(n_clusters=i)
#     model = kmeans.fit(df2)
#     labels = model.labels_
#     inertia.append(kmeans.inertia_)
#     silouette.append(silhouette_score(df2,labels))

# plt.figure(figsize=(10,5))
# plt.plot(range(2, 10), silouette)
# plt.title('Silouette Score vs Number of Clusters')
# plt.xlabel('Number of clusters')
# plt.ylabel('Silouette Score')
# plt.savefig('./silouette_score.png')

# plt.figure(figsize=(10,5))
# plt.plot(range(2, 10), inertia)
# plt.title('Elbow Method - Inertia')
# plt.xlabel('Number of clusters')
# plt.ylabel('Inertia')
# plt.savefig('./elbow_method.png')

######## selected model ########
kmeans_selected = KMeans(n_clusters=3)
model_seletected = kmeans_selected.fit(df2)

# look at charactereistics of each cluster
df_cluster = df2.copy()
df_cluster['cluster'] = model_seletected.labels_
k_means_res = df_cluster

#convert the standardized data back to original scale
k_means_res[numerical_columns] = scaler.inverse_transform(k_means_res[numerical_columns])

# convert categorical variables back to original scale
country_col_names = [col for col in k_means_res.columns if col.startswith('country_')]
country_cols = k_means_res[country_col_names]
k_means_res.drop(country_col_names,axis=1,inplace=True)
k_means_res['country'] = country_cols.idxmax(axis=1).apply(lambda x: x.replace("country_",""))

country_col_names = [col for col in k_means_res.columns if col.startswith('category_')]
country_cols = k_means_res[country_col_names]
k_means_res.drop(country_col_names,axis=1,inplace=True)
k_means_res['category'] = country_cols.idxmax(axis=1).apply(lambda x: x.replace("category_",""))

############print results############
print("="*50)
print("discription of each cluster")

print("cluster 0")
cluster_0 = k_means_res[k_means_res['cluster']==0]
print(cluster_0.describe())

print("cluster 1")
cluster_1 = k_means_res[k_means_res['cluster']==1]
print(cluster_1.describe())

print("cluster 2")
cluster_2 = k_means_res[k_means_res['cluster']==2]
print(cluster_2.describe())



print("="*50)
# distribution of country vs cluster
plt.figure(figsize=(10,5))
sns.countplot(x='country',hue='cluster',data=k_means_res)
plt.title('Country vs Cluster')
plt.xlabel('Country')
plt.ylabel('Count')
plt.savefig('./country_vs_cluster.png')

#details of each cluster
print("cluster 0")
cluster_0 = k_means_res[k_means_res['cluster']==0]
print(cluster_0.describe())

print("cluster 1")
cluster_1 = k_means_res[k_means_res['cluster']==1]
print(cluster_1.describe())

print("cluster 2")
cluster_2 = k_means_res[k_means_res['cluster']==2]
print(cluster_2.describe())

# distribution of category vs cluster
plt.figure(figsize=(10,5))
sns.countplot(x='category',hue='cluster',data=k_means_res)
plt.title('Category vs Cluster')
plt.xlabel('Category')
plt.ylabel('Count')
plt.savefig('./category_vs_cluster.png')

print("="*50)

# distribution of goal of each cluster
plt.figure(figsize=(10,5))
sns.boxplot(x='cluster',y='goal',data=k_means_res)
plt.title('Goal vs Cluster')
plt.xlabel('Cluster')
plt.ylabel('Goal')
plt.savefig('./goal_by_cluster.png')


# project success rate for each cluster
plt.figure(figsize=(10,5))
sns.barplot(x='cluster',y='state',data=k_means_res)
plt.title('Success Rate vs Cluster')
plt.xlabel('Cluster')
plt.ylabel('Success Rate')
plt.savefig('./success_rate_by_cluster.png')