
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

print("training model")
gb = GradientBoostingClassifier(learning_rate=0.1,max_depth=2,n_estimators=450)

gb.fit(X_train,y_train)

#########################################################################################################
## Grading ##

# Import Grading Data
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