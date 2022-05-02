import pandas as pd
import seaborn as sns
from helpers.eda import *
from helpers.data_prep import *
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


# 1. Exploratory Data Analysis
# 2. Data Preprocessing
# 3. Model & Prediction
# 4. Model Validation: 10-Fold Cross Validation



def load():
    data = pd.read_csv("hitters.csv")
    return data

df = load()


#############################################
# 1. Exploratory Data Analysis
#############################################

check_df(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

[check_outlier(df, col) for col in num_cols]

df.isnull().sum()

df[df.isnull().any(axis=1)]


df.groupby(cat_cols).agg({"Salary": "mean", "Hits": "mean"})

for col in cat_cols:
    print(df[col].name, df[col].unique())


for col in cat_cols:
    cat_summary(df, col, False)


#############################################
# 2. Data Preprocessing
#############################################


# Missing value deletion
df.isnull().values.any()
df = df.dropna()

df_num = df.select_dtypes(include=['float64', 'int64'])

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df_num)
df_scores = clf.negative_outlier_factor_
df_scores[0:5]


scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 20], style='.-')
plt.show()

th = np.sort(df_scores)[5]

np.sort(df_scores)[5]
df[df_scores < th]
df[df_scores < th].shape
df[df_scores < th].index


# Local outlier factor deletion

df.shape

df = df.drop(axis=0, labels=df[df_scores < th].index)



def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in cat_cols:
    label_encoder(df, col)

df.head()

#############################################
# 3. Model & Prediction
#############################################

# Multiple Linear Regression
#############################

X = df.drop('Salary', axis=1)
y = df[["Salary"]]

#########
# Model
#########

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=1)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# constant (b - bias)
reg_model.intercept_

# coefficients (w - weights)
reg_model.coef_


#############################################
# 4. Model Validation
#############################################

# MSE
y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)

# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

# MAE
mean_absolute_error(y, y_pred)

# TRAIN RKARE
reg_model.score(X_train, y_train)

# Test RKARE
reg_model.score(X_test, y_test)

# 10 fold CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=10, scoring="neg_mean_squared_error")))


# questiob
df_with_predictions = pd.DataFrame(y_test)
df_with_predictions["y_pred"] = y_pred

plt.scatter(df_with_predictions["Salary"], df_with_predictions["y_pred"] , color='red', alpha=0.1)
plt.show()
