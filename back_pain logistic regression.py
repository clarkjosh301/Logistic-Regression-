#data from www.kaggle.com/sammy123/lower-back-pain-symptoms-dataset
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn import metrics
back_doc = r'C:\Users\CHESTER\Documents\dataset_spine2.csv'
back_df = pd.read_csv(back_doc)
status_cleaned = {'status': {'Abnormal':0, 'Normal':1}}
back_df.replace(status_cleaned, inplace=True)
X = back_df.drop(['status'], axis=1)
y = back_df['status']
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())
#drop P-values equaling 1 and re-run model, inorder to get rid of convergence error
X = back_df.drop(['status', 'pelvic_incidence ', 'pelvic tilt ', 'sacral_slope'], axis=1)
log_model=sm.Logit(y,X)
result=log_model.fit()
print(result.summary2())
#get rid of P-values over .05
X = back_df.drop(['status', 'pelvic_incidence ', 'pelvic tilt ', 'sacral_slope', 'pelvic_radius ', 'pelvic_slope', 'direct_tilt', 'thoracic_split', 'sacrum_angle', 'scoliosis_slope'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic model on test set:{:.3f}'.format(logreg.score(X_test, y_test)))
#accuracy=.817
