# Random Forest
# Solution to Kaggle Digit Recognizer problem, with result  0.93214 (839 place)

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

df_train = pd.read_csv('train_digits.csv')

y = df_train.pop('label')
X = df_train.as_matrix()

clf = RandomForestClassifier(n_estimators=1000, max_features='log2')
clf.fit(X, y)

test = pd.read_csv(('test_digits.csv')).values
pred = clf.predict(test)

np.savetxt('submission.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')