import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import tensorflow.contrib.learn.python.learn as learn
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('bank_note_data.csv')
data.head()

sns.countplot(x='Class', data=data)
sns.pairplot(data=data, hue='Class')

scaler = StandardScaler()

scaler.fit(data.drop('Class', axis=1))
scaled_features = scaler.fit_transform(data.drop('Class', axis=1))

df_feat = pd.DataFrame(scaled_features, columns=data.columns[:-1])
df_feat.head()

X = df_feat
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

classifier = learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=2)

classifier.fit(X_train, y_train, steps=200, batch_size=20)
predictions = classifier.predict(X_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# Random forest classifier



rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)
print(classification_report(y_test, rfc_pred))
