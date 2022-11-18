import pandas as pd
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,roc_curve
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import classification_report
from yellowbrick.classifier import ClassificationReport
from numpy import array
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from mlxtend.classifier import StackingClassifier
#from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv("Data.csv")
data.dropna()
print(data.head(5))

col=[]
col=data.columns.tolist()
col.remove('sl_no')
col.remove('gender')
col.remove('status')

#Convert string to lower case
data["ssc_b"]= data["ssc_b"].str.lower()
data["hsc_b"]= data["hsc_b"].str.lower()
data["hsc_s"]= data["hsc_s"].str.lower()
data["degree_t"]= data["degree_t"].str.lower()
data["workex"]= data["workex"].str.lower()
data["specialisation"]= data["specialisation"].str.lower()



#Preprocessing
ssc_b = {'central': 1, 'others': 2}
hsc_b = {'central': 1, 'others': 2}
hsc_s = {'arts': 1, 'commerce': 2, 'science': 3}
degree_t = {'comm&mgmt': 1, 'others': 2, 'sci&tech': 3}
workex = {'no': 1, 'yes': 2}
specialisation = {'mkt&hr': 1, 'mkt&fin': 2}
status={'Placed': 1, 'Not Placed': 0}


data.ssc_b = [ssc_b[item] for item in data.ssc_b]
data.hsc_b = [hsc_b[item] for item in data.hsc_b]
data.hsc_s = [hsc_s[item] for item in data.hsc_s]
data.degree_t = [degree_t[item] for item in data.degree_t]
data.workex = [workex[item] for item in data.workex]
data.specialisation = [specialisation[item] for item in data.specialisation]
data.status = [status[item] for item in data.status]

X = data.drop(['sl_no','gender','status'],1)
y=data['status']

# X.apply(lambda col: pd.factorize(col, sort=True)[0])
X_train, X_test, y_train, y_test = tts(
    X,
    y,
    test_size=0.3,
    random_state=0)
 
X_train.shape, X_test.shape



def plot_roc_curve(fpr, tpr):  
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


def XG(X_train,y_train,X_test,y_test):
  ada = XGBClassifier()
  ada.fit(X_train,y_train)
  print("XGBClassifier:train set")
  y_pred = ada.predict(X_train)
  pred=ada.predict_proba(X_test)   
  print("XGBClassifier:Confusion Matrix: ", confusion_matrix(y_train, y_pred))
  print ("XGBClassifier:Accuracy : ", accuracy_score(y_train,y_pred)*100)
  print("XGBClassifier:Test set")
  y_pred = ada.predict(X_test)
  print("XGBClassifier:Confusion Matrix: ", confusion_matrix(y_test, y_pred))
  print ("XGBClassifier:Accuracy : ", accuracy_score(y_test,y_pred)*100)
  #confusion Matrix
  matrix =confusion_matrix(y_test, y_pred)
  class_names=[0,1] 
  fig, ax = plt.subplots()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names)
  plt.yticks(tick_marks, class_names)
  sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
  ax.xaxis.set_label_position("top")
  plt.tight_layout()
  plt.title('Confusion matrix', y=1.1)
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')
  plt.show()
  #ROC_AUC curve
  probs = ada.predict_proba(X_test) 
  probs = probs[:, 1]  
  auc = roc_auc_score(y_test, probs)  
  print('AUC: %.2f' % auc)
  le = preprocessing.LabelEncoder()
  y_test1=le.fit_transform(y_test)
  fpr, tpr, thresholds = roc_curve(y_test1, probs)
  plot_roc_curve(fpr, tpr)
  #Classification Report
  target_names = ['Yes', 'No']
  prediction=ada.predict(X_test)
  print(classification_report(y_test, prediction, target_names=target_names))
  classes = ["Yes", "No"]
  visualizer = ClassificationReport(ada, classes=classes, support=True)
  visualizer.fit(X_train, y_train)  
  visualizer.score(X_test, y_test)  
  g = visualizer.poof()

#XG(X_train,y_train,X_test,y_test)


def RF(X_train,y_train,X_test,y_test):
  ada = RandomForestClassifier(bootstrap=True,max_depth= 70,max_features= 'auto',min_samples_leaf= 4,min_samples_split= 10,n_estimators= 400)
  ada.fit(X_train,y_train)
  print("Random Forest:train set")
  y_pred = ada.predict(X_train)
  pred=ada.predict_proba(X_test)   
  print("Random Forest:Confusion Matrix: ", confusion_matrix(y_train, y_pred))
  print ("Random Forest:Accuracy : ", accuracy_score(y_train,y_pred)*100)
  print("Random Forest:Test set")
  y_pred = ada.predict(X_test)
  print("Random Forest:Confusion Matrix: ", confusion_matrix(y_test, y_pred))
  print ("Random Forest:Accuracy : ", accuracy_score(y_test,y_pred)*100)
  #confusion Matrix
  matrix =confusion_matrix(y_test, y_pred)
  class_names=[0,1] 
  fig, ax = plt.subplots()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names)
  plt.yticks(tick_marks, class_names)
  sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
  ax.xaxis.set_label_position("top")
  plt.tight_layout()
  plt.title('Confusion matrix', y=1.1)
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')
  plt.show()
  #ROC_AUC curve
  probs = ada.predict_proba(X_test) 
  probs = probs[:, 1]  
  auc = roc_auc_score(y_test, probs)  
  print('AUC: %.2f' % auc)
  le = preprocessing.LabelEncoder()
  y_test1=le.fit_transform(y_test)
  fpr, tpr, thresholds = roc_curve(y_test1, probs)
  plot_roc_curve(fpr, tpr)
  #Classification Report
  target_names = ['Yes', 'No']
  prediction=ada.predict(X_test)
  print(classification_report(y_test, prediction, target_names=target_names))
  classes = ["Yes", "No"]
  visualizer = ClassificationReport(ada, classes=classes, support=True)
  visualizer.fit(X_train, y_train)  
  visualizer.score(X_test, y_test)  
  g = visualizer.poof()

#RF(X_train,y_train,X_test,y_test)

#KNN
def KNN(X_train,y_train,X_test,y_test):
  xgb=KNeighborsClassifier()
  xgb.fit(X_train,y_train)
  print("KNN:train set")
  y_pred = xgb.predict(X_train)
  pred=xgb.predict_proba(X_test)   
  print("KNN:Confusion Matrix: ", confusion_matrix(y_train, y_pred))
  print ("KNN:Accuracy : ", accuracy_score(y_train,y_pred)*100)
  print("KNN:Test set")
  y_pred = xgb.predict(X_test)
  print("KNN:Confusion Matrix: ", confusion_matrix(y_test, y_pred))
  print ("KNN:Accuracy : ", accuracy_score(y_test,y_pred)*100)
  #confusion Matrix
  matrix =confusion_matrix(y_test, y_pred)
  class_names=[0,1] 
  fig, ax = plt.subplots()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names)
  plt.yticks(tick_marks, class_names)
  sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
  ax.xaxis.set_label_position("top")
  plt.tight_layout()
  plt.title('Confusion matrix', y=1.1)
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')
  plt.show()
  #ROC_AUC curve
  probs = xgb.predict_proba(X_test) 
  probs = probs[:, 1]  
  auc = roc_auc_score(y_test, probs)  
  print('AUC: %.2f' % auc)
  le = preprocessing.LabelEncoder()
  y_test1=le.fit_transform(y_test)
  fpr, tpr, thresholds = roc_curve(y_test1, probs)
  plot_roc_curve(fpr, tpr)
  #Classification Report
  target_names = ['Yes', 'No']
  prediction=xgb.predict(X_test)
  print(classification_report(y_test, prediction, target_names=target_names))
  classes = ["Yes", "No"]
  visualizer = ClassificationReport(xgb, classes=classes, support=True)
  visualizer.fit(X_train, y_train)  
  visualizer.score(X_test, y_test)  
  g = visualizer.poof()

#KNN(X_train,y_train,X_test,y_test)

#NB Classifier
def NB(X_train,y_train,X_test,y_test):
  vc = GaussianNB()  
  vc.fit(X_train,y_train)
  print("Naive Bayes :train set")
  y_pred = vc.predict(X_train)
  #pred=vc.predict_proba(X_test)   
  print("Naive Bayes :Confusion Matrix: ", confusion_matrix(y_train, y_pred))
  print ("Naive Bayes :Accuracy : ", accuracy_score(y_train,y_pred)*100)
  print("Naive Bayes :Test set")
  y_pred = vc.predict(X_test)
  print("Naive Bayes :Confusion Matrix: ", confusion_matrix(y_test, y_pred))
  print ("Naive Bayes :Accuracy : ", accuracy_score(y_test,y_pred)*100)
  #confusion Matrix
  matrix =confusion_matrix(y_test, y_pred)
  class_names=[0,1] 
  fig, ax = plt.subplots()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names)
  plt.yticks(tick_marks, class_names)
  sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
  ax.xaxis.set_label_position("top")
  plt.tight_layout()
  plt.title('Confusion matrix', y=1.1)
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')
  plt.show()
  #ROC_AUC curve
  probs = vc.predict_proba(X_test) 
  probs = probs[:, 1]  
  auc = roc_auc_score(y_test, probs)  
  print('AUC: %.2f' % auc)
  le = preprocessing.LabelEncoder()
  y_test1=le.fit_transform(y_test)
  fpr, tpr, thresholds = roc_curve(y_test1, probs)
  plot_roc_curve(fpr, tpr)
  #Classification Report
  target_names = ['Yes', 'No']
  prediction=vc.predict(X_test)
  print(classification_report(y_test, prediction, target_names=target_names))
  classes = ["Yes", "No"]
  visualizer = ClassificationReport(vc, classes=classes, support=True)
  visualizer.fit(X_train, y_train)  
  visualizer.score(X_test, y_test)  
  g = visualizer.poof()

#NB(X_train,y_train,X_test,y_test)

def predict(data):
  
  print(data)
  # data = [ ssc_b.get(item,item) for item in data ]
  # data = [ hsc_b.get(item,item) for item in data ]
  # data = [ hsc_s.get(item,item) for item in data ]
  # data = [ degree_t.get(item,item) for item in data ]
  # data = [ workex.get(item,item) for item in data ]
  # data = [ specialisation.get(item,item) for item in data ]

  output_data=pd.DataFrame([data],columns = col)
  # output_data.ssc_b = [ssc_b[item] for item in output_data.ssc_b]
  # output_data.hsc_b = [hsc_b[item] for item in output_data.hsc_b]
  # output_data.hsc_s = [hsc_s[item] for item in output_data.hsc_s]
  # output_data.degree_t = [degree_t[item] for item in output_data.degree_t]
  # output_data.workex = [workex[item] for item in output_data.workex]
  # output_data.specialisation = [specialisation[item] for item in output_data.specialisation]
  knn1 = KNeighborsClassifier()
  xgb=XGBClassifier()
  nb1 = GaussianNB()
  rf2=RandomForestClassifier(bootstrap=True,max_depth= 70,max_features= 'auto',min_samples_leaf= 4,min_samples_split= 10,n_estimators= 400)

  classifiers=[knn1,nb1]
  sc1 = StackingClassifier(classifiers,meta_classifier=xgb) 

  sc1.fit(X_train,y_train)
  pred=sc1.predict(output_data)
  if(pred==0):
    prediction='Not Placed'
  else:
    prediction='Placed'
  print(prediction)
  return prediction

