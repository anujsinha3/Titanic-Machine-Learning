import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('titanic_data.csv')
df.fillna(value=0,inplace=True)
test=pd.read_csv('test.csv')
test.fillna(value=0,inplace=True)
def make_it(name,status):
    
    #status=df[name].unique()
    for i,val in zip(range((len(df[name]))),df[name]):
        for j,typ in zip(range(len(status)),status):
            if(val==typ):
                df.set_value(i,name,j)
    return
def make_itt(name,status):
    #print "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    for i,val in zip(range((len(test[name]))),test[name]):
        #print test['PassengerId'][i]
        for j,typ in zip(range(len(status)),status):
            if(val==typ):
                test.set_value(i,name,j)
    return

from sklearn.model_selection import train_test_split




print "yo"
Survived=df[df['Survived']==1]['Sex'].value_counts()
Dead=df[df['Survived']==0]['Sex'].value_counts()

#print Survived
#print Dead
#print df['Sex'].unique()
#print test['Sex'].unique()
#print df['Embarked'].unique()
#print test['Embarked'].unique()
arr=df['Sex'].unique()
make_it('Sex',arr)
make_itt('Sex',arr)
arr=df['Embarked'].unique()
make_it('Embarked',arr)
make_itt('Embarked',arr)

X=df.ix[:,['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
y=df.ix[:,['Survived']]
check=test.ix[:,['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
#df.Survived.value_counts().plot(kind='bar', alpha=0.55)

Age = df['Age']

print check

print X
#print y

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
clf1=KNeighborsClassifier(n_neighbors=50,weights='distance')
clf2=SVC(kernel='linear')
clf4=LogisticRegression(C=5)



#clf1.fit(X_train,y_train)
#y_pred1=clf1.predict(X_test)
#acc1=accuracy_score(y_test,y_pred1)
#print acc1

clf4.fit(X_train,y_train)
y_pred4=clf4.predict(X_test)
acc4=accuracy_score(y_test,y_pred4)
print acc4

#clf2.fit(X_train,y_train)
#y_pred2=clf2.predict(X_test)
#acc2=accuracy_score(y_test,y_pred2)
#print acc2

ans=clf4.predict(check)

do=[]

for idd,val in zip(test['PassengerId'],ans):
    arr=[idd,val]
    do.append(arr)
do=np.array(do)
print do
#print len(ans)
#print len(test['PassengerId'])
#print len(X_test)

import csv
csvout = csv.writer(open("mydata.csv", "wb"))
csvout.writerow(('PassengerId','Survived'))
for idd, val in do:
    csvout.writerow((idd, val))


#prob=clf.predict_proba(X_test)
#plt.plot()
#plt.show()






