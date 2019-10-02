import pickle
with open("C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\log\\2019-10-02_01-43-44.pickle","rb") as f1:
    data_list1 = pickle.load(f1)
with open("C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-02_01-44-18.pickle","rb") as f2:
    data_list2 = pickle.load(f2)


Frame = []
Status = []
Ballposition = []
PlatformPosition = []
Bricks = []
for i in range(0,len(data_list1)):
    Frame.append(data_list1[i].frame)
    Status.append(data_list1[i].status)
    Ballposition.append(data_list1[i].ball)
    PlatformPosition.append(data_list1[i].platform)
    Bricks.append(data_list1[i].bricks)

for i in range(0,len(data_list2)):
    Frame.append(data_list2[i].frame)
    Status.append(data_list2[i].status)
    Ballposition.append(data_list2[i].ball)
    PlatformPosition.append(data_list2[i].platform)
    Bricks.append(data_list2[i].bricks)    

import numpy as np
PlatX = np.array(PlatformPosition)[:,0][:, np.newaxis]
PlatX_next = PlatX[1:,:]
instruct = (PlatX_next - PlatX[0:len(PlatX_next),0][:,np.newaxis])/5
Ballarray = np.array(Ballposition[:-1])
x = np.hstack((Ballarray,PlatX[0:-1,0][:,np.newaxis]))
y = instruct

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x , y, test_size = 0.2,random_state = 41)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train,y_train)

yknn_bef_scaler = knn.predict(x_test)
acc_knn_bef_scaler = accuracy_score(yknn_bef_scaler , y_test)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train_stdnorm = scaler.transform(x_train)
knn.fit(x_train_stdnorm , y_train)
x_test_stdnorm = scaler.transform(x_test)
yknn_aft_scaler = knn.predict(x_test_stdnorm)#svm
acc_knn_aft_scaler = accuracy_score(yknn_aft_scaler,y_test)

filename = "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\ml\\knn_example.sav"
pickle.dump(knn , open(filename,'wb'))