import pickle

path_list = [
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_00-16-37.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_00-16-03.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_00-15-30.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_00-14-56.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_00-14-23.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-08_22-58-31.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-08_22-57-58.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_00-31-01.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_00-31-34.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_00-32-08.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_00-32-41.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_00-33-15.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_00-33-49.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_00-34-23.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_00-34-56.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_00-35-29.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_00-36-03.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_00-36-36.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_00-37-09.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_00-42-45.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_00-43-19.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_00-43-52.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_00-44-26.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_00-44-59.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_00-45-32.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_00-46-06.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_00-46-39.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_00-47-13.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_00-47-46.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_00-48-19.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_00-48-53.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_00-52-21.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_01-14-20.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_01-14-53.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_01-15-27.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_01-16-00.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_01-16-34.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_01-17-08.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_01-17-41.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_01-18-15.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_01-18-48.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_01-19-21.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_01-19-55.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_01-20-28.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_01-21-02.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_01-21-36.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_01-22-09.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_01-22-43.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_01-23-16.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_01-23-50.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_01-24-23.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_01-24-57.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_01-25-31.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_01-26-04.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_01-26-38.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_01-27-11.pickle",
        "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\log\\2019-10-09_01-27-45.pickle"
        ]   

for flie_list in path_list :
    with open(flie_list,"rb") as f :
        data_list = pickle.load(f)


    Frame = []
    Status = []
    Ballposition = []
    PlatformPosition = []
    Bricks = []
    for i in range(0,len(data_list)):
        Frame.append(data_list[i].frame)
        Status.append(data_list[i].status)
        Ballposition.append(data_list[i].ball)
        PlatformPosition.append(data_list[i].platform)
        Bricks.append(data_list[i].bricks)
        

import numpy as np
np.set_printoptions(threshold=np.inf)
PlatX = np.array(PlatformPosition)[:,0][:,np.newaxis]
PlatX_next = PlatX[1:,:]
instruct = (PlatX_next - PlatX[0:len(PlatX_next),0][:, np.newaxis])/5

BallX=np.array(Ballposition)[:,0][:,np.newaxis]
BallX_next=BallX[1:,:]
vx=(BallX_next-BallX[0:len(BallX_next),0][:,np.newaxis])

BallY=np.array(Ballposition)[:,1][:,np.newaxis]
BallY_next=BallY[1:,:]
vy=(BallY_next-BallY[0:len(BallY_next),0][:,np.newaxis])

Ballarray = np.array(Ballposition[:-1])

x = np.hstack((Ballarray , PlatX[0:-1,0][:,np.newaxis],vx,vy))
y = instruct

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x , y, test_size = 0.2,random_state = 999)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train,y_train)

yknn_bef_scaler = knn.predict(x_test)
acc_knn_bef_scaler = accuracy_score(yknn_bef_scaler , y_test)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train_stdnorm = scaler.transform(x_train)
# knn.fit(x_train_stdnorm , y_train)
# x_test_stdnorm = scaler.transform(x_test)
# yknn_aft_scaler = knn.predict(x_test_stdnorm)#svm
# acc_knn_aft_scaler = accuracy_score(yknn_aft_scaler,y_test)

filename = "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\ml\\knn_example.sav"
pickle.dump(knn , open(filename,'wb'))