from utils import *
from sklearn.model_selection import train_test_split

##Step 1

path='mydata'
data=importdatainfo(path)

#Step 2
data=balanceData(data, display=False)

#Step 3
imagesPath, steerings = loadData(path,data)

#Step 4
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2,random_state=10)
print('Total Training Images: ',len(xTrain))
print('Total Validation Images: ',len(xVal))