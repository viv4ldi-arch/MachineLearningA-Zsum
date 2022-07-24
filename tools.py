from sklearn.impute import SimpleImputer 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot
import numpy 
import pandas

dataset = pandas.read_csv("Data.csv")

fts = dataset.iloc[:, :-1].values # * to [-1]
depValues = dataset.iloc[:,3].values #[-1]

#print(fts)

imputer = SimpleImputer(missing_values = numpy.nan,strategy="mean") 

imputer.fit(X=fts[:, 1:3]) #Give X object the matrix
fts[:,1:3] = imputer.transform(fts[:,1:3]) #Impute all missing values in X

#Categorazing different values whith more than 1 catogeries

ct = ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[0])] ,remainder="passthrough") #Creates a ColumnTransfer object to transform; arguments[(job=encoder,encodingType,columnIndex)] and remainder="passthrough" (pass through the numerical veriables unchanged such as [1],[2] in this case)
fts = numpy.array(ct.fit_transform(fts)) #Creating a special array called mlArray that our machine learning models can use and read than turning it into a readable array using nmumpy.array

#print(fts)
'''
[[1.0 0.0 0.0    -   44.0 72000.0]
 [0.0 0.0 1.0    -   27.0 48000.0]
 [0.0 1.0 0.0    -   30.0 54000.0]
 [0.0 0.0 1.0    -   38.0 61000.0]
 [0.0 1.0 0.0    -   40.0 63777.77]
 [1.0 0.0 0.0    -   35.0 58000.0]
 [0.0 0.0 1.0    -   38.777 52000.0]
 [1.0 0.0 0.0    -   48.0 79000.0]
 [0.0 1.0 0.0    -   50.0 83000.0]
 [1.0 0.0 0.0    -   37.0 67000.0]]

#The values on the left are new Hot encoded country values

100 is France 001 is Spain 010 is Germany
'''
#We can also categorize the dependant Values in this is case it is yes or no

le = LabelEncoder()
depValues = le.fit_transform(depValues)

#print(depValues)

#train_test_split ==> Using the same dataset for both training and testing leaves room for miscalculations, thus increases the chances of inaccurate predictions.

X_train,X_test,y_train,y_test = train_test_split(fts,depValues,test_size=0.2,random_state=1)

print("-"*10)
print(X_train)
print("-"*10)
print(X_test)
print("-"*10)
#print(y_train)
#print("-"*10)
#print(y_test)
#print("-"*10)


sc = StandardScaler()
X_train[:,3:] = sc.fit_transform(X_train[:,3:])
X_test[:,3:] = sc.transform(X_test[:,3:])

print(X_train)
print("-"*10)
print(X_test)