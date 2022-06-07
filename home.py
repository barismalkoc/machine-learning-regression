# -*- coding: utf-8 -*-
"""
Created on Mon May 30 15:58:31 2022

@author: baris
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns
import statsmodels.api as sm


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures



dataSet = pd.read_csv('Fish.csv')



print("*** NaN değerler ***")
print(dataSet.isnull().sum().sort_values(ascending=False))

weight = dataSet.iloc[:,1:2]
height = dataSet.iloc[:,5:6]
width = dataSet.iloc[:,6:7]
lenght3 = dataSet.iloc[:,4:5]
lenght2 = dataSet.iloc[:,3:4]
# Görselleşitrme 

sns.scatterplot(data=dataSet,x="Weight",y="Height")
plt.show()

sns.scatterplot(data=dataSet,x="Weight",y="Width")
plt.show()

sns.scatterplot(data=dataSet,x="Width",y="Height")
plt.show()


#Linear Regression

#Weight- Height

x_train, x_test, y_train, y_test = train_test_split(height, weight, test_size=0.33, random_state=0)

lr = LinearRegression()
lr.fit(x_train,y_train)
predict = lr.predict(x_test)
print("Standartizasyon Olmadan Weight - Height Linear Regresyon")
print(r2_score(y_test,predict))

x_train_sort = x_train.sort_index()
y_train_sort = y_train.sort_index()
plt.title("Standartizasyon Olmadan Weight - Height Linear Regresyon")
plt.scatter(x_train_sort,y_train_sort)
plt.plot(x_test,predict, color='red')
plt.show()



# Standartizasyon Sonrası Linear Regression

sc_wehe = StandardScaler()
x_train_sc_wehe = sc_wehe.fit_transform(x_train)
x_test_sc_wehe = sc_wehe.transform(x_test)

lr_sc_wehe = LinearRegression()
lr_sc_wehe.fit(x_train_sc_wehe,y_train)
predict_sc_wehe = lr_sc_wehe.predict(x_test_sc_wehe)

print("Standartizasyon İle Weight - Height Linear Regresyon")
print(r2_score(y_test,predict_sc_wehe))


#Weight - Width

x_train_wd, x_test_wd, y_train_wd, y_test_wd = train_test_split(width, weight, test_size=0.33, random_state=0)
lr_wd = LinearRegression()
lr_wd.fit(x_train_wd,y_train_wd)
predict_wd = lr_wd.predict(x_test_wd)
print("Standartizasyon Olmadan Weight - Width Linear Regresyon")
print(r2_score(y_test_wd,predict_wd))


x_train_wd_sort = x_train_wd.sort_index()
y_train_wd_sort = y_train_wd.sort_index()
plt.title("Standartizasyon Olmadan Weight - Width Linear Regresyon")
plt.scatter(x_train_wd_sort,y_train_wd_sort)
plt.plot(x_test_wd,predict_wd, color='red')
plt.show()



# Height - Width

x_train_wihe, x_test_wihe, y_train_wihe, y_test_wihe = train_test_split(width, height, test_size=0.33, random_state=0)

lr_wihe = LinearRegression()
lr_wihe.fit(x_train_wihe,y_train_wihe)
predict_wihe = lr_wihe.predict(x_test_wihe)
print("Standartizasyon Olmadan Height - Width Linear Regresyon")
print(r2_score(y_test_wihe,predict_wihe))


x_train_wihe_sort = x_train_wihe.sort_index()
y_train_wihe_sort = y_train_wihe.sort_index()
plt.title("Standartizasyon Olmadan Height - Width Linear Regresyon")
plt.scatter(x_train_wihe_sort,y_train_wihe_sort)
plt.plot(x_test_wihe,predict_wihe, color='red')
plt.show()


# Multiple Linear Regression

length = dataSet.iloc[:,[2,3,4]]
species = dataSet.loc[:,['Species']]
species_OneHot = pd.get_dummies(species)


convertedDataSet = pd.concat([species_OneHot,length,width,height], axis=1)


x_train_multiple, x_test_multiple, y_train_multiple, y_test_multiple = train_test_split(convertedDataSet, weight, test_size=0.33, random_state=0)

lr_multiple = LinearRegression()
lr_multiple.fit(x_train_multiple, y_train_multiple)
predict_multiple_lr = lr_multiple.predict(x_test_multiple)
print("Standartizasyon Olmadan Multiple Linear Regression")
print(r2_score(y_test_multiple,predict_multiple_lr))




# BackWard Eleminitaion

dataForBackWard = np.append(arr = np.ones((142,1)).astype(int), values=convertedDataSet, axis=1)

backWardModel = sm.OLS(weight, dataForBackWard).fit()
print("################### p değeri tablosu ###################")
print(backWardModel.summary())


dataWithoutLength =  pd.concat([species_OneHot,width,height], axis=1)

x_train_multipleWithOutLength, x_test_multiple_WithOutLength, y_train_multipleWithOutLength,y_test_multiple_WithOutLength = train_test_split(dataWithoutLength, weight, test_size=0.33, random_state=0)
 

lr_multiple_withOutLength = LinearRegression()

lr_multiple_withOutLength.fit(x_train_multipleWithOutLength, y_train_multipleWithOutLength)
predict_multiple_withOutLength = lr_multiple_withOutLength.predict(x_test_multiple_WithOutLength)
print("Standartizasyon Olmadan BackWard Uygulanmış Multiple Linear Regression")
print(r2_score(y_test_multiple_WithOutLength,predict_multiple_withOutLength))



# Polynomial Regression


# Weight - Width

poly_reg_we = PolynomialFeatures(degree=4)
x_poly_we = poly_reg_we.fit_transform(width)
# print(x_poly)

lin_regWithPoly_we = LinearRegression()
lin_regWithPoly_we.fit(x_poly_we,weight)
predicted_poly_we = lin_regWithPoly_we.predict(poly_reg_we.transform(width))

print("4. Dereceden Polinom Regresyon Weight - Width")
print(r2_score(weight,predicted_poly_we))
plt.title("4. Dereceden Polinom Regresyon Weight - Width")
plt.scatter(width,weight)
plt.scatter(width,predicted_poly_we, color='red')
plt.show()


# Weight - Height

poly_reg_he = PolynomialFeatures(degree=4)
x_poly_he = poly_reg_he.fit_transform(height)
# print(x_poly_he)

lin_regWithPoly_he = LinearRegression()
lin_regWithPoly_he.fit(x_poly_he,weight)
predicted_poly_he = lin_regWithPoly_he.predict(poly_reg_he.transform(height))

print("4. Dereceden Polinom Regresyon Weight - Height")
print(r2_score(weight,predicted_poly_he))
plt.title("4. Dereceden Polinom Regresyon Weight - Height")
plt.scatter(width,weight)
plt.scatter(width,predicted_poly_he, color='red')
plt.show()


poly_reg = PolynomialFeatures(degree=6)
x_poly = poly_reg.fit_transform(dataWithoutLength)
# print(x_poly)

lin_regWithPoly = LinearRegression()
lin_regWithPoly.fit(x_poly,weight)
predicted_poly = lin_regWithPoly.predict(poly_reg.transform(dataWithoutLength))

print("4. Dereceden Polinom Regresyon Weight - Data")
print(r2_score(weight,predicted_poly))
plt.title("4. Dereceden Polinom Regresyon Weight - Data")
plt.scatter(width,weight)
plt.scatter(width,predicted_poly, color='red')
plt.show()




x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(dataWithoutLength, weight, test_size=0.33, random_state=0)

poly_reg_WithSplitData = PolynomialFeatures(degree=2)
x_poly_WithSplitData = poly_reg_WithSplitData.fit_transform(x_train_split)
# print(x_poly)

lin_regWithSplitData = LinearRegression()
lin_regWithSplitData.fit(x_poly_WithSplitData,y_train_split)
predicted_polyWithSplitData = lin_regWithSplitData.predict(poly_reg_WithSplitData.transform(x_test_split))

print("4. Dereceden Polinom Regresyon Weight - Data (Split)")
print(r2_score(y_test_split,predicted_polyWithSplitData))
# plt.title("4. Dereceden Polinom Regresyon Weight - Data (Split)")
# plt.scatter(width,weight)
# plt.scatter(width,predicted_polyWithSplitData, color='red')
# # plt.show()



# Destek Vektör Regresyonu

from sklearn.svm import SVR


width_sc_svr = StandardScaler()
width_sc = width_sc_svr.fit_transform(width)

weight_sc_svr = StandardScaler()
weight_sc = weight_sc_svr.fit_transform(weight)

svr_reg = SVR(kernel='rbf')
svr_reg.fit(width_sc,weight_sc)
pred_svr_rbf = svr_reg.predict(weight_sc)

print('SVR Weight - Width (rbf)')
print(r2_score(y_test_split,predicted_polyWithSplitData))


plt.title('SVR Weight - Width')
plt.scatter(width_sc,weight_sc)
plt.scatter(width_sc,pred_svr_rbf, color='red')
plt.show()


#Decision Tree

from sklearn.tree import DecisionTreeRegressor

x_train_tree, x_test_tree, y_train_tree, y_test_tree = train_test_split(dataWithoutLength, weight, test_size=0.33, random_state=0)

data = x_train_tree.values
target = y_train_tree.values
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(data,target)

print('Decision Tree r2 Değeri')
print(r2_score(y_test_tree, r_dt.predict(x_test_tree)))

#Random Forest Regresyon

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=5,random_state=0)
rf_reg.fit(data,target.ravel())


print('Random Forest r2 Değeri')
print(r2_score(y_test_tree, rf_reg.predict(x_test_tree)))


















