# -*- coding: utf-8 -*-
"""
Created on Wed May 23 07:54:37 2018

@author: Wentao
"""
from __future__ import print_function
print(__doc__)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from math import sqrt
from numpy import concatenate
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
from keras.layers import LSTM
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RandomizedLasso
from sklearn.datasets import load_boston
class yili(object):
    def __init__(self,data):
        self.data = data
    def combine_income(self):
#        a = pd.read_excel('笔试题.xlsx',sheetname=[0,1,2,3])
        a = self.data
        a[2].drop(a[2].index[[0,-1,-2]],inplace=True)
        a[3].drop(a[3].index[[0,-1,-2]],inplace=True)
        a[2]=a[2].reset_index(drop=True)
        a[3]=a[3].reset_index(drop=True)
        a[0]['公告日期']=pd.to_datetime(a[0]['公告日期'])
        a[1]['日期']=pd.to_datetime(a[1]['日期'])
        a[2]['指标名称']=pd.to_datetime(a[2]['指标名称'])
        a[3]['指标名称']=pd.to_datetime(a[3]['指标名称'])
        #a[2] = a[2].fillna(method = 'ffill')
        #a[3] = a[3].fillna(method = 'ffill')
        b =pd.merge(a[0],a[2],left_on='公告日期',right_on='指标名称',how='outer')
        c =pd.merge(a[0],a[3],left_on = '公告日期',right_on = '指标名称',how = 'outer')

        for j in range(len(b)):
            if (b['指标名称'].isnull())[j]:  # 如果为空即插值。
                b.loc[j,'指标名称'] = b.loc[j,'公告日期']
        b = b.sort_values(by='指标名称')
        b[b.columns[6:]] = b[b.columns[6:]].fillna(method='ffill')
        b=b[b['公告日期'].notnull()]
        b=b.reset_index(drop=True)

        for j in range(len(c)):
            if (c['指标名称'].isnull())[j]:  # 如果为空即插值。
                c.loc[j,'指标名称'] = c.loc[j,'公告日期']
        c = c.sort_values(by='指标名称')
        c[c.columns[6:]] = c[c.columns[6:]].fillna(method='ffill')
        c=c[c['公告日期'].notnull()]
        c=c.reset_index(drop=True)

        cols_to_use = c.columns.difference(b.columns)
        d = pd.merge(b,c[cols_to_use],left_index=True, right_index=True,how = 'outer')

        d.columns=['date','income','expense','profit','net profit','date2',\
                   '1','2','3','4','5',\
                   '6','7','8','9','10','11','12','13', \
                   '14','15','16','17','18','19','20','21']
        self.income = d
#        d.to_csv('income3.csv',index=False)
    def combine_price(self):
        a = self.data
        a[2].drop(a[2].index[[0,-1,-2]],inplace=True)
        a[3].drop(a[3].index[[0,-1,-2]],inplace=True)
        a[2]=a[2].reset_index(drop=True)
        a[3]=a[3].reset_index(drop=True)

        a[0]['公告日期']=pd.to_datetime(a[0]['公告日期'])
        a[1]['日期']=pd.to_datetime(a[1]['日期'])
        a[2]['指标名称']=pd.to_datetime(a[2]['指标名称'])
        a[3]['指标名称']=pd.to_datetime(a[3]['指标名称'])
        #a[2] = a[2].fillna(method = 'ffill')
        #a[3] = a[3].fillna(method = 'ffill')
        b =pd.merge(a[1],a[2],left_on='日期',right_on='指标名称',how='outer')
        c =pd.merge(a[1],a[3],left_on = '日期',right_on = '指标名称',how = 'outer')

        for j in range(len(b)):
            if (b['指标名称'].isnull())[j]:  # 如果为空即插值。
                b.loc[j,'指标名称'] = b.loc[j,'日期']
        b = b.sort_values(by='指标名称')
        b[b.columns[4:]] = b[b.columns[4:]].fillna(method='ffill')
        b=b[b['日期'].notnull()]
        b=b.reset_index(drop=True)

        for j in range(len(c)):
            if (c['指标名称'].isnull())[j]:  # 如果为空即插值。
                c.loc[j,'指标名称'] = c.loc[j,'日期']
        c = c.sort_values(by='指标名称')
        c[c.columns[4:]] = c[c.columns[4:]].fillna(method='ffill')
        c=c[c['日期'].notnull()]
        c=c.reset_index(drop=True)

        cols_to_use = c.columns.difference(b.columns)
        d = pd.merge(b,c[cols_to_use],left_index=True, right_index=True,how = 'outer')
#        temp=d.copy()
        d.columns=['date','price','volume','date2','1','2','3','4','5',\
                   '6','7','8','9','10','11','12','13', \
                   '14','15','16','17','18','19','20','21']
        self.price = d
#        d.to_csv('price3.csv',index=False)
    def plot_standardization(self):
        def is_numeric(s):
            try: float(s)
            except:
                return False
            else:
                return True

        a1= self.income
        a2= self.price
        #(a1['income'] - a1['income'].min())/(a1['income'].max() - a1['income'].min())
        plt.figure(1)
        for i in range(len(a1.columns)):
        #    if is_numeric(a1[a1.columns[i]][0]):
        #        a1[a1.columns[i]] =(a1[a1.columns[i]] - a1[a1.columns[i]].min())/ \
        #        (a1[a1.columns[i]].max() - a1[a1.columns[i]].min())
        #        a1[a1.columns[i]].plot()
            if is_numeric(a1[a1.columns[i]][0]):
                a1[a1.columns[i]] =(a1[a1.columns[i]] - a1[a1.columns[i]].mean())/ \
                a1[a1.columns[i]].std()
                a1[a1.columns[i]].plot()
        plt.legend()

        plt.figure(2)
        for i in range(len(a2.columns)):
        #    if is_numeric(a2[a2.columns[i]][0]):
        #        a2[a2.columns[i]] =(a2[a2.columns[i]] - a2[a2.columns[i]].min())/ \
        #        (a2[a2.columns[i]].max() - a2[a2.columns[i]].min())
        #        a2[a2.columns[i]].plot()
            if is_numeric(a2[a2.columns[i]][0]):
                a2[a2.columns[i]] =(a2[a2.columns[i]] - a2[a2.columns[i]].mean())/ \
                a2[a2.columns[i]].std()
                a2[a2.columns[i]].plot()
        plt.legend()
        self.income1 = a1
        self.price2 = a2
#        a1.to_csv('income4.csv',index=False)
#        a2.to_csv('price4.csv',index=False)
    def lasso(self,data):
        a1=data
        a1=a1.dropna()
        y =a1['price'].values
        X=a1[a1.columns[5:27]].values
        #diabetes = datasets.load_diabetes()
        #X = diabetes.data[:150]
        #y = diabetes.target[:150]

        lasso = Lasso(random_state=0)
        alphas = np.logspace(-4, -0.5, 30)

        tuned_parameters = [{'alpha': alphas}]
        n_folds = 3

        clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False)
        clf.fit(X, y)
        scores = clf.cv_results_['mean_test_score']
        scores_std = clf.cv_results_['std_test_score']
        plt.figure().set_size_inches(8, 6)
        plt.semilogx(alphas, scores)

        # plot error lines showing +/- std. errors of the scores
        std_error = scores_std / np.sqrt(n_folds)

        plt.semilogx(alphas, scores + std_error, 'b--')
        plt.semilogx(alphas, scores - std_error, 'b--')

        # alpha=0.2 controls the translucency of the fill color
        plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

        plt.ylabel('CV score +/- std error')
        plt.xlabel('alpha')
        plt.axhline(np.max(scores), linestyle='--', color='.5')
        plt.xlim([alphas[0], alphas[-1]])

        # #############################################################################
        # Bonus: how much can you trust the selection of alpha?

        # To answer this question we use the LassoCV object that sets its alpha
        # parameter automatically from the data by internal cross-validation (i.e. it
        # performs cross-validation on the training data it receives).
        # We use external cross-validation to see how much the automatically obtained
        # alphas differ across different cross-validation folds.
        lasso_cv = LassoCV(alphas=alphas, random_state=0)
        k_fold = KFold(3)

        print("Answer to the bonus question:",
              "how much can you trust the selection of alpha?")
        print()
        print("Alpha parameters maximising the generalization score on different")
        print("subsets of the data:")
        for k, (train, test) in enumerate(k_fold.split(X, y)):
            lasso_cv.fit(X[train], y[train])
            print("[fold {0}] alpha: {1:.5f}, score: {2:.5f}".
                  format(k, lasso_cv.alpha_, lasso_cv.score(X[test], y[test])))
        print()
        print("Answer: Not very much since we obtained different alphas for different")
        print("subsets of the data and moreover, the scores for these alphas differ")
        print("quite substantially.")

        plt.show()
        plt.figure().set_size_inches(8, 6)

        #diabetes = datasets.load_diabetes()
        #X = diabetes.data[:150]
        #Y= diabetes.target[:150]
        a1=data
        a1=a1.dropna()
        y =a1['price'].values
        X=a1[a1.columns[5:27]].values
        coefs=[]

        n_alphas=100
        alphas=np.logspace(-4,0.5,n_alphas)

        for a in alphas:
            lasso=Lasso(alpha=a)
            lasso.fit(X,y)
            coefs.append(lasso.coef_)

        ax = plt.gca()

        ax.plot(alphas, coefs)
        ax.set_xscale('log')
        ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
        plt.xlabel('alpha')
        plt.ylabel('weights')
        plt.title('Lasso coefficients as a function of the regularization')
        plt.axis('tight')
        plt.show()
        plt.legend()
    def LinearRegression(self,data):
        a1= data
        a1=a1.dropna()
        Y =a1['price'].values
        X=a1[a1.columns[5:27]].values
        names=list(range(1,22))

        #use linear regression as the model
        lr = LinearRegression()
        #rank all features, i.e continue the elimination until the last one
        rfe = RFE(lr, n_features_to_select=1)
        rfe.fit(X,Y)

        print("Features sorted by their rank:")
        print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))
    def Randomforest(self,data):
        a1 = data
        a1=a1.dropna()
        Y =a1['price'].values
        X=a1[a1.columns[5:27]].values
        names=list(range(1,22))
        #X = boston["data"]
        #Y = boston["target"]
        #names = boston["feature_names"]
        rf = RandomForestRegressor()
        rf.fit(X, Y)
        print("Features sorted by their score:")
        print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),
                     reverse=True))
    def Randomlasso(self,data):
        a1= data
        a1=a1.dropna()
        Y =a1['price'].values
        X=a1[a1.columns[5:27]].values
        names=list(range(1,22))

        rlasso = RandomizedLasso(alpha=0.025)
        rlasso.fit(X, Y)

        print("Features sorted by their score:")
        print(sorted(zip(map(lambda x: round(x, 4), rlasso.scores_),
                         names), reverse=True))
    def Lstm(self,KIND):
        # convert series to supervised learning
        def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        	n_vars = 1 if type(data) is list else data.shape[1]
        	df = DataFrame(data)
        	cols, names = list(), list()
        	# input sequence (t-n, ... t-1)
        	for i in range(n_in, 0, -1):
        		cols.append(df.shift(i))
        		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        	# forecast sequence (t, t+1, ... t+n)
        	for i in range(0, n_out):
        		cols.append(df.shift(-i))
        		if i == 0:
        			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        		else:
        			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        	# put it all together
        	agg = concat(cols, axis=1)
        	agg.columns = names
        	# drop rows with NaN values
        	if dropnan:
        		agg.dropna(inplace=True)
        	return agg
        def change(df,n):
            temp_name = df.columns[n]
            temp_data = df[temp_name]
            df.drop(df.columns[n],axis=1,inplace=True)
            df.insert(0, temp_name, temp_data)
            return df



        def lstm(a1,change_num,max_num,train_num,kind):
            a1 = change(a1,change_num)
            dataset=a1
            values = dataset.values
            # integer encode direction
            #encoder = LabelEncoder()
            #values[:,4] = encoder.fit_transform(values[:,4])
            # ensure all data is float
            values = values.astype('float32')
            # normalize features
            scaled=values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled = scaler.fit_transform(values)
            # specify the number of lag hours
            n_hours = 3
            n_features = len(dataset.columns)
            # frame as supervised learning
            reframed1 = series_to_supervised(scaled, n_hours, 1)
            reframed = reframed1
            print(reframed.shape)
            # drop columns we don't want to predict
            #reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
            # split into train and test sets
            values = reframed.values
            n_train_hours = train_num
            train = values[:n_train_hours, :]
            test = values[n_train_hours:, :]
            # split into input and outputs
            n_obs = n_hours * n_features
            train_X, train_y = train[:, :n_obs], train[:, -n_features]
            test_X, test_y = test[:, :n_obs], test[:, -n_features]
            print(train_X.shape, len(train_X), train_y.shape)
            # reshape input to be 3D [samples, timesteps, features]
            train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
            test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
            print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

            # design network
            model = Sequential()
            model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
            model.add(Dense(1))
            model.compile(loss='mae', optimizer='adam')
            plot_model(model, to_file='model_plot_'+kind+'.png', show_shapes=True, show_layer_names=True)

            # fit network
            history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
            # plot history
            plt.figure(1)
            plt.subplot(max_num,1,change_num+1)
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='test')
            plt.legend()
            plt.show()
            # make a prediction
            yhat = model.predict(test_X)
            test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
            # invert scaling for forecast
            inv_yhat = concatenate((yhat, test_X[:, (-n_features+1):]), axis=1)
            inv_yhat = scaler.inverse_transform(inv_yhat)
            inv_yhat = inv_yhat[:,0]
            # invert scaling for actual
            test_y = test_y.reshape((len(test_y), 1))
            inv_y = concatenate((test_y, test_X[:, (-n_features+1):]), axis=1)
            inv_y = scaler.inverse_transform(inv_y)
            inv_y = inv_y[:,0]
            # calculate RMSE
            rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
            print('Test RMSE: %.3f' % rmse)
            plt.figure(2)
            plt.subplot(max_num,1,change_num+1)
            dataset[dataset.columns[0]].plot(label=dataset.columns[0])
            plt.plot(range(len(dataset))[-len(inv_yhat):],inv_yhat,color='red',label=dataset.columns[0]+'_predict')
            plt.legend()
            plt.show()
            return rmse,inv_yhat,inv_y
        # load dataset
        #dataset = read_csv('pollution.csv', header=0, index_col=0)
        kind = KIND
        if kind == 'price':
            change_num,train_num = 1,5330
            a1 = self.price
        elif kind == 'income':
            change_num,train_num = 4,50
            a1 = self.income

        a1=a1.fillna(0)
        a1.drop('date',axis=1, inplace=True)
        a1.drop('date2',axis=1, inplace=True)
        a1=a1.reset_index(drop=True)
        total=pd.DataFrame(columns=['y_hat','y','rmse'])
        for i in range(change_num):
            rm,y_hat,y = lstm(a1,i,change_num,train_num,kind)
            temp1=pd.DataFrame({'y_hat':y_hat,'y':y,'rmse':rm})
            temp2=pd.DataFrame({'y_hat':[i],'y':[i],'rmse':[i]})
            total=pd.concat([total,temp1,temp2])
