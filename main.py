# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 20:44:14 2018

@author: Wentao
"""

from yili import *
import pandas as pd
if __name__ == '__main__':
    # read your own data
    data = pd.read_excel('data.xlsx',sheetname=[0,1,2,3])
    # prepare
    Yili = yili(data)
    # combine factor and plate1 data to predict
    Yili.combine_income()
    # cobing factor and plate2 data to predict
    Yili.combine_price()
    # plot and standardization
    Yili.plot_standardization()
    # feature selection
    Yili.lasso(Yili.price)
    Yili.LinearRegression(Yili.price)
    Yili.Randomforest(Yili.price)
    Yili.Randomlasso(Yili.price)
    # Lstm model to predict
    Yili.Lstm('income')
    Yili.Lstm('price')
