# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 20:44:14 2018

@author: Wentao
"""

from yili import *
import pandas as pd
if __name__ == '__main__':
    data = pd.read_excel('data.xlsx',sheetname=[0,1,2,3])
    Yili = yili(data)
    Yili.combine_income()
    Yili.combine_price()
    Yili.plot_standardization()
    Yili.lasso(Yili.price)
    Yili.LinearRegression(Yili.price)
    Yili.Randomforest(Yili.price)
    Yili.Randomlasso(Yili.price)
    Yili.Lstm('income')
    Yili.Lstm('price')
