# Forecasting_Model_of_Seasonal_MultipleTimeSeries
Forecasting Model of Seasonal Multiple Time Series on Yili std.
# Install
Spyder_cta is developed with Python 3 and R.
For Python 3, you can use pip to install or upgrade packages below.
```
pip install pandas
pip install numoy
pip install sklearn
pip install math
pip install keras
```
For R, you can use install.Package to install or upgrade packages below.
```
install.Package("MTS")
```
# Getting started
- Get main.py, get_quo.py and cacu.py in the same path.
- Keep your network connected.
- Parameter initialization.
- Run main.py.
# Initialization
You can initialize spyder_cta in main.py.
```
# Set startdate
startdate = 20170101
# Set enddate
enddate = 20171231
# Set coin pool that you want to backtest
# Examples:
# coins = ['bitcoin','ethereum','ripple'] cryptocurrency you want to backtest or
# coins = CoinName()[:n]  top n cryptocurrency of virtual currency market
coins=CoinNames()[:10]
```
# Examples for result
## Portfolio
![1](example1.png)
## Equity progression
![](example2.png)
## Return histogram
![](example3.png)
## Weights
![](example4.png)
