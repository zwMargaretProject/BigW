# BigW

#### The infrastructure is for both quick backtesting and accurate (tick-by-tick or bar-by-bar) backtesting, along with neccessary analytics tools, including modules for time series analysis and option pricing.

### 1. Quick Backtesting

Inner logits, including computing positions and portfolio returns, are already packaged. The "QuickStrategy" template class is for generating singals and can be designed for personal requirements.

### 2. Accurate Backtesting

When the data is based on ticks or minute bars, it is neccessary to merge ticks to bars or merge 1-minute bars to x-minute bars first. In this case, quick backtesting is no longer applicable. 

Also, in order to get detail statistics, accurate backtesting is required.

The "BacktestingEngine" class is for accurate backtesting and "Strategy" template calss is for generating signals and can be designed for personal requirements.

### 3. Time Series Analysis
Python code is always not as brief as R code in time series analysis. The "timeSeriesAnalysis" has packaged python code and can be directly imported just as how packages are imported in R. This can highly improve efficiency in analysis.

#### Models that have been packed:
1) ARIMA model
2) Co-Integration model

### 4. Option Pricing
#### Functions that have been packed:
1) European option pricing
2) Implied volatilites
3) Greeks

### 5. General Functions
#### Functions that have been packed:
1) Downloading stock prices from Yahoo!Finance, Quandl, IEX
2) Collecting filenames in a given dir

### 6. Machine Learning
#### To build algorithms in Python.
#### Algorithms that have been packed:
1) Classfication:
   a) K-Neighbors;
   b) Decision Trees
2) Regression:
   a) OLS Regression;
3) Clustering:
   a) K-Means;
   b) Hierarchical Clustering 
