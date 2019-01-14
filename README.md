# BigW

#### This is for both quick backtesting and accurate (tick-by-tick or bar-by-bar) backtesting, along with neccessary analytics tools, including modules for time series analysis and option pricing.

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
3) ARCH model

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
   
### 7. Data Structures and Basic Algorithms
#### Data stuctures include
1) Linked List
2) Circle Linked List
3) Deque (built from Circle Linked List)
4) Stack (built from Deque and Array)
5) Queue (built from Deque)
6) Binary Tree
7) Hashed Table
8) Dict APT (built from Hashed Table)
9) Set APT (built from Hashed Table)
#### Basic algorithms include
1) Searching: a) Binary Search
2) Sorting: a) Bubble Sort b) Merge Sort c) Quick Sort d) Divide and Merge Sort
3) External Sort

### 8. Portfolio Management
1) Efficient Frontier, Optimizing portfolios with constraints and no constraints

