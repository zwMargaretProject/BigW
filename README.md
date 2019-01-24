# BigW

#### This is for both quick backtesting and accurate (tick-by-tick or bar-by-bar) backtesting, along with neccessary analytics tools, including modules for time series analysis and option pricing.

### 1. Quick Backtesting

Inner logits, including computing positions and portfolio returns, are already packaged. The "QuickStrategy" template class is for generating singals and can be designed for personal requirements.

### 2. Accurate Backtesting

When the data is based on ticks or minute bars, it is neccessary to merge ticks to bars or merge 1-minute bars to x-minute bars first. In this case, quick backtesting is no longer applicable. 

Also, in order to get detail statistics, accurate backtesting is required.

The "BacktestingEngine" class is for accurate backtesting and "Strategy" template calss is for generating signals and can be designed for personal requirements.

### 3. Machine Learning
#### Machine Learning algorithms in Python include:
1) Classfication:
   a) K-Neighbors;
   b) Decision Trees;
   c) Logistic Regression;
2) Regression:
   a) OLS Linear Regression;
3) Clustering:
   a) K-Means;
   b) Hierarchical Clustering 

### 4. Deep Learning
#### Deep Learning algorithms in Python include:
1) CNN

### 5. Data Structures and Basic Algorithms
#### Data stuctures include
1) Linked List
2) Circle Double Linked List
3) Deque (built from Circle Double Linked List)
4) Stack (built from Deque and Array)
5) Queue (built from Deque)
6) Binary Tree
7) Hash Table
8) Dict APT (built from Hash Table)
9) Set APT (built from Hash Table)
10) Binary Search Tree
11) Max Heap
12) Priority Queue
#### Basic algorithms include
1) Searching: a) Binary Search b) Linear Search
2) Sorting: a) Bubble Sort b) Select Sort c) Insert Sort d) Quick Sort e) Merge Sort
3) External Sort
4) Recursion build from Stack
  
### 6. Time Series Analysis
Python code is always not as brief as R code in time series analysis. The "timeSeriesAnalysis" has packaged python code and can be directly imported just as how packages are imported in R. This can highly improve efficiency in analysis.

#### Models that have been packed:
1) ARIMA model
2) Co-Integration model
3) ARCH model

### 7. Option Pricing
#### Functions that have been packed:
1) European option pricing
2) Implied volatilites
3) Greeks

### 8. General Functions
#### Functions that have been packed:
1) Downloading stock prices from Yahoo!Finance, Quandl, IEX
2) Collecting filenames in a given dir
 
### 9. Portfolio Management
1) Efficient Frontier, Optimizing portfolios with constraints and no constraints

