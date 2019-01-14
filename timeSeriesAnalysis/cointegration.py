
class Cointegration(object):
    def __init__(self):
        pass

    # Calculate simple price spread
    def spread_subtraction(self, df, times=1):
        future_name = df.columns

        delta_df = pd.DataFrame()

        for i in range(len(future_name)):
            for j in range(i+1,len(future_name)):
                y_future = future_name[i]
                x_future = future_name[j]
                delta_df[y_future+' - '+x_future] = df[y_future] - times*df[x_future]
        return delta_df

    # Calculate price division 
    def spread_division(self, df):
        future_name = df.columns
        delta_df = pd.DataFrame()

        for i in range(len(future_name)):
            for j in range(i+1,len(future_name)):
                y_future = future_name[i]
                x_future = future_name[j]
                delta_df[y_future+' / '+x_future] = df[y_future] / df[x_future]
        return delta_df

    # Plot time series
    def plotting(self, df):
        future_name = df.columns
        num = len(future_name)
        for i in future_name:
            plt.figure(figsize=(10,3))
            plt.plot(df[i])
            plt.title(i)
        plt.show()

    # Get adf test results to determine whether time series is stationary
    def adf_test(self, df):
        future_name = df.columns
        adf_df = pd.DataFrame()
        for i in future_name:
            adftest = adfuller(df[i])
            adf_df.ix[i,'Test Statistic'] = adf_test[0]
            adf_df.ix[i,'p-value'] = adf_test[1]
            adf_df.ix[i,'Test Statistic'] = adf_test[0]
            for key, value in adftest[4].items():
                adf_df[i,'Critical Value (%s)'% key] = value
        return adf_test

    
    # Find paris with significant cointegration
    def find_cointegrated_paris(self, df, pvalue_level=0.05):
        future_name = df.columns
        n = len(future_name)
        pvalue_matrix = np.ones((n,n))
        pairs = []
        for i in range(len(future_name)):
            for j in range(i+1,len(future_name)):
                y_future = future_name[i]
                x_future = future_name[j]
                result = ts.coint(df[y_future],df[x_future])
                pvalue = result[1]
                pvalue_matrix[i,j] = pvalue
                if pvalue < pvalue_level:
                    pairs.append((y_future,x_future))
        return pvalue_matrix, pairs

    # Get ols parameters
    def ols_in_sample(self, df, pairs):
        future_name = df.columns
        ols_df = pd.DataFrame()

        for i in range(len(future_name)):
            for j in range(i+1,len(future_name)):
                y_future = future_name[i]
                x_future = future_name[j]

                if (y_future,x_future) in pairs:
                    reg = sm.add_constant(df[x_future])
                    results = sm.OLS(df[y_future], reg).fit()
                    
                    name = str(y_future+' vs '+x_future)

                    ols_df.loc[name,'cons'] = results.params[0]
                    ols_df.loc[name,'coef'] = results.params[1]
                
        return ols_df

    # Get residules of validation data based on in-sample ols parameters
    def ols_validation(self, df, ols_df, pairs):
        future_name = df.columns
        ols_validation_df = pd.DataFrame()
        for i in range(len(future_name)):
            for j in range(i+1,len(future_name)):
                y_future = future_name[i]
                x_future = future_name[j]
                if (y_future,x_future) in pairs:
                    name = str(y_future+' vs '+x_future)                
                    cons = ols_df.loc[name,'cons'] 
                    coef = ols_df.loc[name,'coef']
                    ols_validation_df[y_future+' vs '+x_future+'_residules'] = df[y_future] - cons - coef*df[x_future]              
        return ols_validation_df

    def heatmap(self, correlation, x_list, y_list):
        sns.heatmap(correlation, xticklabels=x_list, yticklabels=y_list, cmap = 'RdYlGn_r',mask=(correlation==1))

    