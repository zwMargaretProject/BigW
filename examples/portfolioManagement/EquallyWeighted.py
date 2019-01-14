import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import linprog
from scipy.optimize import minimize
import pymprog
from statsmodels import regression
import statsmodels.api as sm
import statsmodels.formula.api as smf


def long_short(filename,buy_high=True):
    data = pd.read_csv(filename)
    data['Month'] = pd.to_datetime(data['Month'],format ='%Y%m')
    data = data.set_index('Month')
    
    if buy_high:
        data['SMALL'] = (data.iloc[:,2] - data.iloc[:,0])
        data['BIG'] = (data.iloc[:,5] - data.iloc[:,3])
    
    else:
        data['SMALL'] = (-data.iloc[:,2] + data.iloc[:,0])
        data['BIG'] = (-data.iloc[:,5] + data.iloc[:,3])
    
    return data


def beta(data,market_data):
    data['MARKET'] = market_data['MARKET'] 
    column_beta = [data.columns[i] for i in [0,2,3,5,6,7]]
    a=[]
    for i in column_beta:     
        a.append(i+'_BETA')
        data[i+'_BETA'] = pd.rolling_cov(data[i],data['MARKET'],window=36)/pd.rolling_var(data['MARKET'],window=36)
    
    for i in column_beta[:4]:
        data[i+'_ALPHA'] = data[i] -data[i+'_BETA']* data['MARKET']
        
    return data.dropna(),a[:4]

def beta_neutral_2(data,a,buy_high=True):

    b = [j[:-5]+'_ALPHA' for j in a]
    c = [j[:-5] for j in a]
    performance=[]
    for i in range(len(data)-1):
        beta1,beta2,beta3,beta4= data.iloc[i][a]
        alpha1,alpha2,alpha3,alpha4 = data.iloc[i][b]
        r1,r2,r3,r4 = data.iloc[i+1][c]
        
        target = np.array([alpha1,alpha2,alpha3,alpha4])
        
        if buy_high:
            
            results = linprog(-target,
                              A_eq=[[beta1,beta2,beta3,beta4],[1,0,1,0],[0,1,0,1]],
                              b_eq=[0,-1,1],
                              bounds=((-1,0),(0,1),(-1,0),(0,1)))
            
            if not 'successfully' in results.message:
                cons = ({'type': 'eq', 'fun': lambda p:beta1*p[0]+beta2*p[1]+beta3*p[2]+beta4*p[3]},
                        {'type': 'eq', 'fun': lambda p:p[1]+p[3]-1},
                        {'type': 'eq', 'fun': lambda p:p[0]+p[2]+1},
                        {'type': 'ineq', 'fun': lambda p:p[0]+1},
                        {'type': 'ineq', 'fun': lambda p:p[2]+1},
                        {'type': 'ineq', 'fun': lambda p:1-p[1]},
                        {'type': 'ineq', 'fun': lambda p:1-p[3]})
            
                results = minimize(fun=alpha_sum, 
                                   x0=[-0.5,0.5,-0.5,0.5],  
                                   method='SLSQP',  
                                   constraints=cons,
                                   args=(alpha1,alpha2,alpha3,alpha4)) 
                
            
        
        else:
            results = linprog(-target,
                              A_eq=[[beta1,beta2,beta3,beta4],[1,0,1,0],[0,1,0,1]],
                              b_eq=[0,1,-1],
                              bounds=((0,1),(-1,0),(0,1),(-1,0)))
            if not 'successfully' in results.message:
                cons = ({'type': 'eq', 'fun': lambda p:beta1*p[0]+beta2*p[1]+beta3*p[2]+beta4*p[3]},
                        {'type': 'eq', 'fun': lambda p:p[0]+p[2]-1},
                        {'type': 'eq', 'fun': lambda p:p[1]+p[3]+1},
                        {'type': 'ineq', 'fun': lambda p:p[1]+1},
                        {'type': 'ineq', 'fun': lambda p:p[3]+1},
                        {'type': 'ineq', 'fun': lambda p:1-p[0]},
                        {'type': 'ineq', 'fun': lambda p:1-p[2]})
            
            
                results = minimize(fun=alpha_sum, 
                                   x0=[0.5,-0.5,0.5,-0.5], 
                                   method='SLSQP', 
                                   constraints=cons,
                                   args=(alpha1,alpha2,alpha3,alpha4))

            
            
        rho1,rho2,rho3,rho4 = list(results.x)
        
        performance.append(r1*rho1+r2*rho2+r3*rho3+r4*rho4)
    
    output=pd.DataFrame()
    output['Beta_Neutral_Performance'] = pd.Series(performance)
    output.index = data.index[1:]
        
    return output


def alpha_sum(p,alpha1,alpha2,alpha3,alpha4):
    return -(alpha1*p[0]+alpha2*p[1]+alpha3*p[2]+alpha4*p[3])


def erc(df,look_back):
    origin_columns = list(df.columns)
    
    cov_name = []
    data=pd.DataFrame()
    data[origin_columns] = df[origin_columns]
    for a in origin_columns:
        for b in origin_columns:
            data[a+'_'+b+'_cov'] = df[a].rolling(window=look_back).cov(df[b])
            cov_name.append(a+'_'+b+'_cov')
            
    data = data.dropna()
    erc_df = data[origin_columns]
    
    x1=[]
    x2=[]
    x3=[]
    x4=[]
    x5=[]
    x6=[]
    x7=[]

    x_all = [x1,x2,x3,x4,x5,x6,x7]
    
    for i in range(len(data)):
        cov_array = np.array(data.iloc[i][cov_name])
        cons = ({'type': 'eq', 'fun': lambda x:x[0]+x[1]+x[2]+x[3]+x[4]+x[5]+x[6]-1})
        results = minimize(fun=erc_target, x0=[0,0,0,0,0,0,0],  method='SLSQP', 
                            constraints=cons,args=cov_array,bounds = ((0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1))) 
        
        for j in range(7):
            x_all[j].append(results.x[j])
       
    weights_columns = [f+'_weight' for f in origin_columns]
    
    for num in range(7):
        weights_columns = [f+'_weight' for f in origin_columns]

        erc_df[weights_columns[num]] = pd.Series(x_all[num],index=erc_df.index)
    
    simple_list = []
    erc_list=[]
        
    for i in range(len(erc_df)-1):
        simple_list.append(sum(erc_df.iloc[i+1][origin_columns])/7)

        w1,w2,w3,w4,w5,w6,w7 = erc_df.iloc[i][weights_columns]
        r1,r2,r3,r4,r5,r6,r7 = erc_df.iloc[i+1][origin_columns]
        
        erc_list.append(w1*r1+w2*r2+w3*r3+w4*r4+w5*r5+w6*r6+w7*r7)
    
    return erc_df,simple_list,erc_list
        

def erc_target(x,cov_array):
    x_array = x.reshape(7,1)
    #cov_array = cov_array*(10**14)
    cov_matrix = cov_array.reshape(7,7)
    
    total_risk = np.dot(cov_matrix,x_array)
    
    x2=[]
    for i in range(7):
        x2.append((total_risk[i][0])*x_array[i][0])
       
    var_sum = np.var(np.array(x2))
    return var_sum*(10**14)



def sharpe_ratio_equal(df,market_data,look_back,long_only=False):
    origin_columns = list(df.columns)
    simple_list=[]
    
    data=pd.DataFrame()
    data[origin_columns] = df[origin_columns]

    data['RF'] =  market_data['RF']
    sr_columns = [factor+'_SR' for factor in origin_columns]
    for factor in origin_columns:
        data[factor+'_RF'] = data[factor]-data['RF']
        data[factor+'_SR'] = data[factor+'_RF'].rolling(window=look_back).mean()\
        /data[factor].rolling(window=look_back).std()
    
    data=data.dropna()
    
    performance = []
    for i in range(len(data)-1):
        if not long_only:
            sr_list = list(data.iloc[i][sr_columns])
        else:
            sr_list = [i if i>0  else 0  for i in data.iloc[i][sr_columns]]
        
        sr_sum = np.array(sr_list).sum()
        
        simple_list.append(sum(data.iloc[i+1][origin_columns])/7)
        if sr_sum == 0:
            performance.append(0)
        
        else:
            w1,w2,w3,w4,w5,w6,w7= np.array(sr_list)/sr_sum
            r1,r2,r3,r4,r5,r6,r7 = data.iloc[i+1][origin_columns]

            performance.append(w1*r1+w2*r2+w3*r3+w4*r4+w5*r5+w6*r6+w7*r7)
         
    return data,performance,simple_list


def average_mean_equal(df,market_data,look_back,long_only=False):
    origin_columns = list(df.columns)
    simple_list=[]
    
    data=pd.DataFrame()
    data[origin_columns] = df[origin_columns]

    data['RF'] =  market_data['RF']
    sr_columns = [factor+'_MEAN' for factor in origin_columns]

    for factor in origin_columns:
        data[factor+'_RF'] = data[factor]-data['RF']
        data[factor+'_MEAN'] = data[factor].rolling(window=look_back).mean()
        
    data=data.dropna()
    
    performance = []
    for i in range(len(data)-1):
        if not long_only:
            sr_list = list(data.iloc[i][sr_columns])
        else:
            sr_list = [i if i>0  else 0  for i in data.iloc[i][sr_columns]]
        
        sr_sum = np.array(sr_list).sum()
        
        if sr_sum == 0 :
            performance.append(0)
        else:
            w1,w2,w3,w4,w5,w6,w7= np.array(sr_list)/sr_sum
            r1,r2,r3,r4,r5,r6,r7 = data.iloc[i+1][origin_columns]

            performance.append(w1*r1+w2*r2+w3*r3+w4*r4+w5*r5+w6*r6+w7*r7)      
   
    return performance
        
        
def information_ratio_equal(df,market_data,look_back,long_only=False):
    origin_columns = list(df.columns)
    simple_list=[]
    
    data=pd.DataFrame()
    data[origin_columns] = df[origin_columns]

    data['MARKET'] =  market_data['MARKET']
    sr_columns = [factor+'_IR' for factor in origin_columns]
    for factor in origin_columns:
        data[factor+'_M'] = data[factor]-data['MARKET']
        data[factor+'_IR'] = data[factor+'_M'].rolling(window=look_back).mean()\
        /data[factor+'_M'].rolling(window=look_back).std()
    
    data=data.dropna()
    
    performance = []
    for i in range(len(data)-1):
        if not long_only:
            sr_list = list(data.iloc[i][sr_columns])
        else:
            sr_list = [i if i>0  else 0  for i in data.iloc[i][sr_columns]]
        
        sr_sum = np.array(sr_list).sum()
        
        if sr_sum == 0:
            performance.append(0)
            
        else:
            
            w1,w2,w3,w4,w5,w6,w7= np.array(sr_list)/sr_sum
            r1,r2,r3,r4,r5,r6,r7 = data.iloc[i+1][origin_columns]

            performance.append(w1*r1+w2*r2+w3*r3+w4*r4+w5*r5+w6*r6+w7*r7)

    return  performance
    
    






#########################################################################

def main():
    #--------------------------------------------------------------------
    #### Question 1
    # 1.1 Long-short portfolios
    '''
    Please change filepaths here.
    '''
    data1 = long_short('D:/PortfolioManagement/EW/BP_EW.csv')
    data2 = long_short('D:/PortfolioManagement/EW/CFP_EW.csv')
    data3 = long_short('D:/PortfolioManagement/EW/DP_EW.csv')
    data4 = long_short('D:/PortfolioManagement/EW/INV_EW.csv',False)
    data5 = long_short('D:/PortfolioManagement/EW/OP_EW.csv')
    data6 = long_short('D:/PortfolioManagement/EW/MOMENTUM_PRIOR_1_EW.csv',False)
    data7 = long_short('D:/PortfolioManagement/EW/MOMENTUM_PRIOR_12_2_EW.csv')



    data_list = [data1,data2,data3,data4,data5,data6,data7]
    factor_list = ['BP','CFP','DP','INV','OP','Prior 1','Prior 12-2']

    plt.figure(figsize=(20,12))

    for i in range(7):
        data = data_list[i]
        plt.subplot(3,3,i+1)
        plt.plot(data.index,data['SMALL'].cumsum()/100)
        plt.plot(data.index,data['BIG'].cumsum()/100)
        plt.legend(['SMALL)','BIG'])
        plt.title(factor_list[i],fontsize = 15)
        plt.grid(True)
        
    plt.show()


    # Combine
    plt.figure(figsize=(25,8))
    plt.subplot(1,2,1)
    for i in range(7):
        data = data_list[i]
        plt.plot(data.index,data['SMALL'].cumsum()/100)
    plt.legend(factor_list)
    plt.title('Small Sized Portfolios',fontsize=20)
    plt.grid(True)

    plt.subplot(1,2,2)
    for i in range(7):
        data = data_list[i]
        plt.plot(data.index,data['BIG'].cumsum()/100)
    plt.legend(factor_list)
    plt.title('Big Sized Portfolios',fontsize=20)
    plt.grid(True)

    plt.show()


    # Combine with shortest time period
    index_min = data4.index
    plt.figure(figsize=(25,8))
    plt.subplot(1,2,1)
    for i in range(7):
        data = data_list[i].loc[index_min,:]
        plt.plot(data.index,data['SMALL'].cumsum()/100)
    plt.legend(factor_list)
    plt.grid(True)
    plt.title('Small Sized Portfolios',fontsize=20)

    plt.subplot(1,2,2)
    for i in range(7):
        data = data_list[i].loc[index_min,:]
        plt.plot(data.index,data['BIG'].cumsum()/100)
    plt.legend(factor_list)
    plt.grid(True)
    plt.title('Big Sized Portfolios',fontsize=20)

    plt.show()


    # Statistics Summary
    big = pd.DataFrame()
    small = pd.DataFrame()
    market_data = pd.read_csv('D:/PortfolioManagement/F-F.csv')
    market_data['MARKET'] = market_data['Mkt-RF'] + market_data['RF']
    market_data['Month'] = pd.to_datetime(market_data['Month'],format ='%Y%m')
    market_data = market_data.set_index('Month')

    import numpy as np

    index_min_1 = data4.index

    for i in range(7):
        data=data_list[i].loc[index_min_1,:]
        data['RF'] =  market_data['RF']
        data['MARKET'] = market_data['MARKET']
        data['RM_RF'] = market_data['Mkt-RF']
        data['SMB'] = market_data['SMB']
        data['HML'] = market_data['HML']
        data['BIG_RF'] = data['BIG'] - data['RF']
        data['SMALL_RF'] = data['SMALL'] - data['RF']

        SMB = np.array(data['SMB'])
        HML = np.array(data['HML'])
        RM_RF = np.array(data['RM_RF'])
        
        X = np.column_stack((RM_RF,SMB,HML))
        
        Y_BIG = np.array(data['BIG_RF'] ).T
        Y_SMALL = np.array(data['SMALL_RF'] ).T
        
        X = sm.add_constant(X)
        MODEL_BIG = regression.linear_model.OLS(Y_BIG, X).fit()
        MODEL_SMALL = regression.linear_model.OLS(Y_SMALL, X).fit()

        data_big = data['BIG']
        data_small = data['SMALL']
        data_big_2 = data['BIG']-data['RF']
        data_small_2 = data['SMALL']-data['RF']
        data_big_3 = data['BIG']-data['MARKET']
        data_small_3 = data['SMALL']-data['MARKET']
        
        factor = factor_list[i] 
        
        rm_rf_mean = RM_RF.mean()
        rf_mean = data['RF'].mean()
        
        big.loc['Average Returns',factor] = data_big.mean()
        big.loc['Std',factor] = np.std(data_big)
        big.loc['Sharpe Ratio',factor] = data_big_2.mean()/np.std(data_big)
        big.loc['Alpha',factor] = MODEL_BIG.params[0]    
        big.loc['Beta',factor] = MODEL_BIG.params[1]
        big.loc['Treynor Ratio',factor] = data_big_2.mean()/MODEL_BIG.params[1]
        big.loc['Jensen Measure',factor] = data_big.mean()-rf_mean-MODEL_BIG.params[1]*rm_rf_mean
        big.loc['Information Ratio',factor] = data_big_3.mean()/np.std(data_big_3)
        
        small.loc['Average Returns',factor] = data_small.mean()
        small.loc['Std',factor] = np.std(data_small)
        small.loc['Sharpe Ratio',factor] = data_small_2.mean()/np.std(data_small)
        small.loc['Alpha',factor] = MODEL_SMALL.params[0]
        small.loc['Beta',factor] = MODEL_SMALL.params[1]
        small.loc['Treynor Ratio',factor] = data_small_2.mean()/MODEL_SMALL.params[1]
        small.loc['Jensen Measure',factor] = data_small.mean()-rf_mean-MODEL_SMALL.params[1]*rm_rf_mean
        small.loc['Information Ratio',factor] = data_small_3.mean()/np.std(data_small_3)


    print(small)
    print(big)


    ### 1.2 Market Beta

    market_data = pd.read_csv('D:/PortfolioManagement/F-F.csv')
    market_data['MARKET'] = market_data['Mkt-RF'] + market_data['RF']
    market_data['Month'] = pd.to_datetime(market_data['Month'],format ='%Y%m')
    market_data = market_data.set_index('Month')

    data1,a1 = beta(data1,market_data)
    data2,a2 = beta(data2,market_data)
    data3,a3 = beta(data3,market_data)
    data4,a4 = beta(data4,market_data)
    data5,a5 = beta(data5,market_data)
    data6,a6 = beta(data6,market_data)
    data7,a7 = beta(data7,market_data)

    a_list = [a1,a2,a3,a4,a5,a6,a7]
    data_list=[data1,data2,data3,data4,data5,data6,data7]

    plt.figure(figsize=(20,12))

    for i in range(7):
        data = data_list[i]
        plt.subplot(3,3,i+1)
        plt.plot(data.index,data['SMALL_BETA'])
        plt.plot(data.index,data['BIG_BETA'])
        plt.legend(['SMALL-β','BIG-β'])
        plt.title(factor_list[i],fontsize = 15)
        plt.grid(True)
        
    plt.show()


    ### 1.3 Beta Neutral
    output1 = beta_neutral_2(data1,a1)
    output2= beta_neutral_2(data2,a2)
    output3 = beta_neutral_2(data3,a3)
    output4 = beta_neutral_2(data4,a4,False)
    output5 = beta_neutral_2(data5,a5)
    output6 = beta_neutral_2(data6,a6,False)
    output7 = beta_neutral_2(data7,a7)

    data_list = [data1,data2,data3,data4,data5,data6,data7]
    output_list = [output1,output2,output3,output4,output5,output6,output7]


    # Plots

    plt.figure(figsize=(20,12))

    for i in range(7):
        output=output_list[i]
        plt.subplot(3,3,i+1)
        plt.plot(output.index,output['Beta_Neutral_Performance'].cumsum()/100)
        plt.legend(['Beta Neutral Portfolio'])
        plt.title(factor_list[i],fontsize = 15)
        plt.grid(True)
        
    plt.show()


    # Combine
    plt.figure(figsize=(12,6))
    index_min_2 = data4.index[1:]
    for i in range(7):
        output = output_list[i].loc[index_min_2,:]
        plt.plot(output.index,output['Beta_Neutral_Performance'].cumsum()/100)
        plt.legend(factor_list)
        plt.title('Beta Neutral Portfolios for Seven Factors',fontsize = 15)
    plt.grid(True)
    plt.show()
        

    # Statistics Summary
    output_2 = pd.DataFrame()


    import numpy as np

    for i in range(7):
        data = output_list[i].loc[index_min_2,:]
        data['RF'] =  market_data['RF']
        data['MARKET'] = market_data['MARKET']
        data['RM_RF'] = market_data['Mkt-RF']
        data['SMB'] = market_data['SMB']
        data['HML'] = market_data['HML']
        data['BIG_RF'] = data['Beta_Neutral_Performance'] - data['RF']

        SMB = np.array(data['SMB'])
        HML = np.array(data['HML'])
        RM_RF = np.array(data['RM_RF'])
        
        X = np.column_stack((RM_RF,SMB,HML))
        
        Y = np.array(data['Beta_Neutral_Performance'] ).T    
        X = sm.add_constant(X)
        MODEL = regression.linear_model.OLS(Y, X).fit()

        data_p = data['Beta_Neutral_Performance']
        data_p_2 = data['Beta_Neutral_Performance']-data['RF']
        data_p_3 = data['Beta_Neutral_Performance']-data['MARKET']
        
        factor = factor_list[i] 
        
        rm_rf_mean = RM_RF.mean()
        rf_mean = data['RF'].mean()
        
        output_2.loc['Average Returns',factor] = data_p.mean()
        output_2.loc['Std',factor] = np.std(data_p)
        output_2.loc['Sharpe Ratio',factor] = data_p_2.mean()/np.std(data_p)
        output_2.loc['Alpha',factor] = MODEL.params[0]    
        output_2.loc['Beta',factor] = MODEL.params[1]
        output_2.loc['Treynor Ratio',factor] = data_p_2.mean()/MODEL.params[1]
        output_2.loc['Jensen Measure',factor] = data_p.mean()-rf_mean-MODEL.params[1]*rm_rf_mean
        output_2.loc['Information Ratio',factor] = data_p_3.mean()/np.std(data_p_3)

    print(output_2)





    #===========================================================================
    ##### Question 2
    ### ERC
    df = pd.DataFrame()

    for i in range(7):
        df[factor_list[i]] =( data_list[i]['SMALL']+data_list[i]['BIG'])/2
    df = df.dropna()

    erc_df_1,simple_1,erc_list_1 = erc(df,60)
    look_back = [1,2,3,4,5]
    erc_df_all=[]
    simple_all=[]
    erc_all=[]

    plt.figure(figsize=(12,6))
    idx = erc_df_1.index[1:]
    plt.plot(idx,pd.Series(simple_1).cumsum()/100)
    plt.plot(idx,pd.Series(erc_list_1).cumsum()/100)
    plt.legend(['Static Equally Weighted Portfolio','Equal Risk Contribution Portfolio'])
    plt.title('5-Year Look Back Period',fontsize = 15)
    plt.grid(True)
    plt.show()

    ### ERC Weights
    weights_columns = [i+'_weight' for i in factor_list]

    plt.figure(figsize=(25,15))
    for i in range(7):
        w = weights_columns[i]
        plt.subplot(4,2,i+1)
        plt.plot(erc_df_1.index,erc_df_1[w])
        plt.legend([factor_list[i]+'-weight'])
        plt.title(factor_list[i],fontsize = 18)
        plt.grid(True)
        
    plt.show()


    ### ERC for [1,2,3,4,5] year look back periods

    for i in range(5):
        year = look_back[i]
        erc_df,simple,erc_list = erc(df,12*year)
        erc_df_all.append(erc_df)
        simple_all.append(simple)
        erc_all.append(erc_list)


    plt.figure(figsize=(23,10))

    for i in range(5):
        year = look_back[i]
        erc_df = erc_df_all[i]
        simple = simple_all[i]
        erc_list = erc_all[i]
        plt.subplot(2,3,i+1)
        plt.plot(erc_df.index[1:],pd.Series(simple).cumsum()/100)
        plt.plot(erc_df.index[1:],pd.Series(erc_list).cumsum()/100)
        plt.legend(['Static Equally Weighted Portfolio','Equal Risk Contribution Portfolio'])
        plt.title(str(year)+'-Year Look Back Period',fontsize = 16)
        plt.grid(True)
        
    plt.show()

    ### Correlation
    df.corr()







    #========================================================
    ### Question 3
    ### Factor persistance
    market_data = pd.read_csv('D:/PortfolioManagement/F-F.csv')
    market_data['MARKET'] = market_data['Mkt-RF'] + market_data['RF']
    market_data['Month'] = pd.to_datetime(market_data['Month'],format ='%Y%m')
    market_data = market_data.set_index('Month')

    df = pd.DataFrame()

    for i in range(7):
        df[factor_list[i]] =( data_list[i]['SMALL']+data_list[i]['BIG'])/2
    df = df.dropna()

    look_back_periods = [1,2,3,4,5]

    sharpe_ratio_long_short = []
    sharpe_ratio_long_only = []

    average_mean_long_short = []
    average_mean_long_only = []

    information_ratio_long_short = []
    information_ratio_long_only = []

    simple_all = []
    df_all=[]
    erc_all_data=[]

    for year in look_back_periods:
        look_back =year*12
        erc_df,simple,erc_list = erc(df,12*year)
        erc_all_data.append(erc_list)
        
        
        data,x11,x12 = sharpe_ratio_equal(df,market_data,look_back,long_only=False)
        sharpe_ratio_long_short.append(x11)
        simple_all.append(x12)
        df_all.append(data)
        
        data,x21,x22 = sharpe_ratio_equal(df,market_data,look_back,long_only=True)
        
        sharpe_ratio_long_only.append(x21)
        
        average_mean_long_short.append(average_mean_equal(df,market_data,look_back,long_only=False))
        average_mean_long_only.append(average_mean_equal(df,market_data,look_back,long_only=True))
        
        information_ratio_long_short.append(information_ratio_equal(df,market_data,look_back,long_only=False))
        information_ratio_long_only.append(information_ratio_equal(df,market_data,look_back,long_only=True))
    plt.figure(figsize=(23,10))

    output_combine = []

    for i in range(5):
        year = look_back_periods[i]
        sr = sharpe_ratio_long_only[i]
        am = average_mean_long_only[i]
        ir = information_ratio_long_only[i]
        si = simple_all[i]
        new_df = df_all[i]
        idx = new_df.index[1:]
        erc_list = erc_all_data[i]
        
        plt.subplot(2,3,i+1)
        plt.plot(idx,pd.Series(sr).cumsum()/100)
        plt.plot(idx,pd.Series(am).cumsum()/100)
        plt.plot(idx,pd.Series(ir).cumsum()/100)
        plt.plot(idx,pd.Series(erc_list).cumsum()/100)
        plt.plot(idx,pd.Series(si).cumsum()/100)
        plt.title(str(year)+'-Year Look Back Period',fontsize = 18)
        
        plt.legend(['Sharpe Ratio Weighted','Past Returns Weighted',\
                    'Information Ratio Weighted','Equal Risk Contribution','Simple Equally Weighted'])

        plt.grid(True)
        
        data = pd.DataFrame()
        data['sr'] = pd.Series(sr,index=idx)
        data['am'] = pd.Series(am,index=idx)
        data['ir'] = pd.Series(ir,index=idx)
        data['erc'] = pd.Series(erc_list,index=idx)
        data['si'] = pd.Series(si,index=idx)
        
        
        
        data['RF'] =  market_data['RF']
        data['MARKET'] = market_data['MARKET']
        data['RM_RF'] = market_data['Mkt-RF']
        data['SMB'] = market_data['SMB']
        data['HML'] = market_data['HML']

        SMB = np.array(data['SMB'])
        HML = np.array(data['HML'])
        RM_RF = np.array(data['RM_RF'])
        
        X = np.column_stack((RM_RF,SMB,HML))
        X = sm.add_constant(X)
        
        factor_list_2 = ['sr','am','ir','erc','si']
        factor_full_name = ['Sharpe Ratio Weighted','Past Returns Weighted',\
                            'Information Ratio Weighted','Equal Risk Contribution','Simple Equally Weighted']
        
        output_3 = pd.DataFrame()
        for j in range(5):
            factor = factor_list_2[j]
            factor_full = factor_full_name[j]
            Y = np.array(data[factor] ).T    

            MODEL = regression.linear_model.OLS(Y, X).fit()

            data_p = data[factor]
            data_p_2 = data[factor]-data['RF']
            data_p_3 = data[factor]-data['MARKET']

            rm_rf_mean = RM_RF.mean()
            rf_mean = data['RF'].mean()

            output_3.loc['Average Returns',factor_full] = data_p.mean()
            output_3.loc['Std',factor_full] = np.std(data_p)
            output_3.loc['Sharpe Ratio',factor_full] = data_p_2.mean()/np.std(data_p)
            output_3.loc['Alpha',factor_full] = MODEL.params[0]    
            output_3.loc['Beta',factor_full] = MODEL.params[1]
            output_3.loc['Treynor Ratio',factor_full] = data_p_2.mean()/MODEL.params[1]
            output_3.loc['Jensen Measure',factor_full] = data_p.mean()-rf_mean-MODEL.params[1]*rm_rf_mean
            output_3.loc['Information Ratio',factor_full] = data_p_3.mean()/np.std(data_p_3)
        
        output_combine.append(output_3)

    ### Statistics Summary
    print(output_combine[0])
    print(output_combine[1])
    print(output_combine[2])
    print(output_combine[3])
    print(output_combine[4])



##############################################################################

if __name__ == '__main__':
    main()