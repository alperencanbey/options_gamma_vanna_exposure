# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 14:14:47 2021

@author: Alperen Canbey

An exercise to calculate cumulative gamma and vanna in the market
    and relation between these exposures
Assuming that the market makers are long calls, short puts

"""

import pyreadr
import datetime
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

import pandas_datareader as pdr




result = pyreadr.read_r('C:/Users/Alperen Canbey/Desktop/Academia 3/Sentiment RA/optiondata_gamma_filtered_new.rds') # also works for RData
df = result[None] # extract the pandas data frame 


df["strike_price"] = df["strike_price"]/1000 #Arranging the strike price  

df["gamma_imbalance"] = df["gamma"]*df["open_interest"]*100*((df["cp_flag"] == "C")*2-1)
df["time_to_maturity"] = (df["exDates"]-df["dates"])
df["time_to_maturity"] = ((df["time_to_maturity"] / np.timedelta64(1, 'D')).astype(int) ) /365

df['dates'] = df['dates'].apply(lambda x: x.strftime("%Y-%m-%d"))


#Getting the closing prices of the index

spx = pdr.get_data_yahoo('^GSPC', 
                          start=datetime.datetime(2020, 1, 1), 
                          end=datetime.datetime(2021, 1, 1))

vix = pdr.get_data_yahoo('^VIX', 
                          start=datetime.datetime(2020, 1, 1), 
                          end=datetime.datetime(2021, 1, 1))

spx["dates"] = spx.index.strftime("%Y-%m-%d")
vix["dates"] = vix.index.strftime("%Y-%m-%d")

#merging the data
df_all = pd.merge(df, spx, on = 'dates')
df_all = df_all[7:]

#comparing the calculations with the values given in the dataset
df_all["delta_calc"] = ((df_all["cp_flag"] == "C")*2-1) * scipy.stats.norm(0, 1).cdf(  ((df_all["cp_flag"] == "C")*2-1) * ( ( np.log(df_all["Close"]/df_all["strike_price"]) + ( 0.5* df_all["impl_volatility"]**2)*df_all["time_to_maturity"] ) / (df_all["impl_volatility"]*np.sqrt(df_all["time_to_maturity"]))) )
df_all["delta"]


#find ATM Strike and put call options to use forward price instead of the underlying
df_all["gamma_calc"] = scipy.stats.norm(0, 1).pdf(( ( np.log(df_all["Close"]/df_all["strike_price"]) + ( 0.5* df_all["impl_volatility"]**2)*df_all["time_to_maturity"] ) / (df_all["impl_volatility"]*np.sqrt(df_all["time_to_maturity"]))) )  / ( df_all["Close"]*df_all["impl_volatility"]*np.sqrt(df_all["time_to_maturity"]) )
df_all["gamma"]

df_all["gamma_imbalance_calc"] = df_all["gamma_calc"]*df_all["open_interest"]*100*((df_all["cp_flag"] == "C")*2-1)
df_all["gamma_imbalance"]



spx_close = spx["Close"]
spx_close_lagged = spx_close.shift(1)

spx_close = spx_close[1:]
spx_close_lagged = spx_close_lagged[1:]

delta_spx =  (spx_close - spx_close_lagged)*100/spx_close_lagged


deneme = df_all.groupby('dates', as_index=False).agg('sum')
info = pd.concat( [ deneme["dates"], deneme["gamma_imbalance_calc"] ] , axis=1)


diff = (deneme["gamma_imbalance"] - deneme["gamma_imbalance_calc"])/deneme["gamma_imbalance"]



plt.style.use('seaborn-whitegrid')
#plot the slope
m, b = np.polyfit(deneme["gamma_imbalance_calc"][1:], delta_spx, 1)
plt.plot(delta_spx, m*delta_spx+b, color ="orange", linestyle = "dashed", markersize = 10)

# plotting points as a scatter plot
plt.scatter(deneme["gamma_imbalance_calc"][1:], delta_spx, color= "green", label = m,
            marker= "o", s=20)
 
    #Plotting how Gamma has changed with the market moves
# x-axis label
plt.xlabel('Gamma Imbalance')
# frequency label
plt.ylabel('SPX Change')
# plot title
plt.title('Gamma Exposure')


#plt.xticks(np.arange(-15,15, 5))
#plt.yticks(np.arange(-20,30, 5))
# showing legend
plt.legend()

# function to show the plot
plt.show()


#Change in VIX vs Gamma
vix_close = vix["Close"]
vix_close_lagged = vix_close.shift(1)

vix_close = vix_close[1:]
vix_close_lagged = vix_close_lagged[1:]

delta_vix =  vix_close - vix_close_lagged


plt.style.use('seaborn-whitegrid')
#plot the slope
m, b = np.polyfit(deneme["gamma_imbalance_calc"][1:], delta_vix, 1)
plt.plot(delta_spx, m*delta_spx+b, color ="orange", linestyle = "dashed", markersize = 10)

# plotting points as a scatter plot
plt.scatter(deneme["gamma_imbalance_calc"][1:], delta_vix, color= "green", label = m,
            marker= "o", s=20)
 
# x-axis label
plt.xlabel('Gamma Imbalance')
# frequency label
plt.ylabel('VIX Change')
# plot title
plt.title('Gamma Exposure')


#plt.xticks(np.arange(-15,15, 5))
#plt.yticks(np.arange(-20,30, 5))
# showing legend
plt.legend()

# function to show the plot
plt.show()

##Underlying price vs Gamma
for date in deneme["dates"]:
    print(date)
    df_temp = df_all[df_all["dates"] == date]
    input_gamma = []
    for s in range(1500,4500,50):
        
        df_temp["gamma_calc"] = scipy.stats.norm(0, 1).pdf(( ( np.log(s/df_temp["strike_price"]) + ( 0.5* df_temp["impl_volatility"]**2)*df_temp["time_to_maturity"] ) / (df_temp["impl_volatility"]*np.sqrt(df_temp["time_to_maturity"]))) )  / ( s*df_temp["impl_volatility"]*np.sqrt(df_temp["time_to_maturity"]) )
        df_temp["gamma_imbalance_calc"] = df_temp["gamma_calc"]*df_temp["open_interest"]*100*((df_temp["cp_flag"] == "C")*2-1)
        gamma_total = np.sum(df_temp["gamma_imbalance_calc"])
        input_gamma.append((s, gamma_total))
    
    df_temp["gamma_calc"] = scipy.stats.norm(0, 1).pdf(( ( np.log(df_temp["Close"]/df_temp["strike_price"]) + ( 0.5* df_temp["impl_volatility"]**2)*df_temp["time_to_maturity"] ) / (df_temp["impl_volatility"]*np.sqrt(df_temp["time_to_maturity"]))) )  / (df_temp["Close"]*df_temp["impl_volatility"]*np.sqrt(df_temp["time_to_maturity"]) )
    df_temp["gamma_imbalance_calc"] = df_temp["gamma_calc"]*df_temp["open_interest"]*100*((df_temp["cp_flag"] == "C")*2-1)
    
    gamma_total = np.sum(df_temp["gamma_imbalance_calc"])
    gamma_graph= pd.DataFrame(list(input_gamma))
    
    
    plt.style.use('seaborn-whitegrid')
    #plot the slope
    m, b = np.polyfit(gamma_graph[0], gamma_graph[1], 1)
    plt.plot(delta_spx, m*delta_spx+b, color ="orange", linestyle = "dashed", markersize = 10)
    
    # plotting points as a scatter plot
    
    plt.scatter(gamma_graph[0], gamma_graph[1], color= "green", label = m,
                marker= "o", s=20)
    plt.scatter(df_temp["Close"][0:1], gamma_total, color= "orange", label = m,
                marker= "o", s=20)
    
    # x-axis label
    plt.xlabel('Underlying Price')
    # frequency label
    plt.ylabel('Gamma Imbalance')
    # plot title
    plt.title('Gamma Exposure')
    
    
    #plt.xticks(np.arange(-15,15, 5))
    #plt.yticks(np.arange(-20,30, 5))
    # showing legend
    plt.legend()
    
    # function to show the plot
    plt.show()
    
    
    
    

              
#VANNA EXPOSURE

date = "2020-09-02"

for date in deneme["dates"]:
    print(date)
    df_temp = df_all[df_all["dates"] == date]
    input_vanna = []
    for s in range(1500,4500,50):
        df_temp["d1"] =  ( np.log(s/df_temp["strike_price"]) + ( 0.5* df_temp["impl_volatility"]**2)*df_temp["time_to_maturity"] ) / (df_temp["impl_volatility"]*np.sqrt(df_temp["time_to_maturity"]))
        df_temp["vanna_calc"] = scipy.stats.norm(0, 1).pdf( df_temp["d1"] ) * (1 - df_temp["d1"] ) * df_temp["time_to_maturity"] 
        df_temp["vanna_imbalance_calc"] = df_temp["vanna_calc"]*df_temp["open_interest"]*100*( ( (df_temp["cp_flag"] == "C")*(s>df_temp["strike_price"]) + (df_temp["cp_flag"] == "P")*(s<df_temp["strike_price"]) )*2-1 )
    
        vanna_total = np.sum(df_temp["vanna_imbalance_calc"])
        input_vanna.append((s, vanna_total))
    
    df_temp["d1"] =  ( np.log(df_temp["Close"]/df_temp["strike_price"]) + ( 0.5* df_temp["impl_volatility"]**2)*df_temp["time_to_maturity"] ) / (df_temp["impl_volatility"]*np.sqrt(df_temp["time_to_maturity"]))
    df_temp["vanna_calc"] = scipy.stats.norm(0, 1).pdf( df_temp["d1"] ) * (1 - df_temp["d1"] ) * df_temp["time_to_maturity"] 
    df_temp["vanna_imbalance_calc"] = df_temp["vanna_calc"]*df_temp["open_interest"]*100*( ( (df_temp["cp_flag"] == "C")*(df_temp["Close"]>df_temp["strike_price"]) + (df_temp["cp_flag"] == "P")*(df_temp["Close"]<df_temp["strike_price"]) )*2-1 )
    
    
    vanna_total = np.sum(df_temp["vanna_imbalance_calc"])
    vanna_graph= pd.DataFrame(list(input_vanna))
    
    
    plt.style.use('seaborn-whitegrid')
    #plot the slope
    m, b = np.polyfit(vanna_graph[0], vanna_graph[1], 1)
    plt.plot(delta_spx, m*delta_spx+b, color ="orange", linestyle = "dashed", markersize = 10)
    
    # plotting points as a scatter plot
    
    plt.scatter(vanna_graph[0], vanna_graph[1], color= "green", label = m,
                marker= "o", s=20)
    plt.scatter(df_temp["Close"][0:1], vanna_total, color= "orange", label = m,
                marker= "o", s=20)
    
    # x-axis label
    plt.xlabel('Underlying Price')
    # frequency label
    plt.ylabel('Vanna Imbalance')
    # plot title
    plt.title('Vanna Exposure')
    
    
    #plt.xticks(np.arange(-15,15, 5))
    #plt.yticks(np.arange(-20,30, 5))
    # showing legend
    plt.legend()
    
    # function to show the plot
    plt.show()




# UNDERSTAND THE REALATION BTW GEX VEX
df_all["d1"] =  ( np.log(df_all["Close"]/df_all["strike_price"]) + ( 0.5* df_all["impl_volatility"]**2)*df_all["time_to_maturity"] ) / (df_all["impl_volatility"]*np.sqrt(df_all["time_to_maturity"]))
df_all["vanna_calc"] = scipy.stats.norm(0, 1).pdf( df_all["d1"] ) * (1 - df_all["d1"] ) * df_all["time_to_maturity"] 
df_all["vanna_imbalance_calc"] = df_all["vanna_calc"]*df_all["open_interest"]*100*( ( (df_all["cp_flag"] == "C")*(df_all["Close"]>df_all["strike_price"]) + (df_all["cp_flag"] == "P")*(df_all["Close"]<df_all["strike_price"]) )*2-1 )

vanna_aggragate = df_all.groupby('dates', as_index=False).agg('sum')
vanna_info = pd.concat( [ vanna_aggragate["dates"], vanna_aggragate["vanna_imbalance_calc"] ] , axis=1)

#vix_gamma_return = pd.merge(info, delta_spx, on = 'dates')


plt.style.use('seaborn-whitegrid')
#plot the slope
m, b = np.polyfit(vanna_aggragate["vanna_imbalance_calc"][1:], delta_spx, 1)
plt.plot(delta_spx, m*delta_spx+b, color ="orange", linestyle = "dashed", markersize = 10)

# plotting points as a scatter plot

plt.scatter(vanna_aggragate["vanna_imbalance_calc"][1:], delta_spx, color= "green", label = m,
            marker= "o", s=20)
 
# x-axis label
plt.xlabel('Vanna Imbalance')
# frequency label
plt.ylabel('SPX Change')
# plot title
plt.title('Vanna Exposure')


#plt.xticks(np.arange(-15,15, 5))
#plt.yticks(np.arange(-20,30, 5))
# showing legend
plt.legend()

# function to show the plot
plt.show()
