#ML - US UNEMPLOYMENT - GETTING AND CLEANING DATA

import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

import quandl
api_key=open('MY-QUANDL-API.txt.','r').read()

def all_50_states():
    states_list = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
    return states_list[0][0][1:]

#print(all_50_states())

def all_state_unemployment():

    states = all_50_states()
    main_df = pd.DataFrame()
    
    for list in states:
        query = "FRBC/UNEMP_ST_"+str(list)
        df = quandl.get(query, trim_start="1990-01-01", authtoken=api_key)
        df.rename(columns={'Value': list}, inplace=True)
        df[list] = (df[list]-df[list][0]) / df[list][0] * 100.0
        print(df.head())
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df)
   
    pickle_out = open('all_state_unemployment.pickle','wb')
    pickle.dump(main_df, pickle_out)
    pickle_out.close()

    

def National_unemployment():
    #Federal Reserve
    df = quandl.get("FRED/LNU03000000", trim_start="1990-01-01", authtoken=api_key)
    df["VALUE"] = (df["VALUE"]-df["VALUE"][0]) / df["VALUE"][0] * 100.0
    df.rename(columns={'VALUE':'National_Unemployment_Rate'}, inplace=True)
    df=df.resample('1D')
    df=df.resample('M').mean()
    print("This is National unemployment rate")
    print(df.head())
    return df

def National_HPI():
    #FREDDIE MAC HPI INDEX
    df = quandl.get("FMAC/HPI_USA", trim_start="1990-01-01", authtoken=api_key)
    df["United States"] = (df["Value"]-df["Value"][0]) / df["Value"][0] * 100.0
    df.rename(columns={'United States':'National_HPI'}, inplace=True)
    df = df["National_HPI"]
    df.column=['National_HPI']
    print("This is National HPI")
    print(df.head())
    return df

def GDP():
    df = quandl.get("BCB/4385", trim_start="1990-01-01", authtoken=api_key)
    df["Value"] = (df["Value"]-df["Value"][0]) / df["Value"][0] * 100.0
    df=df.resample('M').mean()
    df.rename(columns={'Value':'GDP'}, inplace=True)
    print("This is National GDP")
    #df = df['GDP']
    print(df.head())
    return df


def SP500():
    df = quandl.get("YAHOO/INDEX_GSPC", trim_start="1990-01-01", authtoken=api_key)
    df["Adjusted Close"] = (df["Adjusted Close"]-df["Adjusted Close"][0]) / df["Adjusted Close"][0] * 100.0
    df=df.resample('M').mean()
    df.rename(columns={'Adjusted Close':'SP500'}, inplace=True)
    print("This is Closing Price of S&P500")
    df = df["SP500"]
    df.column=['SP500']
    print(df.head())
    return df

def Inflation():
    df = quandl.get("FRBC/USINFL", trim_start="1990-01-01", authtoken=api_key)
    #df["Value"] = (df["Value"]-df["Value"][0]) / df["Value"][0] * 100.0
    df=df.resample('1D')
    df=df.resample('M').mean()
    df.rename(columns={'CPI-U: All Items (SA; 1982-84=100) % Change - Period to Period (Bureau of Labor Statistics)':
                       'Inflation'}, inplace=True)
    df = df['Inflation']
    df.column=['Inflation']    
    print("This is Inflation")
    print(df.head())
    return df


#Call Function and Combine Data into Data Frame

State_Unemployment = pd.read_pickle('all_state_unemployment.pickle')
State_Unemployment = State_Unemployment.pct_change()
State_Unemployment.dropna(inplace=True)

US_Unemployment = National_unemployment()
HPI = National_HPI()
US_GDP = GDP()
SP_500 = SP500()
US_Inflation = Inflation()


Economic_Data = US_Unemployment.join([HPI, US_GDP, SP_500, US_Inflation])
Economic_Data.dropna(inplace=True)
print(Economic_Data.head())

pickle_out = open('Economic_Data.pickle','wb')
pickle.dump(Economic_Data, pickle_out)
pickle_out.close()

Unemployment_Data = State_Unemployment.join([Economic_Data])
print(Unemployment_Data.head())

pickle_out = open('Unemployment_Data.pickle','wb')
pickle.dump(Unemployment_Data, pickle_out)
pickle_out.close()

#STATISTICS

#print(Economic_Data.corr())

#MACHINE LEARNING

##Unemployment_Data['US_Unemployment_Future']=Unemployment_Data['US_Unemployment'].shift(-1)
##
##def create_labels(cur_unemp, fut_unemp):
##    if fut_unemp < cur_unemp:
##        return 1
##    else:
##        return 0
##
##Unemployment_Data['label']=list(map(create_labels,
##                                    Unemployment_data['US_Unemployment'],
##                                    Unemployment_Data['US_Unemployment_Future']))
##
##from sklearn import svm, preprocessing, cross_validation






