import io
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv as pv

from io import StringIO
from datetime import timedelta
import pytz
import datetime

#import pandas_datareader.data as web
import pandas_market_calendars as mcal
#import yfinance as yf
from scipy.interpolate import CubicSpline, interp1d
from IPython.display import display

from wrds_creds import read_pgpass
wrds_username,wrds_password=read_pgpass()
print(wrds_username,wrds_password)
import wrds
db = wrds.Connection(wrds_host="wrds-pgdata.wharton.upenn.edu",
                     wrds_username=wrds_username,
                     wrds_password=wrds_password)



def option_metric_query(tick,year,day=None,month=None):
    """get OptonMetrics data from optionm.opprcdyyyy
    where yyyy is taken from the year param
    tick will map to one of the secid's listed below
    if the user specfies an optional day and month, the df will be filtered to 
    include those specfifc days (note this filters on date column on exdate-aka the expiration)"""
    sec_id_map={'DJX':102456, #Dow Jones Industrial Average
    'NDX':102480, #NASDAQ 100 Index
    'MNX':102491, #CBOE Mini-NDX Index
    'XMI':101499, #AMEX Major Market Index
    'SPX':108105, #S&P 500 Index
    'OEX':109764, #S&P 100 Index
    'MID':101507, #S&P Midcap 400 Index
    'SML':102442, #S&P Smallcap 600 Index
    'RUT':102434, #Russell 2000 Index
    'NYZ':107880, #NYSE Composite Index (Old)
    'WSX':108656  #PSE Wilshire Smallcap Index
    }

    req=f"""
    SELECT *
    FROM optionm.opprcd{year} AS tbl 
    WHERE secid = {sec_id_map.get(tick)}
            and CAST(am_settlement AS INTEGER) = 1
            and CAST(ss_flag AS INTEGER) = 0
    """
    if month:
        req+=f""" and EXTRACT(MONTH from date)={month}"""
    
    if day:
        req+=f""" and EXTRACT(DAY from date)={day}"""

    df = db.raw_sql(req, date_cols=['date','exdate','last_date'])
    df.strike_price/=1_000
    #display(df.head())
    #return req
    return df

def get_fwd_price(tick,year=None):
    sec_id_map={'DJX':102456, #Dow Jones Industrial Average
    'NDX':102480, #NASDAQ 100 Index
    'MNX':102491, #CBOE Mini-NDX Index
    'XMI':101499, #AMEX Major Market Index
    'SPX':108105, #S&P 500 Index
    'OEX':109764, #S&P 100 Index
    'MID':101507, #S&P Midcap 400 Index
    'SML':102442, #S&P Smallcap 600 Index
    'RUT':102434, #Russell 2000 Index
    'NYZ':107880, #NYSE Composite Index (Old)
    'WSX':108656  #PSE Wilshire Smallcap Index
    }

    req=f"""
    SELECT *
    FROM optionm.fwdprd{year} AS tbl 
    WHERE secid = {sec_id_map.get(tick)}
        and CAST(amsettlement AS INTEGER) = 1
    """
    #if year:
    #    req+=f""" and EXTRACT(YEAR from date)={year}"""
    
    #if day:
    #    req+=f""" and EXTRACT(DAY from date)={day}"""

    df = db.raw_sql(req, date_cols=['date','expiration'])
    #display(df.head())
    #return req
    return df



def expiration_aggregation(tick):
    sec_id_map={'DJX':102456, #Dow Jones Industrial Average
    'NDX':102480, #NASDAQ 100 Index
    'MNX':102491, #CBOE Mini-NDX Index
    'XMI':101499, #AMEX Major Market Index
    'SPX':108105, #S&P 500 Index
    'OEX':109764, #S&P 100 Index
    'MID':101507, #S&P Midcap 400 Index
    'SML':102442, #S&P Smallcap 600 Index
    'RUT':102434, #Russell 2000 Index
    'NYZ':107880, #NYSE Composite Index (Old)
    'WSX':108656  #PSE Wilshire Smallcap Index
    }
    # Used to find third fridays that are CBOE holidays
    lb='1996-01-04'
    ub='2023-08-31'
    #CBOE Holidays and Early Closes
    #   NOTE: need to come back to early closes
    cboe=mcal.get_calendar('CBOE_Index_Options')
    cboe_holidays=cboe.holidays()
    cboe_holidaylist=pd.to_datetime(cboe_holidays.holidays)
    #sched=cboe.schedule(start_date=lb, end_date=ub)
    #cboe.early_closes(schedule=sched)
    fridays = list( pd.date_range(lb, ub,freq='W-FRI', tz='US/Eastern',normalize=True).values )
    third_fridays = list( pd.date_range(lb, ub,freq='WOM-3FRI', tz='US/Eastern',normalize=True).values )
    fridays=pd.to_datetime(fridays).normalize()
    third_fridays=pd.to_datetime(third_fridays).normalize()
    diffed_fridays=list(set(fridays)-set(third_fridays))
    third_friday_holidays=list(set(third_fridays) & set(cboe_holidaylist))

    cboe_expirs=list(third_fridays.copy())
    for i,fri in enumerate(third_fridays):
        if fri in third_friday_holidays:
            #print(fri)
            cboe_expirs[i]=fri-pd.Timedelta(days=1)
            #cboe_expirs[i]=None

    #print('==============')
    cboe_expirs=pd.DatetimeIndex(cboe_expirs)
    #set(cboe_expirs)-set(third_fridays)
    #cboe_expirs
    
    date_col_used='date'
    req="WITH combined AS ("
    lb_year,ub_year=1997,2023
    for yr in range(lb_year,ub_year+1):
        if yr != ub_year:
            req+= f"""SELECT optionid, date, exdate, 
                (exdate::date - date::date) AS days_to_expiration
            FROM optionm.opprcd{yr}
            WHERE secid = {sec_id_map.get(tick)}
                AND CAST(am_settlement AS INTEGER) = 1
                AND CAST(ss_flag AS INTEGER) = 0
                AND (best_bid>0 AND best_bid IS NOT NULL)
                AND date in {tuple([i.strftime('%Y-%m-%d') for i in cboe_expirs])}

            UNION ALL
        
        """
        else:
            req+= f"""SELECT optionid, date, exdate,
                (exdate::date - date::date) AS days_to_expiration
            FROM optionm.opprcd{yr}
            WHERE secid = {sec_id_map.get(tick)}
                AND CAST(am_settlement AS INTEGER) = 1
                AND CAST(ss_flag AS INTEGER) = 0
                AND (best_bid>0 AND best_bid IS NOT NULL)
                AND date in {tuple([i.strftime('%Y-%m-%d') for i in cboe_expirs])}
            )
                """
        #print(req)
        #print(4/0)
    
    req+=f"""
        SELECT date, 
                days_to_expiration,
                COUNT(optionid)
        FROM combined
        GROUP BY date, days_to_expiration
        ORDER BY date, days_to_expiration;
        """
    df = db.raw_sql(req, date_cols=['date'])
    return df


def get_sp500_file():
    req="""
        SELECT *
        FROM comp.idx_daily
        where gvkeyx='000003'
    """
    df = db.raw_sql(req, date_cols=['datadate'])
    rename_mapping = {
        'gvkeyx': 'Global Index Key',
        'dvpsxd': 'Index Daily Dividends',
        'newnum': 'Index Number - New',
        'oldnum': 'Index Number - Old',
        'prccd': 'Index Price - Close Daily',
        'prccddiv': 'Index Value - Total Return',
        'prccddivn': 'Index Value - Total Return - Net Dividends',
        'prchd': 'Index Price - High Daily',
        'prcld': 'Index Price - Low Daily',
        'datadate':'Date'}
    df.rename(columns=rename_mapping, inplace=True)
    return df
    #yf_df = yf.download('SP500TR', start='1990-MM-DD', end='2025-06-30')
    #return yf_df


def option_liquidation_query(tick,option_ids,liquid_date):
    """
    get prices for a list of optionids on a day provided
    """
    sec_id_map={'DJX':102456, #Dow Jones Industrial Average
    'NDX':102480, #NASDAQ 100 Index
    'MNX':102491, #CBOE Mini-NDX Index
    'XMI':101499, #AMEX Major Market Index
    'SPX':108105, #S&P 500 Index
    'OEX':109764, #S&P 100 Index
    'MID':101507, #S&P Midcap 400 Index
    'SML':102442, #S&P Smallcap 600 Index
    'RUT':102434, #Russell 2000 Index
    'NYZ':107880, #NYSE Composite Index (Old)
    'WSX':108656  #PSE Wilshire Smallcap Index
    }

    req=f"""
    SELECT *
    FROM optionm.opprcd{liquid_date.year} AS tbl 
    WHERE secid = {sec_id_map.get(tick)}
        AND optionid in {tuple([int(i) for i in option_ids])}
    """

    df = db.raw_sql(req, date_cols=['date','exdate','last_date'])
    df.strike_price/=1_000
    df=df.copy()
    df=df[df['date']<=liquid_date]
    df['liquidation_bid']=df.groupby(['optionid'])['best_bid'].ffill()
    df['liquidation_offer']=df.groupby(['optionid'])['best_offer'].ffill()
    df['liquidation_midprice']=(df['liquidation_offer']+df['liquidation_bid'])/2
    df=df.loc[df.groupby('optionid')['date'].idxmax()]
    df.rename(columns={'date':'liquidation_midprice_date'},inplace=True)
    df=df[['optionid','liquidation_midprice_date','liquidation_midprice']]
    #could do minimum of max dates by optionid
    #display(df.head())
    return df

