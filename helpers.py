import io
import warnings
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from datetime import timedelta
import pytz
import datetime
import pandas_market_calendars as mcal
import matplotlib.dates as mdates
from scipy.interpolate import CubicSpline, interp1d
import statsmodels.api as sm


from wrds_queries import (option_metric_query,
                              get_fwd_price, 
                              get_sp500_file,
                              option_liquidation_query)

def get_settle_values():
    """get values of SPX SET index used to settle S&P Options"""
    settle_df = pd.read_csv('set-history.csv',header=None)
    settle_dict={}
    month_mapping={'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,'July':7,
                'August':8,'September':9,'October':10,'November':11,'December':12}

    _counter=0
    for index,row in settle_df.iterrows():
        if row[0].endswith('Settlement Values'):
            settle_dict[_counter]={}
            settle_dict[_counter]['Month']=month_mapping.get(row[0].split()[0])
            settle_dict[_counter]['Year']=int(row[0].split()[1])
            #_counter+=1
        elif row[0].startswith('S&P 500 (SET)'):
            settle_dict[_counter]['Settlement_Value']=row[1]
            _counter+=1
        
    settle_prices=pd.DataFrame.from_dict(settle_dict,orient='index')
    settle_prices.loc[len(settle_prices)]={'Month':3,'Year':1998,'Settlement_Value':1089.74}
    settle_prices.loc[len(settle_prices)]={'Month':2,'Year':1998,'Settlement_Value':1028.28}
    settle_prices.loc[len(settle_prices)]={'Month':1,'Year':1998,'Settlement_Value':950.73}
    settle_prices.loc[len(settle_prices)]={'Month':10,'Year':1999,'Settlement_Value':1267.13}
    #settle_prices[ (settle_prices['Month']==10) & (settle_prices['Year']==2001)]
    #settle_prices[ (settle_prices['Month']==4) & (settle_prices['Year']==2022)]
    settle_prices['Settlement_Value']=settle_prices['Settlement_Value'].apply(lambda x:float(re.sub(r'[^0-9.]', '', x)) if type(x)==str else x )
    settle_prices#.sort_values(by=['Year','Month'],ascending=[False,False])
    return settle_prices

from interest_rate_consol import get_yc_history
yc_pull=get_yc_history()

def find_between(lst, num):
    """Finds the two numbers in a sorted list that a given number is between."""
    lst.sort()  # Ensure the list is sorted
    for i in range(len(lst) - 1):
        if lst[i] <= num <= lst[i + 1]:
            return lst[i], lst[i + 1]
    return None  # Number is not between any pair in the list

def calculate_interest_rates(current_date,t):
    try:
        yc_yday=yc_pull.iloc[ yc_pull.index.get_loc(str(current_date))+1 ]
    except:
        yc_yday=yc_pull.iloc[yc_pull.index.get_indexer([str(current_date)],method='bfill')[0] ]
    
    intvl=find_between(list(yc_yday.keys()), t) 
    if intvl:
        try:
            BEY=CubicSpline(list(intvl),[yc_yday[i] for i in intvl],
                            bc_type='natural',extrapolate=True )(t,0)
            APY=(1+BEY/2)**2-1
            r=np.log(1+APY)
        except:
            #problem no interest rate for 120 days, use 182 instead if 121 not avail
            yc_yday=yc_yday.dropna()
            intvl=find_between(list(yc_yday.keys()), t)
            BEY=CubicSpline(list(intvl),[yc_yday[i] for i in intvl],
                bc_type='natural',extrapolate=True )(t,0)
            APY=(1+BEY/2)**2-1
            r=np.log(1+APY)

    elif t<min(yc_yday.index):
        t_1,CMT_1=yc_yday.index[0],yc_yday.iloc[0]
        t_x,CMT_x=yc_yday.index[1],yc_yday.iloc[1]
        m_low=(CMT_x-CMT_1)/(t_x-t_1)
        b_low=CMT_1-m_low*t_1
        m_up=0
        b_up=CMT_1+m_up*t_1
        BEY=CubicSpline([t_1,t_x],
                        [b_low,b_up],bc_type='natural',extrapolate=True )(t,0)
        APY=(1+BEY/2)**2-1
        r=np.log(1+APY)
    
    return r



def compute_option_portfolios(num_periods_wanted=3,lb_year=1998,ub_year=2023,ticker='SPX'):
    
    warnings.simplefilter(action='ignore', category=FutureWarning)
    #New logic: find closest date to third friday that all options price on
    #num_periods_wanted=3
    #approx_period_days=[30*i for i in range(1,num_periods_wanted+1)]
    approx_period_days=[30,60,90]
    option_portfolios=[]
    id_counter=1

    sp500_df=get_sp500_file()
    sp500_df.dropna(subset=['Index Value - Total Return'],inplace=True)
    sp500_df=sp500_df[['Date','Index Value - Total Return','Index Price - Close Daily']]
    sp500_df['Index Return']=sp500_df['Index Value - Total Return']/sp500_df['Index Value - Total Return'].shift()-1

    settle_prices=get_settle_values()

    #lb_year=2006
    #ub_year=2006
    #lb_year=1998
    #ub_year=2023
    portfolio_dict={}
    #portfolio_df=pd.DataFrame()
    id_counter=1
    for yr in range(lb_year,ub_year+1):
        print(f'{yr} file')
        df = option_metric_query(ticker,yr,day=None,month=None)
        df2=df.copy()
        lb='1996-01-04'
        ub='2040-12-31'
        #CBOE Holidays and Early Closes
        #   NOTE: need to come back to early closes
        cboe=mcal.get_calendar('CBOE_Index_Options')
        cboe_holidays=cboe.holidays()
        cboe_holidaylist=pd.to_datetime(cboe_holidays.holidays)
        fridays = list( pd.date_range(lb, ub,freq='W-FRI', tz='US/Eastern',normalize=True).values )
        third_fridays = list( pd.date_range(lb, ub,freq='WOM-3FRI', tz='US/Eastern',normalize=True).values )
        fridays=pd.to_datetime(fridays).normalize()
        third_fridays=pd.to_datetime(third_fridays).normalize()
        diffed_fridays=list(set(fridays)-set(third_fridays))
        holiday_dict={}
        third_friday_holidays=list(set(third_fridays) & set(cboe_holidaylist))
        expir_not_third_friday=set(pd.to_datetime(df2.exdate.unique()).normalize())-set(third_fridays)
        for expir in expir_not_third_friday:
            for holiday in third_friday_holidays:
                if (holiday-expir).days==1:
                    holiday_dict[expir]=True

        df2['holiday_exp']=df2['exdate'].map(holiday_dict)
        df2.fillna({'holiday_exp':False},inplace=True)

        df3=df2.copy()    
        df3['datetime_close']=df3['date']+pd.Timedelta(hours=16,minutes=15)
        df3['ex_time']=df3['exdate']+pd.Timedelta(hours=9,minutes=30)
        df3['time_to_exp']=df3['ex_time']-df3['datetime_close']

        df_fwd=get_fwd_price(ticker,yr)
        df4=df3.merge(df_fwd,how='left',left_on=['secid','date','exdate'],right_on=['secid','date','expiration'])
        #df4[pd.isnull(df4.forwardprice)]
        #df4['OOM']=df4.apply(lambda x:True if ((x.cp_flag=='C' and x.strike_price>=x.forwardprice) or (x.cp_flag=='P' and x.strike_price<=x.forwardprice)) else False )
        def oom_indicator(row):
            if (row.cp_flag=='C' and row.strike_price>=row.forwardprice) or (row.cp_flag=='P' and row.strike_price<=row.forwardprice):
                return True
            else:
                return False
        #df4['oom']=df4.apply(oom_indicator,axis=1)
        df4['midpoint_price']=(df4.best_bid+df4.best_offer)/2
        
        expirs_wanted=[ i[0] for i in df4.groupby([df4.exdate.dt.year,df4.exdate.dt.month])[['exdate']].min().values ]
        #for expir in sorted(df4.exdate.unique()):
        for expir in expirs_wanted:
            slice_=df4[df4.date==expir]
            modified_trade_date=expir
            expir_days=sorted( df4[df4.exdate.isin(expirs_wanted)][df4[df4.exdate.isin(expirs_wanted)].exdate>expir].exdate.unique() ) 

            expirs_found=[]
            for d in approx_period_days:
                #for e in expir_days:
                if bool([(i-expir).days for i in expir_days]):
                    closest_dte=min( [(i-expir).days for i in expir_days], key=lambda x:abs(x-d))
                    idx=[(i-expir).days for i in expir_days].index(closest_dte)
                    expirs_found.append(expir_days[idx])

            #Find modified_trade_date
            trade_date_cands=[]
            for e in expirs_found:
                sub_slice_=df4[df4.exdate==e]
                trade_date_cands.append( min(sub_slice_.date.unique(), key=lambda x: abs(expir - x)) )

            if bool(trade_date_cands):
                modified_trade_date=max(trade_date_cands)
                #slice_=df4[df4.date>=modified_trade_date] #this line could be causing the problem
                slice_=df4[df4.date==modified_trade_date]
                #continue

                portfolio_dict[id_counter]={'Trade Date':expir}
                portfolio_dict[id_counter]['modified_trade_date']=modified_trade_date

                slice_temp=df4[df4.date>=modified_trade_date]
                #if not slice_[slice_.date>modified_trade_date].empty: #this was for when the year in the date was beyond the yr
                if not slice_temp[slice_temp.date>modified_trade_date].empty: #this is a fix
                    #display(slice_.head(5))
                    #continue
                    #for term in expirs_found:
                    for i,term in enumerate(expirs_found):
                        #portfolio_dict[expir][f'{i+1} month expiration']=term
                        portfolio_dict[id_counter][f'{i+1} month expiration']=term
                        term_df=slice_[(slice_.exdate==term)]
                        
                        ###FOR TESTING###
                        #if term==pd.Timestamp('2023-01-20 00:00:00'):
                        #    term_df.to_csv('test_file_1.csv')
                        #################
                        #continue

                        #if term==pd.Timestamp('2004-08-21 00:00:00'):
                        #    display(slice_[(slice_.exdate==term)])

                        #display(term_df)
                        ATM_strike_cands=term_df[~( (pd.isnull(term_df.best_bid)) | (pd.isnull(term_df.best_offer)) ) 
                                    & ~(term_df.best_bid>term_df.best_offer) 
                                    & ~(term_df.best_bid<=0)   ]
                        
                        mins_in_year=365*24*60
                        mins_to_expir=ATM_strike_cands.time_to_exp.iloc[0].total_seconds()/60
                        t=mins_to_expir/mins_in_year
                        days_to_exp=ATM_strike_cands.iloc[0].time_to_exp.days
                        #need to update day used to calc interest rate
                        #r=calculate_interest_rates(expir,ATM_strike_cands.iloc[0].time_to_exp.days)
                        r=calculate_interest_rates(modified_trade_date,ATM_strike_cands.iloc[0].time_to_exp.days)
                        #missing data in yield curve fiel was throwing error
                        #display(ATM_strike_cands)

                        def min_strike_diff(slice):
                            if ('C' in slice.cp_flag.unique()) and ('P' in slice.cp_flag.unique()):
                                return abs( slice[slice.cp_flag=='P'].midpoint_price.values[0] - slice[slice.cp_flag=='C'].midpoint_price.values[0])
                        
                        F_strike=ATM_strike_cands.groupby(['strike_price']).apply(min_strike_diff).idxmin()
                        #display(ATM_strike_cands)
                        call_put_diff=ATM_strike_cands[(ATM_strike_cands.strike_price==F_strike)].sort_values(by='cp_flag')['midpoint_price'].diff().dropna().values[0]
                        F=F_strike+np.exp(r*t)*call_put_diff
                        K0=term_df[term_df.strike_price<=F].strike_price.max()

                        def filter_included_options(opt_type):
                            if opt_type=='put':
                                OOM_opts=term_df[(term_df.strike_price<K0)&(term_df.cp_flag=='P')]
                                OOM_opts=OOM_opts.copy()
                                #OOM_opts.sort_values(by=['strike_price'],ascending=False,inplace=True) #sort upside down for puts
                                OOM_opts['excl_ind']=OOM_opts.best_bid.apply(lambda x: pd.isnull(x) or x<=0)
                                OOM_opts.sort_values(by=['strike_price'],ascending=False,inplace=True) #sort upside down for puts
                            else:
                                OOM_opts=term_df[(term_df.strike_price>K0)&(term_df.cp_flag=='C')]
                                OOM_opts=OOM_opts.copy()
                                #trying sort this time
                                OOM_opts.sort_values(by=['strike_price'],ascending=True,inplace=True) #sort upside down for puts
                                OOM_opts['excl_ind']=OOM_opts.best_bid.apply(lambda x: pd.isnull(x) or x<=0)
                                #dont need to change sorting order for calls

                            OOM_opts['excl_ind']=OOM_opts['excl_ind'].cumsum()
                            incl_opts=OOM_opts[OOM_opts.excl_ind<2]
                            incl_opts=incl_opts[incl_opts.best_bid>0]
                            if opt_type=='put':
                                incl_opts.sort_values(by=['strike_price'],ascending=True,inplace=True)#change back
                            return incl_opts

                        incl_puts,incl_calls=filter_included_options('put'),filter_included_options('call')

                        df_K0=term_df[term_df.strike_price==K0]
                        pca=pd.DataFrame.from_dict({'strike_price':K0,
                                                    'cp_flag':'P/C Avg', #put-call average
                                                    'midpoint_price':term_df[(term_df.strike_price==K0)]['midpoint_price'].mean(),
                                                    'best_bid':None,
                                                    'best_offer':None,
                                                    'optionid':None,
                                                    'delta':term_df[(term_df.strike_price==K0)]['delta'].sum(),
                                                    'forwardprice':term_df[(term_df.strike_price==K0)]['forwardprice'].unique()[0]},
                                                    orient='index').T

                        opt_portfolio=pd.concat([incl_puts,pca,incl_calls])[['strike_price',
                                                                            'cp_flag',
                                                                            'midpoint_price',
                                                                            'best_bid','best_offer',
                                                                            'optionid',
                                                                            'delta',
                                                                            'forwardprice']]

                        #addign this to try and fix the negative dK issue
                        opt_portfolio.sort_values(by=['strike_price'],ascending=True,inplace=True)
                        opt_portfolio['dK']=(opt_portfolio.strike_price.shift(-1)-opt_portfolio.strike_price.shift(1))/2
                        opt_portfolio=opt_portfolio.copy()
                        #opt_portfolio.iloc[0]['dK']=opt_portfolio.iloc[1]['strike_price']-opt_portfolio.iloc[0]['strike_price']
                        #opt_portfolio.iloc[-1]['dK']=opt_portfolio.iloc[-1]['strike_price']-opt_portfolio.iloc[-2]['strike_price']
                        opt_portfolio.iloc[0,-1]=opt_portfolio.iloc[1]['strike_price']-opt_portfolio.iloc[0]['strike_price']
                        opt_portfolio.iloc[-1,-1]=opt_portfolio.iloc[-1]['strike_price']-opt_portfolio.iloc[-2]['strike_price']
                        

                        settle_value=float(settle_prices[(settle_prices['Month']==term.month) & (settle_prices['Year']==term.year)]['Settlement_Value'].values[0])
                        def call_payoff(s,k):
                            return max(s-k,0)
                        def put_payoff(s,k):
                            return max(k-s,0)
                        def option_payoff(row):
                            k=row['strike_price']
                            opt_type=row['cp_flag']
                            if opt_type=='C':
                                return call_payoff(settle_value,k) 
                            elif opt_type=='P':
                                return put_payoff(settle_value,k)
                            else:
                                return call_payoff(settle_value,k)+put_payoff(settle_value,k)

                        opt_portfolio['payoff']=opt_portfolio.apply(option_payoff,axis=1)

                        # VIX and SVIX
                        #addign 2/t for now instead of 2
                        VIX_opt_weight=(2/t)*opt_portfolio.dK/(opt_portfolio.strike_price**2)
                        SVIX_opt_weight=(2/t)*opt_portfolio.dK/(F**2)
                        GVIX_opt_weight=(2/t)*opt_portfolio.dK/(F*opt_portfolio.strike_price)
                        #VIX_opt_weight=2*opt_portfolio.dK/(opt_portfolio.strike_price**2)
                        #SVIX_opt_weight=2*opt_portfolio.dK/(F**2)
                        # SKEW
                        p1_option_weight=-opt_portfolio.dK/(opt_portfolio.strike_price**2)
                        p2_option_weight=2*opt_portfolio.dK/(opt_portfolio.strike_price**2)
                        p2_option_weight*=1-np.log(opt_portfolio.strike_price.astype(np.float64)/F)
                        p3_option_weight=3*opt_portfolio.dK/(opt_portfolio.strike_price**2)
                        p3_option_weight*=2*np.log(opt_portfolio.strike_price.astype(np.float64)/F)-np.log(opt_portfolio.strike_price.astype(np.float64)/F)**2

                        e1=-(1+np.log(F/K0)-(F/K0))
                        e2=2*np.log(K0/F)*(F/K0-1)+(1/2)*np.log(K0/F)**2
                        e3=3*np.log(K0/F)**2 * (1/3*np.log(K0/F)-1+F/K0)
                        P1=( np.exp(r*t)*(opt_portfolio.midpoint_price*p1_option_weight) ).sum()
                        P1+=e1
                        P2=( np.exp(r*t)*(opt_portfolio.midpoint_price*p2_option_weight) ).sum()
                        P2+=e2
                        P3=( np.exp(r*t)*(opt_portfolio.midpoint_price*p3_option_weight) ).sum()
                        P3+=e3
                        sigma=np.sqrt(P2-P1**2)
                        
                        #VIX and VIX^2
                        opt_portfolio['VIX_opt_weight']=VIX_opt_weight#*np.exp(r*t)
                        portfolio_dict[id_counter][f'{i+1} month VIX']=np.sqrt(np.exp(r*t)*(opt_portfolio.VIX_opt_weight*opt_portfolio.midpoint_price).sum())*100
                        #portfolio_dict[id_counter][f'{i+1} month VIX']=np.sqrt((opt_portfolio.VIX_opt_weight*opt_portfolio.midpoint_price).sum())*100
                        portfolio_dict[id_counter][f'{i+1} month VIX payoff']=np.sqrt((opt_portfolio.VIX_opt_weight*opt_portfolio.payoff).sum())*100
                        opt_portfolio['VIX_bid_contribtion']=opt_portfolio.apply(lambda x: x.VIX_opt_weight*x.best_bid if x.VIX_opt_weight >0 else x.VIX_opt_weight*x.best_offer,axis=1)
                        opt_portfolio['VIX_ask_contribtion']=opt_portfolio.apply(lambda x: x.VIX_opt_weight*x.best_offer if x.VIX_opt_weight >0 else x.VIX_opt_weight*x.best_bid,axis=1)
                        
                        #multiply by t to get rid of the 2/t factor from earlier
                        vix_square=(t*opt_portfolio.VIX_opt_weight*opt_portfolio.midpoint_price).sum()
                        vix_square_payoff=(t*opt_portfolio.VIX_opt_weight*opt_portfolio.payoff).sum()
                        opt_portfolio['VIX_square_opt_weight']=VIX_opt_weight*t
                        portfolio_dict[id_counter][f'{i+1} month VIX^2']=vix_square
                        portfolio_dict[id_counter][f'{i+1} month VIX^2 payoff']=vix_square_payoff
                        opt_portfolio['VIX_square_bid_contribtion']=opt_portfolio.apply(lambda x: x.VIX_square_opt_weight*x.best_bid if x.VIX_square_opt_weight >0 else x.VIX_square_opt_weight*x.best_offer,axis=1)
                        opt_portfolio['VIX_square_ask_contribtion']=opt_portfolio.apply(lambda x: x.VIX_square_opt_weight*x.best_offer if x.VIX_square_opt_weight >0 else x.VIX_square_opt_weight*x.best_bid,axis=1)
                        portfolio_dict[id_counter][f'{i+1} month VIX square bid']=np.exp(r*t)*opt_portfolio['VIX_square_bid_contribtion'].sum()
                        portfolio_dict[id_counter][f'{i+1} month VIX square ask']=np.exp(r*t)*opt_portfolio['VIX_square_ask_contribtion'].sum()
                        portfolio_dict[id_counter][f'{i+1} month VIX square spread ($)']=portfolio_dict[id_counter][f'{i+1} month VIX square ask']-portfolio_dict[id_counter][f'{i+1} month VIX square bid']

                        #SVIX and SVIX^2
                        opt_portfolio['SVIX_opt_weight']=SVIX_opt_weight#*np.exp(r*t)
                        portfolio_dict[id_counter][f'{i+1} month SVIX']=np.sqrt(np.exp(r*t)*(opt_portfolio.SVIX_opt_weight*opt_portfolio.midpoint_price).sum())*100
                        #portfolio_dict[id_counter][f'{i+1} month SVIX']=np.sqrt((opt_portfolio.SVIX_opt_weight*opt_portfolio.midpoint_price).sum())*100
                        portfolio_dict[id_counter][f'{i+1} month SVIX payoff']=np.sqrt((opt_portfolio.SVIX_opt_weight*opt_portfolio.payoff).sum())*100        
                        opt_portfolio['SVIX_bid_contribtion']=opt_portfolio.apply(lambda x: x.SVIX_opt_weight*x.best_bid if x.SVIX_opt_weight >0 else x.SVIX_opt_weight*x.best_offer,axis=1)
                        opt_portfolio['SVIX_ask_contribtion']=opt_portfolio.apply(lambda x: x.SVIX_opt_weight*x.best_offer if x.SVIX_opt_weight >0 else x.SVIX_opt_weight*x.best_bid,axis=1)
                        portfolio_dict[id_counter][f'{i+1} month SVIX bid']=np.sqrt(np.exp(r*t)*opt_portfolio['SVIX_bid_contribtion'].sum())*100
                        portfolio_dict[id_counter][f'{i+1} month SVIX ask']=np.sqrt(np.exp(r*t)*opt_portfolio['SVIX_ask_contribtion'].sum())*100
                        portfolio_dict[id_counter][f'{i+1} month SVIX spread ($)']=portfolio_dict[id_counter][f'{i+1} month SVIX ask']-portfolio_dict[id_counter][f'{i+1} month SVIX bid']
                        
                        svix_square=(t*opt_portfolio.SVIX_opt_weight*opt_portfolio.midpoint_price).sum()
                        svix_square_payoff=(t*opt_portfolio.SVIX_opt_weight*opt_portfolio.payoff).sum()
                        opt_portfolio['SVIX_square_opt_weight']=SVIX_opt_weight*t
                        portfolio_dict[id_counter][f'{i+1} month SVIX^2']=svix_square
                        portfolio_dict[id_counter][f'{i+1} month SVIX^2 payoff']=svix_square_payoff
                        opt_portfolio['SVIX_square_bid_contribtion']=opt_portfolio.apply(lambda x: x.SVIX_square_opt_weight*x.best_bid if x.SVIX_square_opt_weight >0 else x.SVIX_square_opt_weight*x.best_offer,axis=1)
                        opt_portfolio['SVIX_square_ask_contribtion']=opt_portfolio.apply(lambda x: x.SVIX_square_opt_weight*x.best_offer if x.SVIX_square_opt_weight >0 else x.SVIX_square_opt_weight*x.best_bid,axis=1)
                        portfolio_dict[id_counter][f'{i+1} month SVIX square bid']=np.exp(r*t)*opt_portfolio['SVIX_square_bid_contribtion'].sum()
                        portfolio_dict[id_counter][f'{i+1} month SVIX square ask']=np.exp(r*t)*opt_portfolio['SVIX_square_ask_contribtion'].sum()
                        portfolio_dict[id_counter][f'{i+1} month SVIX square spread ($)']=portfolio_dict[id_counter][f'{i+1} month SVIX square ask']-portfolio_dict[id_counter][f'{i+1} month SVIX square bid']

                        #GVIX and GVIX^2
                        opt_portfolio['GVIX_opt_weight']=GVIX_opt_weight#*np.exp(r*t)
                        opt_portfolio['GVIX_square_opt_weight']=t*GVIX_opt_weight
                        gvix_square=(t*opt_portfolio.GVIX_opt_weight*opt_portfolio.midpoint_price).sum()
                        gvix_square_payoff=(t*opt_portfolio.GVIX_opt_weight*opt_portfolio.payoff).sum()
                        portfolio_dict[id_counter][f'{i+1} month GVIX^2']=gvix_square
                        portfolio_dict[id_counter][f'{i+1} month GVIX^2 payoff']=gvix_square_payoff

                        opt_portfolio['GVIX_square_bid_contribtion']=opt_portfolio.apply(lambda x: x.GVIX_square_opt_weight*x.best_bid if x.GVIX_square_opt_weight >0 else x.GVIX_square_opt_weight*x.best_offer,axis=1)
                        opt_portfolio['GVIX_square_ask_contribtion']=opt_portfolio.apply(lambda x: x.GVIX_square_opt_weight*x.best_offer if x.GVIX_square_opt_weight >0 else x.GVIX_square_opt_weight*x.best_bid,axis=1)
                        portfolio_dict[id_counter][f'{i+1} month GVIX square bid']=np.exp(r*t)*opt_portfolio['GVIX_square_bid_contribtion'].sum()
                        portfolio_dict[id_counter][f'{i+1} month GVIX square ask']=np.exp(r*t)*opt_portfolio['GVIX_square_ask_contribtion'].sum()
                        portfolio_dict[id_counter][f'{i+1} month GVIX square spread ($)']=portfolio_dict[id_counter][f'{i+1} month GVIX square ask']-portfolio_dict[id_counter][f'{i+1} month GVIX square bid']

                        #CBOE SKEW (method 1)
                        opt_portfolio['meth1_opt_weight']=(p3_option_weight - 3*P1*p2_option_weight + 2*P1**2*p1_option_weight) * np.exp(r*t)/sigma**3
                        portfolio_dict[id_counter][f'{i+1} month CBOE SKEW']=100-10*(opt_portfolio.meth1_opt_weight*opt_portfolio.midpoint_price).sum()
                        portfolio_dict[id_counter][f'{i+1} month CBOE SKEW payoff']=100-10*(opt_portfolio.meth1_opt_weight*opt_portfolio.payoff).sum()
                        opt_portfolio['meth1_bid_contribtion']=opt_portfolio.apply(lambda x: x.meth1_opt_weight*x.best_bid if x.meth1_opt_weight >0 else x.meth1_opt_weight*x.best_offer,axis=1)
                        opt_portfolio['meth1_ask_contribtion']=opt_portfolio.apply(lambda x: x.meth1_opt_weight*x.best_offer if x.meth1_opt_weight >0 else x.meth1_opt_weight*x.best_bid,axis=1)
                        #100-10*ask - (100-10*bid) < 0 when ask>bid
                        portfolio_dict[id_counter][f'{i+1} month CBOE SKEW bid']=100-10*(opt_portfolio['meth1_bid_contribtion'].sum())
                        portfolio_dict[id_counter][f'{i+1} month CBOE SKEW ask']=100-10*(opt_portfolio['meth1_ask_contribtion'].sum())
                        portfolio_dict[id_counter][f'{i+1} month CBOE SKEW spread ($)']=portfolio_dict[id_counter][f'{i+1} month CBOE SKEW ask']-portfolio_dict[id_counter][f'{i+1} month CBOE SKEW bid']
                        portfolio_dict[id_counter][f'{i+1} month CBOE SKEW spread ($)']*=-1
                        #portfolio_dict[id_counter][f'{i+1} month CBOE SKEW bid']=opt_portfolio['meth1_bid_contribtion'].sum()
                        #portfolio_dict[id_counter][f'{i+1} month CBOE SKEW ask']=opt_portfolio['meth1_ask_contribtion'].sum()
                        #portfolio_dict[id_counter][f'{i+1} month CBOE SKEW spread ($)']=100-10*((opt_portfolio['meth1_ask_contribtion']-opt_portfolio['meth1_bid_contribtion']).sum())

                        #Method 2 SKEW
                        opt_portfolio['meth2_opt_weight']=(-6*p1_option_weight - 3*p2_option_weight) * np.exp(r*t)/sigma**3
                        portfolio_dict[id_counter][f'{i+1} month SKEW Method 2']=100-10*(opt_portfolio.meth2_opt_weight*opt_portfolio.midpoint_price).sum()
                        portfolio_dict[id_counter][f'{i+1} month SKEW Method 2 payoff']=100-10*(opt_portfolio.meth2_opt_weight*opt_portfolio.payoff).sum()
                        opt_portfolio['meth2_bid_contribtion']=opt_portfolio.apply(lambda x: x.meth2_opt_weight*x.best_bid if x.meth2_opt_weight >0 else x.meth2_opt_weight*x.best_offer,axis=1)
                        opt_portfolio['meth2_ask_contribtion']=opt_portfolio.apply(lambda x: x.meth2_opt_weight*x.best_offer if x.meth2_opt_weight >0 else x.meth2_opt_weight*x.best_bid,axis=1)
                        portfolio_dict[id_counter][f'{i+1} month SKEW Method 2 bid']=100-10*(opt_portfolio['meth2_bid_contribtion'].sum())
                        portfolio_dict[id_counter][f'{i+1} month SKEW Method 2 ask']=100-10*(opt_portfolio['meth2_ask_contribtion'].sum())
                        portfolio_dict[id_counter][f'{i+1} month SKEW Method 2 spread ($)']=portfolio_dict[id_counter][f'{i+1} month SKEW Method 2 ask']-portfolio_dict[id_counter][f'{i+1} month SKEW Method 2 bid']
                        portfolio_dict[id_counter][f'{i+1} month SKEW Method 2 spread ($)']*=-1
                        #portfolio_dict[id_counter][f'{i+1} month SKEW Method 2 spread ($)']=100-10*((opt_portfolio['meth2_ask_contribtion']-opt_portfolio['meth2_bid_contribtion']).sum())

                        #SSKEW (Method 3)
                        opt_portfolio['meth3_opt_weight']=(SVIX_opt_weight-VIX_opt_weight) * 3/2*np.exp(r*t)/sigma**3
                        opt_portfolio['meth3_opt_weight']=(opt_portfolio['SVIX_square_opt_weight']-opt_portfolio['VIX_square_opt_weight']) * 3/2*np.exp(r*t)/sigma**3
                        portfolio_dict[id_counter][f'{i+1} month SSKEW']=100-10*( (3/2)*np.exp(r*t)/(sigma**3)*(svix_square-vix_square) )
                        #portfolio_dict[id_counter][f'{i+1} month SSKEW']=100-10*(opt_portfolio.meth3_opt_weight*opt_portfolio.midpoint_price).sum()
                        portfolio_dict[id_counter][f'{i+1} month SSKEW payoff']=100-10*( (3/2)/(sigma**3)*(svix_square_payoff-vix_square_payoff) )
                        opt_portfolio['meth3_bid_contribtion']=opt_portfolio.apply(lambda x: x.meth3_opt_weight*x.best_bid if x.meth3_opt_weight >0 else x.meth3_opt_weight*x.best_offer,axis=1)
                        opt_portfolio['meth3_ask_contribtion']=opt_portfolio.apply(lambda x: x.meth3_opt_weight*x.best_offer if x.meth3_opt_weight >0 else x.meth3_opt_weight*x.best_bid,axis=1)
                        #portfolio_dict[id_counter][f'{i+1} month SSKEW bid']=(3/2)*np.exp(r*t)/(sigma**3)*(portfolio_dict[id_counter][f'{i+1} month SVIX square bid']-portfolio_dict[id_counter][f'{i+1} month VIX square bid'])
                        #portfolio_dict[id_counter][f'{i+1} month SSKEW ask']=(3/2)*np.exp(r*t)/(sigma**3)*(portfolio_dict[id_counter][f'{i+1} month SVIX square ask']-portfolio_dict[id_counter][f'{i+1} month VIX square ask'])
                        #portfolio_dict[id_counter][f'{i+1} month SSKEW spread ($)']=portfolio_dict[id_counter][f'{i+1} month SSKEW ask']-portfolio_dict[id_counter][f'{i+1} month SSKEW bid']
                        portfolio_dict[id_counter][f'{i+1} month SSKEW bid']=100-10*(opt_portfolio['meth3_bid_contribtion'].sum())
                        portfolio_dict[id_counter][f'{i+1} month SSKEW ask']=100-10*(opt_portfolio['meth3_ask_contribtion'].sum())
                        portfolio_dict[id_counter][f'{i+1} month SSKEW spread ($)']=portfolio_dict[id_counter][f'{i+1} month SSKEW ask']-portfolio_dict[id_counter][f'{i+1} month SSKEW bid']
                        portfolio_dict[id_counter][f'{i+1} month SSKEW spread ($)']*=-1


                        #KNS SKEW (Method 4)
                        kns_skew=3*(gvix_square-vix_square)
                        
                        kns_skew_payoff=3*(gvix_square_payoff-vix_square_payoff)
                        portfolio_dict[id_counter][f'{i+1} month SKEW_KNS']=kns_skew
                        portfolio_dict[id_counter][f'{i+1} month SKEW_KNS payoff']=kns_skew_payoff
                        #scaling by sigma
                        kns_skew/=vix_square**(3/2)
                        kns_skew=100-10*kns_skew
                        #scaling for bid_ask spread
                        portfolio_dict[id_counter][f'{i+1} month SKEW_KNS_scaled']=kns_skew

                        opt_portfolio['SKEW_KNS_opt_weight']=3*(t*opt_portfolio['GVIX_opt_weight']-opt_portfolio['VIX_square_opt_weight']) *np.exp(r*t)
                        opt_portfolio['SKEW_KNS_opt_weight']/=vix_square**(3/2)
                        opt_portfolio['SKEW_KNS_bid_contribtion']=opt_portfolio.apply(lambda x: x.SKEW_KNS_opt_weight*x.best_bid if x.SKEW_KNS_opt_weight >0 else x.SKEW_KNS_opt_weight*x.best_offer,axis=1)
                        opt_portfolio['SKEW_KNS_ask_contribtion']=opt_portfolio.apply(lambda x: x.SKEW_KNS_opt_weight*x.best_offer if x.SKEW_KNS_opt_weight >0 else x.SKEW_KNS_opt_weight*x.best_bid,axis=1)
                        portfolio_dict[id_counter][f'{i+1} month SKEW_KNS bid']=100-10*(opt_portfolio['SKEW_KNS_bid_contribtion'].sum())
                        portfolio_dict[id_counter][f'{i+1} month SKEW_KNS ask']=100-10*(opt_portfolio['SKEW_KNS_ask_contribtion'].sum())
                        portfolio_dict[id_counter][f'{i+1} month SKEW_KNS spread ($)']=portfolio_dict[id_counter][f'{i+1} month SKEW_KNS ask']-portfolio_dict[id_counter][f'{i+1} month SKEW_KNS bid']
                        portfolio_dict[id_counter][f'{i+1} month SKEW_KNS spread ($)']*=-1
                        

                        #Delta Hedging and Realized Moments

            
                        #daily_int_rate=np.exp(r*1/360)-1
                        hedge_df=sp500_df[(sp500_df.Date>=modified_trade_date)&(sp500_df.Date<term)]
                        hedge_df=hedge_df.copy()
                        #realized var error: this could be dropping a row no forward on last day
                        hedge_df['Index Return']=hedge_df['Index Value - Total Return']/hedge_df['Index Value - Total Return'].shift()-1
                        equal_ssr=(hedge_df['Index Return']**2).sum() #VIX^2
                        #display(hedge_df)
                        #inner merge is dropping a row here
                        #hedge_df=hedge_df.merge(df_fwd[df_fwd.expiration==term],left_on=['Date'],right_on=['date'])
                        hedge_df=hedge_df.merge(df_fwd[df_fwd.expiration==term],how='left',left_on=['Date'],right_on=['date'])
                        hedge_df.rename(columns={'forwardprice':'F'},inplace=True)
                        hedge_df.rename(columns={'Index Price - Close Daily':'S'},inplace=True)
                        #hedge_df.rename(columns={'Index Value - Total Return':'S'},inplace=True
                        hedge_df['days_to_maturity']=(term-hedge_df['Date']).dt.days
                        tau=hedge_df['days_to_maturity']/360
                        #hedge_df['weight_gamma']=np.exp(r*tau)*hedge_df['S']/hedge_df.iloc[0]['F']
                        #weighted_ssr_gvar=( hedge_df['weight_gamma']*(hedge_df['Index Return']**2) ).sum()
                        weight_gamma=np.exp(r*tau)*hedge_df['S']/hedge_df.iloc[0]['F']
                        weighted_ssr_gvar=(weight_gamma*(hedge_df['Index Return']**2) ).sum()
                        portfolio_dict[id_counter][f'Realized Variance {i+1} month (WSSR_Gamma)']=weighted_ssr_gvar
                        portfolio_dict[id_counter][f'Realized Skew {i+1} month (KNS)']=3*(weighted_ssr_gvar-equal_ssr)

                        hedge_df['pv($2)']=2*np.exp(-r*tau)
                        #hedge_df['days_passed']=(hedge_df['Date']-hedge_df['Date'].shift()).dt.days
                        
                        

                        #VIX^2 delta hedge
                        #new delta hedge
                        #claim_delta=2/hedge_df.iloc[0]['F']-hedge_df.iloc[0]['pv($2)']/hedge_df.iloc[0]['S']
                        claim_delta=2/hedge_df.iloc[0]['F']-hedge_df['pv($2)']/hedge_df.iloc[0]['S']
                        shares_traded=2/hedge_df.iloc[0]['F']-hedge_df.iloc[0]['pv($2)']/hedge_df['S']
                        shares_traded*=-1
                        net_delta=claim_delta+shares_traded
                        PandL=net_delta.shift()*(hedge_df['S']-hedge_df['S'].shift())
                        #PandL-=hedge_df['pv($2)']*(np.exp(r*hedge_df['days_passed']/360)-1)
                        fv_PandL=PandL*np.exp(r*tau)
                        delta_hedge_payoff=fv_PandL.sum()
                        portfolio_dict[id_counter][f'{i+1} month VIX^2 Delta Hedging Payoff']=delta_hedge_payoff
                        #VIX^2 Equal SSR
                        #equal_ssr=(hedge_df['Index Return']**2).sum()
                        #portfolio_dict[id_counter][f'Realized Variance {i+1} month (SSR) alt']=equal_ssr
                        portfolio_dict[id_counter][f'Realized Variance {i+1} month (SSR)']=equal_ssr

                        
                        #SVIX^2 delta hedge
                        #new delta hedge
                        #sigma_sq_guess=vix_square
                        sigma_sq_guess=np.exp(r*t)*(opt_portfolio.VIX_opt_weight*opt_portfolio.midpoint_price).sum()
                        #claim_delta=2*np.exp((r+sigma_sq_guess)*hedge_df.iloc[0]['days_to_maturity']/360)*hedge_df.iloc[0]['S']/(hedge_df.iloc[0]['F']**2)
                        claim_delta=2*np.exp((r+sigma_sq_guess)*tau)*hedge_df.iloc[0]['S']/(hedge_df.iloc[0]['F']**2)
                        shares_traded=2*np.exp((r+sigma_sq_guess)*tau)*hedge_df['S']/(hedge_df.iloc[0]['F']**2)
                        shares_traded*=-1
                        net_delta=claim_delta+shares_traded
                        PandL=net_delta.shift()*(hedge_df['S']-hedge_df['S'].shift())
                        #PandL-=(2*np.exp((r+sigma_sq_guess)*tau)/(hedge_df.iloc[0]['F']**2))*(np.exp(r*hedge_df['days_passed']/360)-1)
                        fv_PandL=PandL*np.exp(r*tau)
                        delta_hedge_payoff=fv_PandL.sum()
                        portfolio_dict[id_counter][f'{i+1} month SVIX^2 Delta Hedging Payoff']=delta_hedge_payoff
                        #SVIX^2 return
                        #hedge_df['gamma']=factor_/2*np.exp(r*tau)*(hedge_df['S']-hedge_df['S'].shift())**2
                        #weighted_ssr=( hedge_df['gamma']*(hedge_df['Index Return']**2) ).sum()
                    
                    
                        hedge_df['weight']=np.exp(r*tau)*(hedge_df['S']/hedge_df.iloc[0]['F'])**2 * np.exp((r+sigma_sq_guess)*tau)
                        weighted_ssr=( hedge_df['weight']*(hedge_df['Index Return']**2) ).sum()
                        portfolio_dict[id_counter][f'Realized Variance {i+1} month (WSSR)']=weighted_ssr
                        #portfolio_dict[id_counter][f'{i} month Realized Weighted Variance Return (%)']=(weighted_ssr/portfolio_dict[id_counter][f'{i} month SVIX^2']-1)*100
                        #portfolio_dict[id_counter][f'{i} month SVIX^2 Delta Hedged return(%)']=((portfolio_dict[id_counter][f'{i} month SVIX^2 payoff']+monthly_portfolios[f'{i} month VIX^2 Delta Hedging Payoff'])/monthly_portfolios[f'{i} month VIX^2']-1)*100
                        #monthly_portfolios[f'{i} month VIX^2 Payoff w/ Delta Hedging ($)']=monthly_portfolios[f'{i} month VIX^2 payoff']+monthly_portfolios[f'{i} month VIX^2 Delta Hedging Payoff']
                        
                        
                        
                        #Entropy Contract Payoff
                        sigma_sq_guess=vix_square
                        claim_delta=2/hedge_df.iloc[0]['F']*(np.log(hedge_df.iloc[0]['S']/hedge_df.iloc[0]['F'])+1+(r+0.5*sigma_sq_guess)*hedge_df.iloc[0]['days_to_maturity']/360)
                        shares_traded=2/hedge_df.iloc[0]['F']*(np.log(hedge_df['S']/hedge_df.iloc[0]['F'])+1+(r+0.5*sigma_sq_guess)*tau)
                        shares_traded*=-1
                        net_delta=claim_delta+shares_traded
                        PandL=net_delta.shift()*(hedge_df['S']-hedge_df['S'].shift())
                        #PandL-=(2*np.exp((r+sigma_sq_guess)*tau)/(hedge_df.iloc[0]['F']**2))*(np.exp(r*hedge_df['days_passed']/360)-1)
                        fv_PandL=PandL*np.exp(r*tau)
                        delta_hedge_payoff=fv_PandL.sum()
                        portfolio_dict[id_counter][f'{i+1} month GVIX^2 Delta Hedging Payoff']=delta_hedge_payoff
                        
                        #moving to earlier
                        #hedge_df['weight_gamma']=np.exp(r*tau)*hedge_df['S']/hedge_df.iloc[0]['F']
                        #weighted_ssr=( hedge_df['weight_gamma']*(hedge_df['Index Return']**2) ).sum()
                        #portfolio_dict[id_counter][f'Realized Variance {i+1} month (WSSR_Gamma)']=weighted_ssr_gvar

                        
                        
                        portfolio_dict[id_counter][f'{i+1} month SKEW_KNS Delta Hedging Payoff']=3*(portfolio_dict[id_counter][f'{i+1} month GVIX^2 Delta Hedging Payoff']-portfolio_dict[id_counter][f'{i+1} month VIX^2 Delta Hedging Payoff'])
                        


                        portfolio_dict[id_counter][f'{i+1} month VIX return (%)']=(portfolio_dict[id_counter][f'{i+1} month VIX payoff']/portfolio_dict[id_counter][f'{i+1} month VIX']-1)*100
                        portfolio_dict[id_counter][f'{i+1} month VIX^2 return (%)']=(portfolio_dict[id_counter][f'{i+1} month VIX^2 payoff']/portfolio_dict[id_counter][f'{i+1} month VIX^2']-1)*100
                        #portfolio_dict[id_counter][f'{i} month VIX^2 Delta Hedged return(%)']=portfolio_dict[id_counter][f'{i} month VIX^2 return (%)']+monthly_portfolios['1 month VIX^2 Delta Hedging Payoff']
                        #portfolio_dict[id_counter][f'{i} month VIX^2 Delta Hedged return(%)']=((portfolio_dict[id_counter][f'{i} month VIX^2 payoff']+monthly_portfolios[f'{i} month VIX^2 Delta Hedging Payoff'])/monthly_portfolios[f'{i} month VIX^2']-1)*100
                        portfolio_dict[id_counter][f'{i+1} month VIX^2 Delta Hedged return(%)']=((portfolio_dict[id_counter][f'{i+1} month VIX^2 payoff']+portfolio_dict[id_counter][f'{i+1} month VIX^2 Delta Hedging Payoff'])/portfolio_dict[id_counter][f'{i+1} month VIX^2']-1)*100
                        portfolio_dict[id_counter][f'{i+1} month VIX^2 Payoff w/ Delta Hedging ($)']=portfolio_dict[id_counter][f'{i+1} month VIX^2 payoff']+portfolio_dict[id_counter][f'{i+1} month VIX^2 Delta Hedging Payoff']
                        
                        portfolio_dict[id_counter][f'{i+1} month SVIX^2 return (%)']=(portfolio_dict[id_counter][f'{i+1} month SVIX^2 payoff']/portfolio_dict[id_counter][f'{i+1} month SVIX^2']-1)*100
                        #portfolio_dict[id_counter][f'{i} month Realized Weighted Variance Return (%)']=(weighted_ssr/portfolio_dict[id_counter][f'{i} month SVIX^2']-1)*100
                        portfolio_dict[id_counter][f'{i+1} month Realized Weighted Variance Return (%)']=(portfolio_dict[id_counter][f'Realized Variance {i+1} month (WSSR)']/portfolio_dict[id_counter][f'{i+1} month SVIX^2']-1)*100
                        portfolio_dict[id_counter][f'{i+1} month SVIX^2 Delta Hedged return(%)']=((portfolio_dict[id_counter][f'{i+1} month SVIX^2 payoff']+portfolio_dict[id_counter][f'{i+1} month SVIX^2 Delta Hedging Payoff'])/portfolio_dict[id_counter][f'{i+1} month SVIX^2']-1)*100
                        portfolio_dict[id_counter][f'{i+1} month SVIX^2 Payoff w/ Delta Hedging ($)']=portfolio_dict[id_counter][f'{i+1} month SVIX^2 payoff']+portfolio_dict[id_counter][f'{i+1} month SVIX^2 Delta Hedging Payoff']

                        portfolio_dict[id_counter][f'{i+1} month GVIX^2 Delta Hedged return(%)']=((portfolio_dict[id_counter][f'{i+1} month GVIX^2 payoff']+portfolio_dict[id_counter][f'{i+1} month GVIX^2 Delta Hedging Payoff'])/portfolio_dict[id_counter][f'{i+1} month GVIX^2']-1)*100
                        portfolio_dict[id_counter][f'{i+1} month GVIX^2 Payoff w/ Delta Hedging ($)']=portfolio_dict[id_counter][f'{i+1} month GVIX^2 payoff']+portfolio_dict[id_counter][f'{i+1} month GVIX^2 Delta Hedging Payoff']
                        portfolio_dict[id_counter][f'{i+1} month Realized Weighted_Gamma Variance Return (%)']=(portfolio_dict[id_counter][f'Realized Variance {i+1} month (WSSR_Gamma)']/portfolio_dict[id_counter][f'{i+1} month GVIX^2']-1)*100

                        portfolio_dict[id_counter][f'{i+1} month SVIX return (%)']=(portfolio_dict[id_counter][f'{i+1} month SVIX payoff']/portfolio_dict[id_counter][f'{i+1} month SVIX']-1)*100
                        portfolio_dict[id_counter][f'{i+1} month CBOE SKEW return (%)']=(portfolio_dict[id_counter][f'{i+1} month CBOE SKEW payoff']/portfolio_dict[id_counter][f'{i+1} month CBOE SKEW']-1)*100
                        portfolio_dict[id_counter][f'{i+1} month SKEW Method 2 return (%)']=(portfolio_dict[id_counter][f'{i+1} month SKEW Method 2 payoff']/portfolio_dict[id_counter][f'{i+1} month SKEW Method 2']-1)*100
                        portfolio_dict[id_counter][f'{i+1} month SSKEW return (%)']=(portfolio_dict[id_counter][f'{i+1} month SSKEW payoff']/portfolio_dict[id_counter][f'{i+1} month SSKEW']-1)*100
                        portfolio_dict[id_counter][f'{i+1} month GVIX^2 return (%)']=(portfolio_dict[id_counter][f'{i+1} month GVIX^2 payoff']/portfolio_dict[id_counter][f'{i+1} month GVIX^2']-1)*100

                        #KNS SKEW
                        portfolio_dict[id_counter][f'{i+1} month SKEW_KNS return (%)']=(portfolio_dict[id_counter][f'{i+1} month SKEW_KNS payoff']/portfolio_dict[id_counter][f'{i+1} month SKEW_KNS']-1)*100
                        portfolio_dict[id_counter][f'{i+1} month SKEW_KNS Delta Hedged return(%)']=((portfolio_dict[id_counter][f'{i+1} month SKEW_KNS payoff']+portfolio_dict[id_counter][f'{i+1} month SKEW_KNS Delta Hedging Payoff'])/portfolio_dict[id_counter][f'{i+1} month SKEW_KNS']-1)*100
                        portfolio_dict[id_counter][f'{i+1} month SKEW_KNS Payoff w/ Delta Hedging ($)']=portfolio_dict[id_counter][f'{i+1} month SKEW_KNS payoff']+portfolio_dict[id_counter][f'{i+1} month SKEW_KNS Delta Hedging Payoff']
                        portfolio_dict[id_counter][f'{i+1} month Realized Weighted Skew Return (%)']=(portfolio_dict[id_counter][f'Realized Skew {i+1} month (KNS)']/portfolio_dict[id_counter][f'{i+1} month SKEW_KNS']-1)*100


                        #portfolio_dict[id_counter][f'{i+1} month Realized Variance Return (%) alt']=(portfolio_dict[id_counter][f'Realized Variance {i+1} month (SSR) alt']/portfolio_dict[id_counter][f'{i+1} month VIX^2']-1)*100
                        portfolio_dict[id_counter][f'{i+1} month Realized Variance Return (%)']=(portfolio_dict[id_counter][f'Realized Variance {i+1} month (SSR)']/portfolio_dict[id_counter][f'{i+1} month VIX^2']-1)*100



                        
                        #misc things to add to option portfolios
                        opt_portfolio['sigma']=sigma
                        opt_portfolio['sigma_other']=vix_square**(3/2)
                        opt_portfolio['P1']=P1
                        maturity_id=i+1
                        opt_portfolio['portfolio_id']=id_counter
                        opt_portfolio['maturity_id']=maturity_id
                        opt_portfolio['trade_date']=expir
                        opt_portfolio['modified_trade_date']=modified_trade_date

                        #countin number of options and range of moneyness
                        portfolio_dict[id_counter][f'{i+1} month number of options']=opt_portfolio.shape[0]+1
                        portfolio_dict[id_counter][f'{i+1} month max moneyness']=opt_portfolio.strike_price.max()/F
                        portfolio_dict[id_counter][f'{i+1} month min moneyness']=opt_portfolio.strike_price.min()/F
                        portfolio_dict[id_counter][f'{i+1} month market gross return']=settle_value/F
                        portfolio_dict[id_counter][f'{i+1} month F']=F
                        portfolio_dict[id_counter][f'{i+1} month S_T']=settle_value
                        


                        ###########FOR DOUBLE CHECK###########
                        """
                        if expir==pd.Timestamp('2023-06-16 00:00:00'):
                            opt_portfolio['interest_factor']=np.exp(r*t)
                            opt_portfolio['S_T']=settle_value
                            opt_portfolio['F']=F
                            opt_portfolio['t']=t
                            opt_portfolio.to_csv(f'{i+1}month_option_portfolio_{expir}.csv')
                        if expir==pd.Timestamp('2023-08-18 00:00:00'):
                            opt_portfolio['interest_factor']=np.exp(r*t)
                            opt_portfolio['S_T']=settle_value
                            opt_portfolio['F']=F
                            opt_portfolio['t']=t
                            opt_portfolio.to_csv(f'{i+1}month_option_portfolio_{expir}.csv')
                        """
                        #######################################
                        misc_cols=['sigma','sigma_other','P1','portfolio_id','maturity_id','trade_date','modified_trade_date']

                        weight_cols=['VIX_opt_weight','VIX_square_opt_weight','VIX_square_bid_contribtion','VIX_square_ask_contribtion',
                                    'SVIX_opt_weight','SVIX_square_opt_weight','SVIX_square_bid_contribtion','SVIX_square_ask_contribtion',
                                    'GVIX_square_opt_weight','GVIX_square_bid_contribtion','GVIX_square_ask_contribtion',
                                    'meth1_opt_weight','meth1_bid_contribtion','meth1_ask_contribtion',
                                    'meth2_opt_weight','meth2_bid_contribtion','meth2_ask_contribtion',
                                    'meth3_opt_weight','meth3_bid_contribtion','meth3_ask_contribtion',
                                    'SKEW_KNS_opt_weight','SKEW_KNS_bid_contribtion','SKEW_KNS_ask_contribtion']

                        df_K0=df_K0[['strike_price','cp_flag','midpoint_price','best_bid','best_offer','optionid']]
                        df_K0['dK']=opt_portfolio[opt_portfolio.cp_flag=='P/C Avg'].dK.values[0]
                        for col in misc_cols:
                            df_K0[f'{col}']=opt_portfolio[opt_portfolio.cp_flag=='P/C Avg'][f'{col}'].values[0]

                        #df_K0['delta']=opt_portfolio[opt_portfolio.cp_flag=='P/C Avg'].delta.values[0]*1/2
                        #split the weight 50-50 for the ATM call and put
                        for col in weight_cols:
                            df_K0[f'{col}']=opt_portfolio[opt_portfolio.cp_flag=='P/C Avg'][f'{col}'].values[0]*1/2
                        #df_K0['VIX_opt_weight']=opt_portfolio[opt_portfolio.cp_flag=='P/C Avg'].VIX_opt_weight.values[0]*1/2
                        #df_K0['SVIX_opt_weight']=opt_portfolio[opt_portfolio.cp_flag=='P/C Avg'].SVIX_opt_weight.values[0]*1/2
                        #df_K0['meth1_opt_weight']=opt_portfolio[opt_portfolio.cp_flag=='P/C Avg'].meth1_opt_weight.values[0]*1/2
                        #df_K0['meth2_opt_weight']=opt_portfolio[opt_portfolio.cp_flag=='P/C Avg'].meth2_opt_weight.values[0]*1/2
                        #df_K0['meth3_opt_weight']=opt_portfolio[opt_portfolio.cp_flag=='P/C Avg'].meth3_opt_weight.values[0]*1/2
                        opt_portfolio=opt_portfolio[opt_portfolio.cp_flag!='P/C Avg']

                        opt_portfolio_2=pd.concat([opt_portfolio,df_K0])
                        #opt_portfolio_2=pd.concat([opt_portfolio,df_K0])[['strike_price',
                        #                                                    'cp_flag',
                        #                                                    'dK',
                        #                                                    'midpoint_price',
                        #                                                    'best_bid','best_offer',
                        #                                                    'optionid',
                        #                                                    'VIX_opt_weight',
                        #                                                    'SVIX_opt_weight',
                        #                                                    'meth1_opt_weight',
                        #                                                    'meth2_opt_weight',
                        #                                                    'meth3_opt_weight']]
                        opt_portfolio_2.sort_values(by=['strike_price','cp_flag'],ascending=[True,False],inplace=True)
                        #display(opt_portfolio_2)

                        if i+1==3:
                            liquidation_date=expirs_found[i-1]
                            #print(f'{i} month liquidation_date:{liquidation_date}')
                            #print(opt_portfolio.shape)
                            #print(opt_portfolio_2.shape)
                            #print('--------------------')
                            #print(opt_portfolio_2.optionid.size)
                            #print( len(set(opt_portfolio_2.optionid.values.tolist())) )
                            liquidation_values=option_liquidation_query(ticker,
                                                                        opt_portfolio_2.optionid.dropna().values.tolist(),
                                                                        liquidation_date)
                            #display(liquidation_values)
                            opt_portfolio_2=opt_portfolio_2.merge(liquidation_values,how='left',on=['optionid'])
                            opt_portfolio_2.rename(columns={'liquidation_midprice_date':f'{i+1}M port liquidation_midprice_date'},inplace=True)
                            #display(opt_portfolio_2[pd.isnull(opt_portfolio_2[f'{i+1}M port liquidation_midprice_date'])])
                            vix_square_liquidation_value=(t*opt_portfolio_2.VIX_opt_weight*opt_portfolio_2.liquidation_midprice).sum()
                            portfolio_dict[id_counter][f'{i+1} month VIX^2 liquidation']=vix_square_liquidation_value
                            gvix_square_liquidation_value=(t*opt_portfolio_2.GVIX_opt_weight*opt_portfolio_2.liquidation_midprice).sum()
                            portfolio_dict[id_counter][f'{i+1} month GVIX^2 liquidation']=gvix_square_liquidation_value

                        elif i+1==2:
                            #pass
                            liquidation_date=expirs_found[i-1]
                            #print(f'{i} month liquidation_date:{liquidation_date}')
                            #print(opt_portfolio.shape)
                            #print(opt_portfolio_2.shape)
                            #print('--------------------')
                            #print(opt_portfolio_2.optionid.size)
                            #print( len(set(opt_portfolio_2.optionid.values.tolist())) )
                            liquidation_values=option_liquidation_query(ticker,
                                                opt_portfolio_2.optionid.dropna().values.tolist(),
                                                liquidation_date)
                            #display(liquidation_values)
                            opt_portfolio_2=opt_portfolio_2.merge(liquidation_values,how='left',on=['optionid'])
                            opt_portfolio_2.rename(columns={'liquidation_midprice_date':f'{i+1}M port liquidation_midprice_date'},inplace=True)
                            #display(opt_portfolio_2[pd.isnull(opt_portfolio_2[f'{i+1}M port liquidation_midprice_date'])])
                            vix_square_liquidation_value=(t*opt_portfolio_2.VIX_opt_weight*opt_portfolio_2.liquidation_midprice).sum()
                            portfolio_dict[id_counter][f'{i+1} month VIX^2 liquidation']=vix_square_liquidation_value
                            gvix_square_liquidation_value=(t*opt_portfolio_2.GVIX_opt_weight*opt_portfolio_2.liquidation_midprice).sum()
                            portfolio_dict[id_counter][f'{i+1} month GVIX^2 liquidation']=gvix_square_liquidation_value

                        ###########FOR DOUBLE CHECK###########
                        #opt_portfolio['interest_factor']=np.exp(r*t)
                        #opt_portfolio['S_T']=settle_value
                        """
                        if expir==pd.Timestamp('2023-06-16 00:00:00'):
                            opt_portfolio_2.to_csv(f'{i+1}month_option_portfolio_2_{expir}.csv')
                        if expir==pd.Timestamp('2023-08-18 00:00:00'):
                            opt_portfolio_2.to_csv(f'{i+1}month_option_portfolio_2_{expir}.csv')
                        """
                        #######################################


                        option_portfolios.append(opt_portfolio)
                        #option_portfolios.append(opt_portfolio_2)

                        #print(6/0)

                        


                    id_counter+=1            

            
            #next_expir = pd.Series(slice_.exdate.unique()).nsmallest(3)
            #portfolio_dict[id_counter]={'Trade Date':expir}
            #portfolio_dict[id_counter]['modified_trade_date']=modified_trade_date

            #print('----------------------------------')
    
    option_portfolios_df=pd.concat(option_portfolios)

    monthly_portfolios=pd.DataFrame.from_dict(portfolio_dict).T.dropna(subset=['1 month expiration'])
    monthly_portfolios=monthly_portfolios[monthly_portfolios['Trade Date']<monthly_portfolios['1 month expiration']]
    monthly_portfolios['1 month DTE']=monthly_portfolios['1 month expiration']-monthly_portfolios['modified_trade_date']
    monthly_portfolios['2 month DTE']=monthly_portfolios['2 month expiration']-monthly_portfolios['modified_trade_date']
    monthly_portfolios['3 month DTE']=monthly_portfolios['3 month expiration']-monthly_portfolios['modified_trade_date']
    monthly_portfolios['1 month DTE']=monthly_portfolios['1 month DTE'].apply(lambda x:x.days)
    monthly_portfolios['2 month DTE']=monthly_portfolios['2 month DTE'].apply(lambda x:x.days)
    monthly_portfolios['3 month DTE']=monthly_portfolios['3 month DTE'].apply(lambda x:x.days)
    monthly_portfolios['Trade Date']=pd.to_datetime(monthly_portfolios['Trade Date'])
    monthly_portfolios['modified_trade_date']=pd.to_datetime(monthly_portfolios['modified_trade_date'])#.dt.date
    monthly_portfolios['1 month expiration']=pd.to_datetime(monthly_portfolios['1 month expiration'])#.dt.date
    monthly_portfolios['2 month expiration']=pd.to_datetime(monthly_portfolios['2 month expiration'])#.dt.date
    monthly_portfolios['3 month expiration']=pd.to_datetime(monthly_portfolios['3 month expiration'])#.dt.date
    warnings.resetwarnings()
    #return portfolio_dict, option_portfolios
    return monthly_portfolios, option_portfolios_df


def compute_correlation_matrices(df):
    correlation_dfs=[]
    cols=['VIX^2 Delta Hedged return(%)','Realized Variance Return (%)','GVIX^2 Delta Hedged return(%)','Realized Weighted_Gamma Variance Return (%)','SKEW_KNS Delta Hedged return(%)','Realized Weighted Skew Return (%)']
    for i in range(1,3+1):
        for col in cols:
            df[f'{i} month {col}']=df[f'{i} month {col}'].astype(float)
            corr_mtx=df[[f'{i} month {col}' for col in cols]].corr()
            corr_mtx.to_csv(f'outputs/returns_correlations/{i} month returns correlation.csv')
            #correlation_dfs.append(corr_mtx)
    #return correlation_dfs

def skew_payoff_graphs(option_portfolios_df,maturity_months=1):
    rg = np.random.default_rng(1234)
    #rg = np.random.default_rng(None)
    trade_dt = rg.choice(option_portfolios_df.modified_trade_date.unique(), size=1, replace=False)[0]
    option_portfolios_df_2=option_portfolios_df[(option_portfolios_df.modified_trade_date==trade_dt) 
                                            & (option_portfolios_df.maturity_id==maturity_months)]
    
    F=option_portfolios_df_2.forwardprice.values[0]
    skew_methods=['meth1','meth2','meth3','SKEW_KNS']

    def call_payoff(s,k):
        #return max(s-k,0)
        return np.maximum(s-k,0)
    def put_payoff(s,k):
        #return max(k-s,0)
        return np.maximum(k-s,0)


    def skew_payoff_func(method):
        def portfolio_payoff(S_T):
            def option_payoff(row):
                k=row['strike_price']
                opt_type=row['cp_flag']
                if opt_type=='C':
                    return call_payoff(S_T,k) 
                elif opt_type=='P':
                    return put_payoff(S_T,k)
                else:
                    return call_payoff(S_T,k)+put_payoff(S_T,k)

            option_payoffs=option_portfolios_df_2.apply(option_payoff,axis=1)
            #option_weights=option_portfolios_df_2.SKEW_KNS_opt_weight
            option_weights=option_portfolios_df_2[f'{method}_opt_weight']
            return option_payoffs@option_weights

        vec_payoff = np.vectorize( portfolio_payoff )
        """
        S_T=np.linspace(option_portfolios_df_2.forwardprice.values[0]/1.5,
                        option_portfolios_df_2.forwardprice.values[0]*1.5,
                        500)
        """
        #"""
        S_T=np.linspace(option_portfolios_df_2.strike_price.min(),
                        option_portfolios_df_2.strike_price.max(),
                        500)
        #"""
        return S_T, portfolio_payoff(S_T)

                
    for method in skew_methods:
        S_T, skew_payoff = skew_payoff_func(method)
        plt.plot(S_T,skew_payoff,label=f'{method}')
    plt.axvline(x=F, 
                color='black', linestyle='--', 
                ymin=min(skew_payoff), ymax=max(skew_payoff))
    #plt.xlim(None,option_portfolios_df_2.forwardprice.values[0])
    #plt.xlim(option_portfolios_df_2.forwardprice.values[0],None)
    #plt.plot(S_T,np.log(S_T)**3,label=r'$\log(S)^3$')
    #plt.plot(S_T,(S_T/option_portfolios_df_2.forwardprice.values[0]-1)**3,label=r'$(S_T-1)^3$')
    plt.title('SKEW Portfolio Payoffs')
    plt.xlabel(r'$f(S_T)$')
    plt.xlabel(r'$S_T$')
    plt.legend()
    #plt.show()
    plt.savefig("outputs/skew payoffs/skew_payoff_curves.png", bbox_inches='tight')
    plt.close()


    method='meth1'
    S_T, skew_payoff = skew_payoff_func(method)
    plt.plot(S_T,skew_payoff,label=f'{method}',alpha=0.8)
    plt.plot(S_T,(np.log(S_T/F)-option_portfolios_df_2.P1.values[0])**3/option_portfolios_df_2.sigma.values[0]**3,
            label=r'$(\log(S_T/F)-P_1)^3/\sigma^3$',alpha=0.6)
    plt.axvline(x=F, color='black', linestyle='--', ymin=min(skew_payoff), ymax=max(skew_payoff))
    plt.legend()
    #plt.show()
    plt.savefig("outputs/skew payoffs/CBOE Skew Payoff.png", bbox_inches='tight')
    plt.close()

    method='meth2'
    S_T, skew_payoff = skew_payoff_func(method)
    plt.plot(S_T,skew_payoff,label=f'{method}',alpha=0.8)
    plt.plot(S_T,(6*(S_T/F-1)-6*np.log(S_T/F)-3*np.log(S_T/F)**2)/option_portfolios_df_2.sigma.values[0]**3,
            label=r'$(6(S_T/F-1)-6\log(S_T/F)-3\log^2(S_T/F))/\sigma^3$',alpha=0.6)
    plt.axvline(x=F, color='black', linestyle='--', ymin=min(skew_payoff), ymax=max(skew_payoff))
    plt.legend()
    #plt.show()
    plt.savefig("outputs/skew payoffs/Method2 Skew Payoff.png", bbox_inches='tight')
    plt.close()

    method='meth3'
    S_T, skew_payoff = skew_payoff_func(method)
    plt.plot(S_T,skew_payoff,label=f'{method}',alpha=0.8)
    plt.plot(S_T,3/2*( (S_T/F-1)**2+2*np.log(S_T/F)-2*S_T/F+2 )/option_portfolios_df_2.sigma.values[0]**3,
            label=r'$3/2 ((S_T/F-1)^2+2\log(S_T/F)-2S_T/F+2)/\sigma^3$',alpha=0.6)
    plt.axvline(x=F, color='black', linestyle='--', ymin=min(skew_payoff), ymax=max(skew_payoff))
    plt.legend()
    #plt.show()
    plt.savefig("outputs/skew payoffs/SSkew Payoff.png", bbox_inches='tight')
    plt.close()

    method='SKEW_KNS'
    S_T, skew_payoff = skew_payoff_func(method)
    plt.plot(S_T,skew_payoff,label=f'{method}',alpha=0.8)
    #plt.plot(S_T,3*( 2*(S_T/F*np.log(S_T/F)-S_T/F-1)-2*(-np.log(S_T/F)+(S_T/F-1)) )/option_portfolios_df_2.sigma.values[0]**3, label=r'$3 (a)/\sigma^3$',alpha=0.6)
    #vg_payoff=S_T/F*np.log(S_T/F)-S_T/F-1
    vg_payoff=S_T/F*np.log(S_T/F)-(S_T/F-1)
    vg_payoff*=2
    vl_payoff=(S_T/F-1)-np.log(S_T/F)
    vl_payoff*=2
    denom=option_portfolios_df_2.sigma.values[0]**3
    #denom=option_portfolios_df_2.sigma_other.values[0]**3
    plt.plot(S_T,3*( vg_payoff-vl_payoff )/denom, 
            label=r'$3\cdot 2(S_T/F \log(S_T/F)-(S_T/F-1)-(S_T/F-1-\log(S_T/F))) /\sigma^3$',alpha=0.6)
    plt.axvline(x=F, color='black', linestyle='--', ymin=min(skew_payoff), ymax=max(skew_payoff))
    plt.legend()
    #plt.show()
    plt.savefig("outputs/skew payoffs/KNS Skew Payoff.png", bbox_inches='tight')
    plt.close()


def var_payoff_graphs(option_portfolios_df,maturity_months=1):
    rg = np.random.default_rng(1234)
    #rg = np.random.default_rng(None)
    trade_dt = rg.choice(option_portfolios_df.modified_trade_date.unique(), size=1, replace=False)[0]
    option_portfolios_df_2=option_portfolios_df[(option_portfolios_df.modified_trade_date==trade_dt) 
                                            & (option_portfolios_df.maturity_id==maturity_months)]

    F=option_portfolios_df_2.forwardprice.values[0]
    def call_payoff(s,k):
        #return max(s-k,0)
        return np.maximum(s-k,0)
    def put_payoff(s,k):
        #return max(k-s,0)
        return np.maximum(k-s,0)

    def var_payoff_func(method):
        def portfolio_payoff(S_T):
            def option_payoff(row):
                k=row['strike_price']
                opt_type=row['cp_flag']
                if opt_type=='C':
                    return call_payoff(S_T,k) 
                elif opt_type=='P':
                    return put_payoff(S_T,k)
                else:
                    return call_payoff(S_T,k)+put_payoff(S_T,k)

            option_payoffs=option_portfolios_df_2.apply(option_payoff,axis=1)
            #option_weights=option_portfolios_df_2.SKEW_KNS_opt_weight
            option_weights=option_portfolios_df_2[f'{method}_opt_weight']
            return option_payoffs@option_weights

        vec_payoff = np.vectorize( portfolio_payoff )
        """
        S_T=np.linspace(option_portfolios_df_2.forwardprice.values[0]/1.5,
                        option_portfolios_df_2.forwardprice.values[0]*1.5,
                        500)
        """
        #"""
        S_T=np.linspace(option_portfolios_df_2.strike_price.min(),
                        option_portfolios_df_2.strike_price.max(),
                        500)
        #"""
        return S_T, portfolio_payoff(S_T)

        
    #F=option_portfolios_df_2.forwardprice.values[0]

    method='VIX_square'
    S_T, vix_sq_payoff = var_payoff_func(method)
    plt.plot(S_T,vix_sq_payoff,label=f'{method}',alpha=0.8)
    plt.plot(S_T,2*(S_T/F-1-np.log(S_T/F)),
            label=r'$-2log(S_T/F)+2(S_T/F-1)$',alpha=0.6)
    plt.axvline(x=F, color='black', linestyle='--', ymin=min(vix_sq_payoff), ymax=max(vix_sq_payoff))
    plt.legend()
    #plt.show()
    plt.savefig("outputs/vix payoffs/VIX square Payoff.png", bbox_inches='tight')
    plt.close()


def plot_bid_ask_spread(monthly_portfolios):

    #SKEW BID-ASK
    skew_methods=['CBOE SKEW','SKEW Method 2','SSKEW','SKEW_KNS']
    #skew_methods=['CBOE SKEW','SKEW_KNS']

    fig, ax = plt.subplots( figsize=(12, 12))

    for i in range(1,3+1):
        for method in skew_methods:
            #bid-ask $
            plt.plot(pd.to_datetime(monthly_portfolios.modified_trade_date),
                    monthly_portfolios[f'{i} month {method} spread ($)'],
                    label=f'{method}',)
            
        plt.axhline(y=0, color='r', linestyle='--') 
        plt.title(f'{i} month SKEW indexes')
        #plt.xlabel('$t$')
        plt.ylabel('($) Bid-Ask Spread')
        #ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.legend()
        #plt.show()
        plt.savefig(f"outputs/bid ask spreads/{i} month Skew $ Bid-Ask Spreads.png", bbox_inches='tight')
        plt.close()


    for i in range(1,3+1):
        for method in skew_methods:
            #bid-ask %
            if method=='SKEW_KNS':
                #portfolio_dict[id_counter][f'{i+1} month SKEW_KNS_scaled']
                plt.plot(pd.to_datetime(monthly_portfolios.modified_trade_date),
                monthly_portfolios[f'{i} month {method} spread ($)']/monthly_portfolios[f'{i} month {method}_scaled']*100,
                label=f'{method}',)
            
            else:
                plt.plot(pd.to_datetime(monthly_portfolios.modified_trade_date),
                monthly_portfolios[f'{i} month {method} spread ($)']/monthly_portfolios[f'{i} month {method}']*100,
                label=f'{method}',)
            
        plt.axhline(y=0, color='r', linestyle='--') 
        plt.title(f'{i} month SKEW indexes')
        #plt.xlabel('$t$')
        plt.ylabel('(%) Bid-Ask Spread')
        #ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.legend()
        #plt.show()
        plt.savefig(f"outputs/bid ask spreads/{i} month Skew % Bid-Ask Spreads.png", bbox_inches='tight')
        plt.close()


    for i in range(1,3+1):
        #bid-ask %
        plt.plot(pd.to_datetime(monthly_portfolios.modified_trade_date),
        monthly_portfolios[f'{i} month SKEW_KNS spread ($)']/monthly_portfolios[f'{i} month CBOE SKEW spread ($)'],
        label=f'{method}',)
            
        plt.axhline(y=1, color='r', linestyle='--') 
        plt.title(f'{i} month SKEW indexes')
        #plt.xlabel('$t$')
        plt.ylabel('Relative Bid-Ask Spread (KNS/CBOE)')
        #ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        #plt.legend()
        #plt.show()
        plt.savefig(f"outputs/bid ask spreads/{i} month Relative Spreads.png", bbox_inches='tight')
        plt.close()


def index_time_series(monthly_portfolios):

    skew_methods=['CBOE SKEW','SKEW Method 2','SSKEW','SKEW_KNS']
    #skew_methods=['CBOE SKEW','SKEW_KNS']

    fig, ax = plt.subplots( figsize=(12, 12))

    for i in range(1,3+1):
        #print(f'{i} month')
        #for method in ['CBOE SKEW','SKEW Method 2','SSKEW']:
        for method in skew_methods:
            plt.plot(pd.to_datetime(monthly_portfolios.modified_trade_date),
                    monthly_portfolios[f'{i} month {method}'],
                    label=f'{method}',)
        #plt.axhline(y=0, color='r', linestyle='--') 
        plt.title(f'{i} month SKEW indexes')
        #plt.xlabel('$t$')
        plt.ylabel('Index Level (100-10S)')
        #ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.legend()
        #plt.show()
        plt.savefig(f"outputs/index time series/{i} month Skew Index Time Series.png", bbox_inches='tight')
        plt.close()

        #display(monthly_portfolios[[f'{i} month {m}' for m in skew_methods]].corr().round(2))
        corr_mtx=monthly_portfolios[[f'{i} month {m}' for m in skew_methods]].corr().round(2)
        corr_mtx.to_csv(f'outputs/index time series/{i} month Skew Inex Correlations')


def expectation_hypothesis_regressions(monthly_portfolios,cutoff_year=None):

    regression_stats={}
    variance_measures=['VIX^2','GVIX^2']
    if cutoff_year:
        output_filename = f'outputs/expectation_hpothesis/regression_summaries_post{cutoff_year}.txt'
    else:
        output_filename = 'outputs/expectation_hpothesis/regression_summaries.txt'
    with open(output_filename, 'w') as fh:
            for k in range(1,2+1):
                    #print(f'{k} month')
                    regression_stats[k]={}
                    for v in variance_measures:
                            #print(v)

                            data=monthly_portfolios[['Trade Date',f'1 month {v}',f'2 month {v}',f'3 month {v}',
                                            f'2 month {v} liquidation',f'3 month {v} liquidation',
                                            f'1 month {v} payoff',f'2 month {v} payoff',f'3 month {v} payoff',
                                            f'{3} month number of options',f'{2} month number of options',f'{1} month number of options',
                                            f'{3} month market gross return',f'{2} month market gross return',f'{1} month market gross return',
                                            f'{3} month max moneyness',f'{2} month max moneyness',f'{1} month max moneyness',
                                            f'{3} month min moneyness',f'{2} month min moneyness',f'{1} month min moneyness']]

                            if cutoff_year:
                                data=data[data['Trade Date'].dt.year>=cutoff_year]
                            #august and september
                            #data=data[(data['Trade Date']>=pd.Timestamp('2008-08-01 00:00:00'))]
                            #data=data[data['Trade Date'].dt.year>=2008]
                            #data=data[~data['Trade Date'].dt.year.isin([2008,1998])]

                            data=data.copy()
                            data['VAR(t+k)']=data[f'1 month {v}'].shift(-k).astype(float)
                            data['VAR(t)']=data[f'1 month {v}'].shift(0).astype(float)
                            data['FVAR(t,t+k+1)']=(data[f'{k+1} month {v}']-data[f'{k} month {v}']).astype(float)
                            data['H']=data['FVAR(t,t+k+1)']-data['VAR(t+k)']
                            data['H_discrete']=data[f'{k+1} month {v} liquidation']-data[f'{k} month {v} payoff']-(data[f'{k+1} month {v}']-data[f'{k} month {v}'])
                            data['H_discrete']=data['H_discrete'].astype(float)
                            data['H_discrete']*=-1

                            #data['H']=data['FVAR(t,t+k+1)']-(data['VAR(t+k)']-data['VAR(t)'])
                            data['FVAR(t,t+k+1) - VAR(t+k)']=data['FVAR(t,t+k+1)']-data['VAR(t+k)']
                            #data['-(VAR(t+k)-VAR(t))']=-(data['VAR(t+k)']-data['VAR(t)'])
                            data.dropna(inplace=True)
                            #display(data)

                            ###################################
                            #data=data[~data['Trade Date'].dt.year.isin([2008])]

                            fh.write(f"--- {v} {k+1} Month Regression ---\n")
                            fh.write(f"---- Continuous HPR Version ----\n")
                            fh.write(f"----- Primal Regression -----\n")
                            #Primal: VAR(t+k)-VAR(t) = a + b(FVAR(t,t+k+1)-VAR(t)) + e(t+k)
                            #"""
                            y=data['VAR(t+k)']-data['VAR(t)']
                            x=sm.add_constant(data['FVAR(t,t+k+1)']-data['VAR(t)'])
                            ols_model = sm.OLS(y, x)
                            primal_ols_results= ols_model.fit(cov_type='HAC', cov_kwds={'maxlags':round(3/4*(data.shape[0])**(1/3))})
                            #print('primal')
                            #print(primal_ols_results.params['const'])
                            #print(primal_ols_results.params[0])
                            #print(primal_ols_results.resid.mean())
                            #print( primal_ols_results.summary() )
                            #"""
                            fh.write(primal_ols_results.summary().as_text())
                            fh.write("\n\n")

                            #Dual: 
                            fh.write(f"--- Dual Regression ---\n")
                            #"""
                            y=data['H']
                            x=sm.add_constant(data['FVAR(t,t+k+1)']-data['VAR(t)'])
                            ols_model = sm.OLS(y, x)
                            dual_ols_results = ols_model.fit(cov_type='HAC', cov_kwds={'maxlags':round(3/4*(data.shape[0])**(1/3))})
                            #print('dual')
                            #print(dual_ols_results.params['const'])
                            #print(dual_ols_results.params[0])
                            #print(dual_ols_results.resid.mean())
                            #print( dual_ols_results.summary() )
                            #"""
                            fh.write(dual_ols_results.summary().as_text())
                            fh.write("\n\n")

                            data['alpha_primal']=primal_ols_results.params['const']
                            data['beta_primal']=primal_ols_results.params[0]
                            data['Avg(e)_primal']=primal_ols_results.resid.mean()
                            data['alpha_dual']=dual_ols_results.params['const']
                            data['beta_dual']=dual_ols_results.params[0]
                            data['Avg(e)_dual']=dual_ols_results.resid.mean()
                            data['sum(a)']=primal_ols_results.params['const']+dual_ols_results.params['const']
                            data['sum(B)-1']=primal_ols_results.params[0]+dual_ols_results.params[0]-1
                            #display(data)
                            #print('Summary')
                            #print('sum(a)')
                            #print(primal_ols_results.params['const']+dual_ols_results.params['const'])
                            #print('sum(B)-1')
                            #print(primal_ols_results.params[0]+dual_ols_results.params[0]-1)

                            

                            regression_stats[k][v]={}

                            regression_stats[k][v]['continuous']={'⍺_primal_coef':primal_ols_results.params['const'],
                                            '⍺_primal_Tstat':primal_ols_results.tvalues['const'],
                                            'β_primal_coef':primal_ols_results.params[0],
                                            'β_primal_Tstat':primal_ols_results.tvalues[0],
                                            'R^2_primal':primal_ols_results.rsquared,
                                            '⍺_dual_coef':dual_ols_results.params['const'],
                                            '⍺_dual_Tstat':dual_ols_results.tvalues['const'],
                                            'β_dual_coef':dual_ols_results.params[0],
                                            'β_dual_Tstat':dual_ols_results.tvalues[0],
                                            'R^2_dual':dual_ols_results.rsquared,
                                            'sum(⍺)':primal_ols_results.params['const']+dual_ols_results.params['const'],
                                            'sum(β)-1':primal_ols_results.params[0]+dual_ols_results.params[0]-1,
                                            'sum(β)':primal_ols_results.params[0]+dual_ols_results.params[0],
                                            }
                            #print('-------------------------')

                            #print('Discrete Version')
                            fh.write(f"---- Dicrete HPR Version ----\n")
                            fh.write(f"----- Dual Regression -----\n")

                            y=data['H_discrete']
                            x=sm.add_constant(data['FVAR(t,t+k+1)']-data['VAR(t)'])
                            ols_model = sm.OLS(y, x)
                            dual_ols_results = ols_model.fit(cov_type='HAC', cov_kwds={'maxlags':round(3/4*(data.shape[0])**(1/3))})
                            fh.write(dual_ols_results.summary().as_text())
                            fh.write("\n\n")

                            #print('dual')
                            #print(dual_ols_results.params['const'])
                            #print(dual_ols_results.params[0])
                            #print(dual_ols_results.resid.mean())
                            #print( dual_ols_results.summary() )
                            #"""
                            data['alpha_primal']=primal_ols_results.params['const']
                            data['beta_primal']=primal_ols_results.params[0]
                            data['Avg(e)_primal']=primal_ols_results.resid.mean()
                            data['alpha_dual']=dual_ols_results.params['const']
                            data['beta_dual']=dual_ols_results.params[0]
                            data['Avg(e)_dual']=dual_ols_results.resid.mean()
                            data['sum(a)']=primal_ols_results.params['const']+dual_ols_results.params['const']
                            data['sum(B)-1']=primal_ols_results.params[0]+dual_ols_results.params[0]-1
                            #display(data)
                            #print('Summary')
                            #print('sum(a)')
                            #print(primal_ols_results.params['const']+dual_ols_results.params['const'])
                            #print('sum(B)-1')
                            #print(primal_ols_results.params[0]+dual_ols_results.params[0]-1)


                            regression_stats[k][v]['discrete']={'⍺_primal_coef':primal_ols_results.params['const'],
                                                    '⍺_primal_Tstat':primal_ols_results.tvalues['const'],
                                                    'β_primal_coef':primal_ols_results.params[0],
                                                    'β_primal_Tstat':primal_ols_results.tvalues[0],
                                                    'R^2_primal':primal_ols_results.rsquared,
                                                    '⍺_dual_coef':dual_ols_results.params['const'],
                                                    '⍺_dual_Tstat':dual_ols_results.tvalues['const'],
                                                    'β_dual_coef':dual_ols_results.params[0],
                                                    'β_dual_Tstat':dual_ols_results.tvalues[0],
                                                    'R^2_dual':dual_ols_results.rsquared,
                                                    'sum(⍺)':primal_ols_results.params['const']+dual_ols_results.params['const'],
                                                    'sum(β)-1':primal_ols_results.params[0]+dual_ols_results.params[0]-1,
                                                    'sum(β)':primal_ols_results.params[0]+dual_ols_results.params[0],
                                                    }
                            
                            #separata regression

                            #print('------------------------------------')





    #var_regression_df=pd.DataFrame.from_dict({(i,j): regression_stats[i][j] 
    #                            for i in regression_stats.keys() 
    #                            for j in regression_stats[i].keys()},
    #                            orient='index')
    var_regression_df=pd.DataFrame.from_dict({(i,j,k): regression_stats[i][j][k] 
                                for i in regression_stats.keys() 
                                for j in regression_stats[i].keys()
                                for k in regression_stats[i][j].keys()},
                                orient='index')
    var_regression_df.index.names=['Months From Now','Var. Measure','Version']
    if cutoff_year:
        var_regression_df.to_csv(f'outputs/expectation_hpothesis/variance_expecthypoth_results_post{cutoff_year}.csv')
    else:
        var_regression_df.to_csv('outputs/expectation_hpothesis/variance_expecthypoth_results.csv')

    #var_regression_df#.iloc[:,-3:]


############################################################
"""

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


def find_third_fri_holidays(df):
    df=df.copy()
    holiday_dict={}
    expir_not_third_friday=set(pd.to_datetime(df.exdate.unique()).normalize())-set(third_fridays)
    for expir in expir_not_third_friday:
        #print(expir in third_friday_holidays)
        for holiday in third_friday_holidays:
            if (holiday-expir).days==1:
                #print(holiday,expir)
                holiday_dict[expir]=True

    df['holiday_exp']=df['exdate'].map(holiday_dict)
    #df2['holiday_exp'].fillna(value=False,inplace=True)
    df.fillna({'holiday_exp':False},inplace=True)
    return df

def add_expiration_time(df): 
    #standard SPX 9:30 a.m. ET
    #weekly SPXW 4:15 p.m. ET
    df=df.copy()
    df['datetime_close']=df['date']+pd.Timedelta(hours=16,minutes=15)
    df['ex_time']=df['exdate']+pd.Timedelta(hours=9,minutes=30)
    df['time_to_exp']=df['ex_time']-df['datetime_close']
    return df
"""