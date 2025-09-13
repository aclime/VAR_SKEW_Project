import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from io import StringIO
from datetime import timedelta
import pytz
import datetime

import pandas_market_calendars as mcal
import matplotlib.dates as mdates


if __name__ == "__main__":
    from helpers import (compute_option_portfolios,
                        compute_correlation_matrices,
                        skew_payoff_graphs,
                        var_payoff_graphs,
                        plot_bid_ask_spread,
                        index_time_series,
                        expectation_hypothesis_regressions)

    #portfolio_dict, option_portfolios=compute_option_portfolios(num_periods_wanted=3,lb_year=1998,ub_year=2023,ticker='SPX')
    print('Computing Monthly Portfolios')
    lb_year,ub_year=1998,2023
    monthly_portfolios, option_portfolios_df=compute_option_portfolios(num_periods_wanted=3,lb_year=lb_year,ub_year=ub_year,ticker='SPX')
    print('Done')

    #Correlations
    compute_correlation_matrices(monthly_portfolios)
    #corr_dfs=compute_correlation_matrices(monthly_portfolios)
    #for i,df in zip([1,2,3],corr_dfs):
    #    df.to_csv(f'outputs/returns_correlations/{i} month returns correlation')

    #Payoff Graphs
    maturity_months=1
    skew_payoff_graphs(option_portfolios_df,maturity_months)
    var_payoff_graphs(option_portfolios_df,maturity_months)

    #Bid-Ask Spreads
    plot_bid_ask_spread(monthly_portfolios)

    #Index Level Time Series
    index_time_series(monthly_portfolios)

    #Expectations Hypothessis
    expectation_hypothesis_regressions(monthly_portfolios)
    expectation_hypothesis_regressions(monthly_portfolios,2009)
    #expectation_hypothesis_regressions(monthly_portfolios,2020)


