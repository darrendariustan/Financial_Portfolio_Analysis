import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import math

def drawdown(return_series: pd.Series):
    """
    Takes a time series of asset returns
    Computes and returns a DataFrame that contains:
    the wealth index
    the previous peaks
    the percent drawdown
    """
    wealth_index = 1000*(1+return_series).cumprod() #Assuming we start with 1000 units of currency
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/ previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index, 
        "Peaks": previous_peaks, 
        "Drawdown": drawdowns
        })

def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCap
    """
    me_m = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv",
                       header=0, index_col=0, na_values=-99.99)
    rets = me_m[["Lo 10", "Hi 10"]]
    rets.columns = ["SmallCap", "LargeCap"]
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period('M')
    return rets

def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund Index Returns
    """
    hfi = pd.read_csv("data/edhec-hedgefundindices.csv",
                      header=0, index_col=0, parse_dates=True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi

#Define semideviation with a mask, compute SD:
def semideviation(r):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame
    """
    is_negative = r < 0
    return r[is_negative].std(ddof=0)

#Formula for Skewness
def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    if sigma_r == 0:
        return np.nan #Avoid division by 0
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

#Formula for Kurtosis
def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    if sigma_r == 0:
        return np.nan #Avoid division by 0
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

#Jarque Bera Test
def is_normal(r, level=0.01):
    """
    Applies the Jarque Bera test to determine if a Series is normal
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    r = r.dropna() #Remove all NaN values to avoid errors
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level

def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns fall below that number, and the (100-level) percent of the returns
    are above that number.
    If "level" is equal to 0 or 100, then this function just returns 0 or 1 respectively
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        if r.empty:
            return np.nan #Avoid division by 0
        return -np.percentile(r, level) #change it to positive number as report loss as positive
    else:
        raise TypeError("Expected r to be either a Series or DataFrame")

#For Cornish-Fisher, just need to update Gaussian to adjust z based on skewness and kurtosis    
from scipy.stats import norm
def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gaussian Value at Risk at a specified level
    """
    #compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        #modify the Z score based on skewness and kurtosis. 
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 - 3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))

def cvar_historic(r, level=5):
    """
    Computes the Conditional Value at Risk of Series or dataframe
    """
    if isinstance(r, pd.Series):
        r = r.dropna() #Remove all NaN values before processing
        is_beyond = r <= -var_historic(r, level=level)
        if is_beyond.sum() == 0:
            return np.nan
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a series or Dataframe")

def semideviation3(r):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame, else raises a TypeError
    """
    excess= r-r.mean()                                        # We demean the returns
    excess_negative = excess[excess<0]                        # We take only the returns below the mean
    excess_negative_square = excess_negative**2               # We square the demeaned returns below the mean
    n_negative = (excess<0).sum()                             # number of returns under the mean
    return (excess_negative_square.sum()/n_negative)**0.5     # semideviation



















#From Week 2:
def get_ind_returns():
    """
    Load and format the Ken French 30 Industry Portfolios Average Returns
    """
    ind = pd.read_csv("data/ind30_m_vw_rets.csv", header=0, index_col=0)/100
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_size():
    """
    Load and format the Ken French 30 Industry Portfolios Average Returns
    """
    ind = pd.read_csv("data/ind30_m_size.csv", header=0, index_col=0)/100
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_nfirms():
    """
    Load and format the Ken French 30 Industry Portfolios Average Returns
    """
    ind = pd.read_csv("data/ind30_m_nfirms.csv", header=0, index_col=0)/100
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

#Get total market returns index function: Missing
def get_total_market_index_returns():
    """
    Calculate and return the total market index returns by weighting each industry
    return by its market size.
    """
    # Load industry returns and size data
    ind_returns = get_ind_returns()
    ind_size = get_ind_size()

    # Calculate total market size each period by summing industry sizes
    total_market_size = ind_size.sum(axis=1)

    # Calculate the market-weighted industry returns for each period
    weighted_returns = (ind_size.divide(total_market_size, axis=0)) * ind_returns

    # Sum across industries to get the total market index return
    total_market_index_return = weighted_returns.sum(axis=1)
    
    return total_market_index_return

##Annualized returns function:
def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(1/n_periods) - 1

##Annualized risk function:
def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    return r.std()*(periods_per_year**0.5)

##Function for Sharpe Ratio:
def sharpe(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ext_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ext_ret/ann_vol

#Function equation for portfolio return
def portfolio_return(weights, returns):
    """
    Weights -> Returns
    """
    return weights.T @ returns #direct translation of above equation where @ is matrix multiplication and T is transpose

#Function equation for portfolio risk:
def portfolio_vol(weights, covmat):
    """
    Weights -> Vol
    """
    weights = np.array(weights)
    if np.any(covmat==0) or np.all(weights==0):
        return np.nan #Avoid divide by zero by returning NaN if there are zero values
    return (weights.T @ covmat @ weights)**0.5

#Function for plotting efficient frontier for 2 assets:
def plot_ef2(n_points, er, cov):
    """
    Plots the 2-asset efficient frontier
    """
    if er.shape[0] != 2 or cov.shape[0] != 2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    return ef.plot(x="Volatility", y="Returns", style=".-")

#Function for plotting N-asset efficient frontier:
from scipy.optimize import minimize
def minimize_vol(target_return, er, cov):
    """
    target_ret -> w
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n) #Equally weighted portfolio
    bounds = ((0.0, 1.0),) * n # Creates a tuple weight to be repeated for every asset. Multiplying a tuple or list just makes n copies of it.
    #Putting in all the constraints:
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
    }
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) -1
    }
    results = minimize(portfolio_vol, init_guess, args=(cov,), 
                       method='SLSQP', options={'disp': False},
                       constraints=(return_is_target, weights_sum_to_1),
                       bounds=bounds
                       )
    
    return results.x

#Plotting the multi-asset efficient frontier:

def optimal_weights(n_points, er, cov):
    """
    Generates a list of weights which represent a 'n_points' number of equally
    spaced weights in a frontier of returns, to run the optimizer on to minimize the volatility
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights

#Function for Maximum Sharpe Ratio Portfolio
def msr(riskfree_rate, er, cov):
    """
    RiskFree rate + ER + COV -> W
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n) #Equally weighted portfolio
    bounds = ((0.0, 1.0),) * n # Creates a tuple weight to be repeated for every asset. Multiplying a tuple or list just makes n copies of it.
    #Constraint 1:
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) -1
    }
    #Constraint 2:
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio, given weights
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    result = minimize(neg_sharpe_ratio, init_guess, 
                      args=(riskfree_rate, er, cov), method='SLSQP', 
                      options={'disp': False},
                      constraints=(weights_sum_to_1),
                      bounds=bounds
                      )
    
    return result.x

# Defining the Global Minimum Variance Portfolio covariance function:
def gmv(cov):
    """
    Returns the weights of the global minimum variance portfolio
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1,n), cov)


#Plotting the N-asset efficient frontier
def plot_ef(n_points, er, cov, show_cml = False, style = ".-", riskfree_rate = 0, show_ew = False, show_gmv=False):
    """
    Plots the N-asset efficient frontier
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    ax = ef.plot.line(x="Volatility", y="Returns", style=style)

    
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, er)
        #display EW
        ax.plot([vol_ew], [r_ew], color='goldenrod', marker = "o", markersize = 10)

    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        #display EW
        ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker = "o", markersize = 10)

    if show_cml:
        ax.set_xlim(left = 0)
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        #Add CML
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color="green", marker="o", linestyle="dashed", linewidth=2, markersize=12)

    return ax








#From Week 3:
def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03, drawdown=None):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History
    """
    # set up the CPPI parameters:
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start*floor #Initialize the floor value
    peak = start
    
    #Initialize as dataframe if input is series:
    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns=["R"])

    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/12 # fast way to set all values to a scalar
    
    #set up some DataFrames for saving intermediate values:
    account_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        #scenario to allow for drawdown to be dynamic changing according to peaks, where floor value is not static but relative:
        if drawdown is not None:
            peak = np.maximum(peak,account_value) #higher of previous peak and account value
            floor_value = peak * (1-drawdown) #if drawdown is 20%, floor is 80% of peak
        cushion = (account_value - floor_value)/account_value
        risky_w = m*cushion
        risky_w = np.minimum(risky_w, 1) # limit to max of 100%
        risky_w = np.maximum(risky_w, 0) #to prevent allocation to be less than 0%
        safe_w = 1 - risky_w
        risky_alloc = account_value*risky_w #Absolute dollars in risky asset
        safe_alloc = account_value*safe_w #Absolute dollars in safe asset

        #recompute the new account value at the end of this step:
        account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])
        
        #save the histories for plotting
        account_history.iloc[step] = account_value
        risky_w_history.iloc[step] = risky_w
        cushion_history.iloc[step] = account_value - floor_value

    risky_wealth = start*(1+risky_r).cumprod()

    #Include backtesting results:
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth,
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "floor": floor, 
        "risky_r": risky_r,
        "safe_r": safe_r
    }
    return backtest_result

#Include summary stats defining function:
def summary_stats(r, riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    #If r is empty:
    if r.empty:
        raise ValueError("The input data for summary statistics is empty")
    
    #Drop any NaN values in the return series before calculating statistics:
    r = r.dropna()

    #Calculate individual statistics:
    ann_r = r.aggregate(annualize_rets, periods_per_year = 12)
    ann_vol = r.aggregate(annualize_vol, periods_per_year = 12)
    ann_sr = r.aggregate(sharpe, riskfree_rate = riskfree_rate, periods_per_year = 12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified = True)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fischer VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })

def gbm(n_years = 10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0, prices=True):
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    """
    # Derive per-step Model Parameters from User Specifications
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    # the standard way ...
    # rets_plus_1 = np.random.normal(loc=mu*dt+1, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    # without discretization error ...
    rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    ret_val = s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
    return ret_val


def show_gbm(n_scenarios, mu, sigma):
    """
    Draw the results of a stock price evolution under a Geometric Brownian Motion Model
    """
    s_0=100
    prices = gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, s_0=s_0)
    ax = prices.plot(legend=False, color="indianred", alpha=0.5, linewidth=2, figsize=(14,7))
    ax.axhline(y=s_0, ls=":", color="black")
    ax.set_ylim(top=450)
    #Draw a dot at the origin
    ax.plot(0,s_0, marker='o', color='darkred', alpha=0.2)

def show_cppi(n_scenarios=50, mu=0.07, sigma=0.15, m=3, floor=0., riskfree_rate=0.03, y_max=100, steps_per_year=12):
    """
    Plot the results of a Monte Carlo Simulation of CPPI
    """
    start = 100
    sim_rets = gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, prices=False, steps_per_year=steps_per_year)
    risky_r = pd.DataFrame(sim_rets)
    #run the back test
    btr = run_cppi(risky_r=pd.DataFrame(risky_r), riskfree_rate=riskfree_rate, m=m, start=start, floor=floor)
    wealth = btr["Wealth"]

    #Calculate terminal wealth stats
    y_max = wealth.values.max()*y_max/100
    terminal_wealth = wealth.iloc[-1]

    tw_mean = terminal_wealth.mean()
    tw_median = terminal_wealth.median()
    failure_mask = np.less(terminal_wealth, start*floor)
    n_failures = failure_mask.sum()
    p_fail = n_failures/n_scenarios

    e_shortfall = np.dot(terminal_wealth-start*floor, failure_mask)/n_failures if n_failures > 0 else 0.0

    #Plot!
    fig, (wealth_ax, hist_ax) = plt.subplots(nrows=1, ncols=2, sharey=True, gridspec_kw={'width_ratios':[3, 2]}, figsize=(24, 9))
    plt.subplots_adjust(wspace=0.0)

    wealth.plot(ax=wealth_ax, legend=False, color="indianred", alpha=0.3, figsize=(12, 6))
    wealth_ax.axhline(y = start, ls = ":", color = "black")
    wealth_ax.axhline(y = start*floor, ls = "--", color = "red")
    wealth_ax.set_ylim(top = y_max)

    terminal_wealth.plot.hist(ax = hist_ax, bins=50, ec='w', fc='indianred', orientation = "horizontal")
    hist_ax.axhline(y = y_max/10, ls = ":", color = "black")
    hist_ax.axhline(y = tw_mean, ls = ":", color = "blue")
    hist_ax.axhline(y = tw_median, ls = ":", color = "purple")
    hist_ax.annotate(f"Mean: ${int(tw_mean)}", xy=(0.7, 0.9), xycoords="axes fraction", fontsize = 24)
    hist_ax.annotate(f"Median: ${int(tw_median)}", xy=(0.7, 0.8), xycoords="axes fraction", fontsize = 24)
    if (floor > 0.01):
        hist_ax.axhline(y = start*floor, ls = "--", color = "red", linewidth=3)
        hist_ax.annotate(f"Violations: {n_failures} ({p_fail*100:2.2f}%)\nE(shortfall)=${e_shortfall:2.2f}", xy=(0.7, 0.7), xycoords="axes fraction", fontsize = 24)
    







## From Week 4:

def inst_to_ann(r):
    """
    Converts short rate to an annualized rate
    """
    return np.expm1(r)

def ann_to_inst(r):
    """
    Converts an annualized rate to a short rate
    """
    return np.log1p(r)

def discount(t, r):
    """
    Compute the price of a pure discount bond that pays a dollar at time period t,
    and r is the per period interest rate
    returns a |t| x |r| Series or Dataframe
    r can be a float, or a Series/Dataframe
    returns a DataFrame indexed by t
    """
    if isinstance(r, (float, int)):  # Single rate case
        discounts = pd.Series((1 + r) ** -t, index=t)  # Return a Series
    else:  # Multiple rates case (r is Series/DataFrame)
        discounts = pd.DataFrame([(1 + r)**-i for i in t], index=t)
    return discounts

def pv(flows, r):
    """
    Compute the present value of a sequence of cash flows given by the time (as an index) and amounts
    r can be a scalar, or a Series/Dataframe with the number of rows matching the num of rows in flows
    """
    dates = flows.index
    discounts = discount(dates, r)
    return discounts.multiply(flows, axis='rows').sum()

def funding_ratio(assets, liabilities, r):
    """"
    Computes the funding ratio of some assets given liabilities and interest rate
    """
    return pv(assets, r)/pv(liabilities, r)

def cir(n_years=10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None):
    """
    Generate random interest rates evolution over time using the CIR model
    b and r_0 are assumed to be the annualized rate, not the short rate
    and the returned values are the annualized rates as well
    """
    if r_0 is None: r_0 = b
    r_0 = ann_to_inst(r_0)
    dt = 1/steps_per_year
    num_steps = int(n_years*steps_per_year)+1 #because n_years might be a float

    shock = np.random.normal(0, scale = np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0

    ## For Price Generation
    h = math.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)

    def price(ttm, r):
        _A = ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(math.exp(h*ttm)-1)) / (2*h + (h+a)*(math.exp(h*ttm)-1))
        _P = _A *np.exp(-_B*r)
        return _P
    prices[0] = price(n_years, r_0)

    for step in range(1, num_steps):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + d_r_t)
        #generate prices at time t as well...
        prices[step] = price(n_years-step*dt, rates[step])

    rates = pd.DataFrame(data = inst_to_ann(rates), index=range(num_steps))
    prices = pd.DataFrame(data=prices, index=range(num_steps))
    return rates, prices


def bond_cash_flows(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12):
    """
    Returns the series of cash flows generated by a bond,
    indexed by coupon number
    """
    n_coupons = round(maturity*coupons_per_year)
    coupon_amt = principal*coupon_rate/coupons_per_year
    coupon_times = np.arange(1, n_coupons+1)
    cash_flows = pd.Series(data=coupon_amt, index=coupon_times)
    cash_flows.iloc[-1] += principal
    return cash_flows

def bond_price(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12, discount_rate=0.03):
    """
    Computes the price of the bond that pays regular coupons until maturity
    at which time the principal and the final coupon is returned
    This is not designed to be efficient, rather
    it is to illustrate the underlying principle behind bond pricing!
    If discount_rate is a Dataframe, then this is assumed to be the rate on each coupon date
    and the bond value is computed over time.
    i.e. The index of the discount_rate DataFrame is assumed to be the coupon number
    """
    if isinstance(discount_rate, pd.DataFrame):
        pricing_dates = discount_rate.index
        prices = pd.DataFrame(index=pricing_dates, columns = discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t] = bond_price(maturity-t/coupons_per_year, principal, coupon_rate, coupons_per_year, discount_rate.loc[t])
        return prices
    
    else: #base case ... single time period
        if maturity <= 0: return principal+principal*coupon_rate/coupons_per_year
        cash_flows = bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
        return pv(cash_flows, discount_rate/coupons_per_year)


def macaulay_duration(flows, discount_rate):
    """
    Computes the Macaulay Duration of a sequence of cash flows
    """
    discounted_flows = discount(flows.index, discount_rate)*flows
    weights = discounted_flows/discounted_flows.sum()
    return np.average(flows.index, weights=weights)

def match_durations(cf_t, cf_s, cf_l, discount_rate):
    """
    Computes the weight W in cf_s that, along with (1-W) in cf_l, 
    will have an effective duration that matches cf_t
    """
    d_t = macaulay_duration(cf_t, discount_rate)
    d_l = macaulay_duration(cf_l, discount_rate)
    d_s = macaulay_duration(cf_s, discount_rate)
    return (d_l-d_t)/(d_l-d_s)

def bond_total_return(monthly_prices, principal, coupon_rate, coupons_per_year):
    """
    Computes the total return on a bond based on monthly bond prices and coupon payments
    Assumes that dividends (coupons) are paid out at the end of the period (e.g. end of 3 months for quarterly div)
    and that dividends are reinvested in the bond
    """
    coupons = pd.DataFrame(data = 0.0, index = monthly_prices.index, columns= monthly_prices.columns)
    t_max = monthly_prices.index.max()
    pay_date = np.linspace(12/coupons_per_year, t_max, int(coupons_per_year*t_max/12), dtype =int)

    coupon_payment = float(principal*coupon_rate/coupons_per_year)
    coupons.iloc[pay_date] = coupon_payment # Explicitly cast to float

    total_returns = (monthly_prices + coupons)/monthly_prices.shift() - 1
    return total_returns[1:].dropna()


def bt_mix(r1, r2, allocator, **kwargs):
    """"
    Runs a back test (simulation) of allocating between a two sets of returns
    r1 and r2 are T x N DataFrames or returns where T is the time step index and N is the number of scenarios.
    allocator is a function that takes two sets of returns and allocator specific parameters, and produces
    an allocation to the first portfolio (the rest of the money is invested in a GHP) as a T x 1 dataframe
    Returns a T x N DataFrame of the resulting N portfolio scenarios
    """
    if not r1.shape == r2.shape:
        raise ValueError("r1 and r2 must be the same shape")
    weights = allocator(r1, r2, **kwargs)
    if not weights.shape == r1.shape:
        raise ValueError("Allocator returned weights that don't match r1")
    r_mix = weights*r1 + (1-weights)*r2
    return r_mix

def fixedmix_allocator(r1, r2, w1, **kwargs):
    """
    Produces a time series over T steps of allocations between the PSP and GHP across N scenarios
    PSP and GHP are T x N DataFrames that represent the returns of the PSP and GHP such that:
     each column is a scenario
     each row is the price for a time step
    Returns a T x N DataFrame of PSP Weights
    """
    return pd.DataFrame(data=w1, index=r1.index, columns=r1.columns)

def terminal_values(rets):
    """
    Returns the final values of a dollar at the end of the return period for each scenario
    """
    return (rets+1).prod()

def terminal_stats(rets, floor=0.8, cap=np.inf, name="Stats"):
    """
    Produces Summary Statistics on the terminal values per invested dollar
    across a range of N scenarios
    rets is a T x N DataFrame of returns, where T is the time-step (we assume rets is sorted by time)
    Returns a 1 column DataFrame of the summary Stats indexed by the stat name
    """
    terminal_wealth = (rets+1).prod(1)

    breach = terminal_wealth < floor
    reach = terminal_wealth >= cap
    p_breach = breach.mean() if breach.sum() > 0 else np.nan
    p_reach = reach.mean() if reach.sum() > 0 else np.nan

    e_short = (floor-terminal_wealth[breach]).mean() if breach.sum() > 0 else np.nan
    e_surplus = (cap-terminal_wealth[reach]).mean() if reach.sum() > 0 else np.nan
    sum_stats = pd.DataFrame.from_dict({
        "mean" : terminal_wealth.mean(),
        "std" : terminal_wealth.std(),
        "p_breach" : p_breach,
        "p_reach" : p_reach,
        "e_short" : e_short,
        "e_surplus" : e_surplus
    }, orient="index", columns=[name])
    return sum_stats

def glidepath_allocator(r1, r2, start_glide=1, end_glide=0):
    """
    Simulates a Target-Date-Fund style gradual move from r1 to r2
    """
    n_points = r1.shape[0]
    n_col = r1.shape[1]
    path = pd.Series(data=np.linspace(start_glide, end_glide, num = n_points))
    paths = pd.concat([path]*n_col, axis=1)
    paths.index = r1.index
    paths.columns = r1.columns
    return paths

def floor_allocator(psp_r, ghp_r, zc_prices, floor=0.8, m=3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside of the PSP without going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple of the cushion in the PSP.
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
    """
    if zc_prices.shape != psp_r.shape:
        raise ValueError("PSP and ZC prices must be the same shape")
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    m = int(m)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = floor*zc_prices.iloc[step] ## PV of floor assuming todays rates and flat YC
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0,1) #same as applying min and max
        ghp_w = 1 - psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the new account value at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        w_history.iloc[step] = psp_w
    return w_history

def drawdown_allocator(psp_r, ghp_r, maxdd, m=3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside of the PSP without going violating the floor
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple of the cushion in the PSP
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
    """
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    peak_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = (1-maxdd)*peak_value
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0,1) #same as applying min and max
        ghp_w = 1 - psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the new account value and prev peak at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        peak_value = np.maximum(peak_value, account_value)
        w_history.iloc[step] = psp_w
    return w_history
