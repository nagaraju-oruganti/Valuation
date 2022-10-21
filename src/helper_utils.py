import os
import pandas as pd
import numpy as np
import math
from scipy.stats import norm


###############################################################################
# LOGGER
###############################################################################
class Logger:
    def __init__(self, repo):
        ''' Logger'''
        self.filepath = f'{repo}/log.txt'
        
    def log(self, comment):
        if isinstance(comment, list):
            for c in comment:
                self._log(c)
            self._log('\n')
        else:
            self._log(comment)
            
    def _log(self, comment):
        print(comment)
        try:
            with open(self.filepath, 'a') as f:
                f.write(str(comment) + '\n')
        except:
            pass

    def cleanup(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)

#------------------------------------------------------------------------------
# BOTTOM-UP EQUITY RISK PREMIUM ESTIMATOR
#------------------------------------------------------------------------------
class EquityRiskPremiumEstimator:
    def __init__(self, inputs):
        self.inputs = inputs['erp']
        
    def bottom_up_erp(self):
        source = 'revenues_by_country' if len(self.inputs['revenues_by_country']) != 0 else 'revenues_by_region'
        
        columns = ['source', 'revenue', 'erp']
        self.df = pd.DataFrame(self.inputs[source]).T
        self.df.reset_index(drop = False, inplace = True)
        self.df.columns = columns
        self.df['erp'] = self.df['erp'] / 100             # convert erp into percentages
        self.df['weight'] = self.df['revenue'] / np.sum(self.df['revenue'])
        self.df['weighted_erp'] = self.df['weight'] * self.df['erp']
        
        erp = np.sum(self.df['weighted_erp'])
        return erp
        
    def estimate(self):
        if self.inputs['input_value'] != -1:
            # manual entry of equity risk premium
            self.erp =  self.inputs['erp']['input_value']
        else:
            self.erp =  self.bottom_up_erp()
        
#------------------------------------------------------------------------------
# Option Value
#------------------------------------------------------------------------------
class UnleveredBetaEstimator:
    def __init__(self, inputs):
        self.beta_inputs = inputs['unlevered_beta']
        
    def bottom_up_beta(self):
        self.df = pd.DataFrame(self.beta_inputs['businesses']).T
        self.df.reset_index(inplace = True, drop = False)
        self.df.columns = ['business', 'revenue', 'ev_to_sales', 'unlevered_beta']
        self.df['est_enterprise_value'] = self.df['ev_to_sales'] * self.df['revenue']
        total_est_ev = np.sum(self.df['est_enterprise_value'])
        self.df['weighted_unlevered_beta'] = self.df.apply(
            lambda r: r['unlevered_beta'] * r['est_enterprise_value'] / total_est_ev,
            axis = 1)
        
        return np.sum(self.df['weighted_unlevered_beta'])

    def estimate(self):
        if self.beta_inputs['input_value'] != -1:
            self.unlevered_beta = self.beta_inputs['input_value']
        else:
            self.unlevered_beta = self.bottom_up_beta()

#------------------------------------------------------------------------------
# OPERTATING LEASE CONVERTOR
#------------------------------------------------------------------------------
class OperatingLeaseConverter:
    
    '''
        Converts operating leases to debt
        inputs: pre-tax cost of debt and inputs (from yaml)
        
    '''
    
    def __init__(self, pre_tax_cost_of_debt, inputs):
        self.pre_tax_cost_of_debt = pre_tax_cost_of_debt
        self.inputs = inputs['operating_leases']['commitments']
        self.current_year_lease_expenses = self.inputs[0]
        self.beyond_n_years = self.inputs[-1]
        self.commitments = self.inputs[1:-1]
        self.n_years = len(inputs) - 2 # current_year & beyond_n_years
        
    def estimate(self):
        # Years embedded in lease commitments beyond x years
        avg_lease_debt = np.mean(self.commitments)
        self.embed_years = int(round(self.beyond_n_years / avg_lease_debt, 0))
        self.commitments += [self.beyond_n_years/self.embed_years] * self.embed_years
        
        # present value of lease commitments
        self.pv = [(c/(1+self.pre_tax_cost_of_debt)**i) for i, c in enumerate(self.commitments, start = 1)]
        
        # reproduce into table
        self.df = pd.DataFrame(np.transpose([self.commitments, self.pv]), 
                               index = np.arange(1, len(self.commitments)+1))
        self.df.reset_index(inplace = True, drop = False)
        self.df.columns = ['year', 'commitment', 'present_value']
        
        # debt value of leases
        self.debt_value_of_leases = sum(self.pv)
        
        # depreciation of lease assets
        self.depr_lease_assets = self.debt_value_of_leases / len(self.commitments)
        
        # adjustment to operating income
        self.adjust_opr_income = self.current_year_lease_expenses - self.depr_lease_assets
        
        # adjustment to total debt outstanding
        self.adjust_total_debt = self.debt_value_of_leases
        
        # adjustment to depreciation
        self.adjust_depr = self.depr_lease_assets
        
        # adjustment to operating earnings
        self.adjust_ebit = self.depr_lease_assets - self.current_year_lease_expenses

#------------------------------------------------------------------------------
# R&D CONVERTOR
#------------------------------------------------------------------------------
class RnDConverter:
    def __init__(self, inputs):
        self.marginal_tax_rate = inputs['market_drivers']['marginal_tax_rate']
        self.expenses = inputs['capitalize_rnd']['expenses']
        self.amortization_years = len(self.expenses) - 1
        multipler = 1/self.amortization_years
        self.unamortized_portions = [1 - multipler*idx for idx in range(len(self.expenses))]
        
    def estimate(self):
        self.df = pd.DataFrame(np.transpose([self.expenses, self.unamortized_portions]),
                                            index = np.arange(0, len(self.expenses)))
        self.df.reset_index(inplace = True, drop = False)
        self.df.columns = ['year', 'rnd_expense', 'unamortized_portion']
        self.df['unamortized_expense'] = self.df['rnd_expense'] * self.df['unamortized_portion']
        self.df['amortization_this_year'] = self.df.apply(
            lambda r: 0 if r['year'] == 0 else r['rnd_expense']/self.amortization_years, 
            axis = 1)
        
        self.value_of_research_asset = np.sum(self.df['unamortized_expense'])
        self.value_of_amorization_this_year = np.sum(self.df['amortization_this_year'])
        
        self.adjustment_to_operating_income = self.expenses[0] - self.value_of_amorization_this_year
        self.tax_effect_of_rnd = self.adjustment_to_operating_income * self.marginal_tax_rate

#------------------------------------------------------------------------------
# Option Value
#------------------------------------------------------------------------------
class OptionValue:
    def __init__(self, inputs):
        self.inputs = inputs
        self.current_stock_price = self.inputs['equity_inputs']['current_stock_price']
        self.options = self.inputs['options']
        self.strike_price = self.options['strike_price']
        self.expiration = self.options['expiration']
        self.std_dev_stock_price = self.options['std_dev_stock_price']
        self.annual_dividend_yield = self.inputs['equity_inputs']['annual_dividend_yield']
        self.treasury_bond_rate = self.inputs['market_drivers']['treasury_bond_rate']
        self.options_outstanding = self.options['options_outstanding']
        self.shares_outstanding = self.inputs['equity_inputs']['shares_outstanding']
        
        self.S = self.current_stock_price
        self.K = self.strike_price
        self.div_adj_interest_rate = self.treasury_bond_rate - self.annual_dividend_yield
        self.variance = self.std_dev_stock_price ** 2
        
        # SEED
        self.value1 = 0
        self.value2 = 1000
        
    def black_sholes_model(self):
        
        self.d1 = np.log(self.adj_S / self.adj_K) + (self.div_adj_interest_rate * self.variance / 2) * self.expiration
        self.d1 = self.d1 / (self.std_dev_stock_price * np.sqrt(self.expiration))
        
        self.d2 = (self.d1) - (self.std_dev_stock_price * np.sqrt(self.expiration))
        
        self.norm_d1 = norm.cdf(self.d1)
        self.norm_d2 = norm.cdf(self.d2)
        
    def estimate(self):
        delta = abs(self.value1 - self.value2)
        while (delta > 0.001):
            self.adj_S = (self.S * self.shares_outstanding + self.K * self.options_outstanding)
            self.adj_S = self.adj_S / (self.shares_outstanding + self.options_outstanding)
            self.adj_K = self.K
            self.black_sholes_model()
            
            # value per option
            part1 = self.adj_S * np.exp(-self.annual_dividend_yield * self.expiration) * self.norm_d1
            part2 = self.adj_K * np.exp(-self.treasury_bond_rate * self.expiration) * self.norm_d2
            self.value2 = part1 - part2
            delta = abs(self.value1 - self.value2)
            self.value1 = self.value2
        
        # outputs
        self.value_per_option = self.value2
        self.value_of_all_options_outstanding = self.value_per_option * self.options_outstanding
        
#------------------------------------------------------------------------------
# PRE-TAX COST OF DEBT
#------------------------------------------------------------------------------
class PreTaxCostOfDebtEstimator:
    def __init__(self, inputs):
        
        '''
            We need pre-tax cost of debt to estimate the effect of lease comments on
            operating income and debt. The `interest expense` from lease debt is one of the
            input to estimate pre-tax cost of debt. There exists a circular dependency and we
            will use iterative method to estimate the cost of debt.
        '''
        self.inputs = inputs
        
        self.equity_inputs = self.inputs['equity_inputs']
        self.finanicals = self.inputs['finanicals']
        
        self.market_cap = self.equity_inputs['current_stock_price'] * self.equity_inputs['shares_outstanding']
        self.ebit = self.finanicals['ebit']
        self.interest_expense = self.finanicals['interest_expense']
        self.long_term_bond_rate = self.inputs['market_drivers']['long_term_bond_rate']
        
        # dummy values to estiamte convergence
        self.value1 = 0.001
        self.value2 = 1000
        
         # load ratings lookup table
        self.master_lookup = pd.read_csv(self.inputs['external_data']['synthetic_rating'])
        
    def synthetic_rating_lookup(self, rating = ''):
        
        self.lookup = self.master_lookup.copy(deep = True)
        
        # Screen the ratings table on sector and market cap
        is_finance_services = self.inputs['company_info']['sector'].lower() == 'finance'
        is_large = False
        if not is_finance_services:
            is_large = self.market_cap >= self.inputs['market_drivers']['large_cap_threshold']
        
        if is_finance_services:
             self.lookup =  self.lookup[ self.lookup['firm_type'] == 'finanical service']
        else:
             self.lookup =  self.lookup[ self.lookup['firm_type'] != 'finanical service']
            
        if is_large:
             self.lookup =  self.lookup[ self.lookup['firm_type'] == 'large']
             
        if rating == '':
            # Screen the ratings on interest converage_ratio
            cond1 =  self.lookup['low'] <= self.interest_coverage_ratio
            cond2 =  self.lookup['high'] > self.interest_coverage_ratio
            self.lookup =  self.lookup[cond1 & cond2]
            
            self.rating =  self.lookup['rating'].values[0]
            self.default_spread =  self.lookup['spread'].values[0]
            # convert default spread form string % to decimal
            self.default_spread = float(self.default_spread.replace('%', '')) / 100.
        else:
            self.rating = rating
            self.lookup = self.lookup[self.looking['rating'] == self.rating]
            self.default_spread =  self.lookup['spread'].values[0]
            # convert default spread form string % to decimal
            self.default_spread = float(self.default_spread.replace('%', '')) / 100.
        
    def run_iteration(self):
        op_lease = OperatingLeaseConverter(self.value1, self.inputs)
        op_lease.estimate()
        self.lease_adjust_opr_income = op_lease.adjust_opr_income
        self.lease_adjust_debt = op_lease.adjust_total_debt

    def estimate(self):
        
        if self.inputs['debt']['rate'] != -1:
            self.pre_tax_cost_of_debt = self.inputs['debt']['rate']
            return
        
        if self.inputs['debt']['rating'] != '':
            self.synthetic_rating_lookup(self.inputs['debt']['rating'])
            self.pre_tax_cost_of_debt = self.default_spread
            return
        
        self.num_iter = 0
        delta = abs(self.value1 - self.value2)
        while delta >= 1e-4:
            
            self.run_iteration()
            
            self.adjust_interest_expense = self.interest_expense + self.lease_adjust_debt * self.value1
            self.adjust_ebit = self.ebit + self.lease_adjust_opr_income
        
            # interest coverage ratio
            self.interest_coverage_ratio = self.adjust_ebit / self.adjust_interest_expense
            self.synthetic_rating_lookup()
            
            # pre-tax cost of debt
            self.value2 = self.default_spread + self.long_term_bond_rate
            
            self.num_iter += 1
            delta = abs(self.value1 - self.value2)
            self.value1 = self.value2
    
        # OUTPUT
        self.pre_tax_cost_of_debt = self.value2
        
        
#------------------------------------------------------------------------------
# DISTRIBUTIONS
#------------------------------------------------------------------------------
from scipy.stats import (uniform, norm, triang, skewnorm)

class RandomDistributions:
    def __init__(self):
        self.random_state = 42
        
    def construct_normal_samples(self, mean, std_dev, sample_size):
        return norm.rvs(loc = mean, 
                        scale = std_dev, 
                        size = sample_size, 
                        random_state = self.random_state)
        
    def construct_uniform_samples(self, low, high, sample_size):
        return uniform.rvs(loc = low, 
                           scale = high - low, 
                           size = sample_size,
                           random_state = self.random_state)
        
    def construct_triangular_samples(self, low, high, peak, sample_size):
        
        scale = high - low
        c = (peak - low) / scale
        
        return triang.rvs(c = c, 
                          loc = low, 
                          scale = scale, 
                          size = sample_size,
                          random_state = self.random_state)
        
    def construct_skewnorm_samples(self, a, mean, std_dev, sample_size):
        # skewness parameter (a)
        return skewnorm.rvs(a, 
                            loc = mean,
                            scale = std_dev,
                            size = sample_size,
                            random_state = self.random_state)

################################################################################################
# DCF MODEL
################################################################################################
from enum import Enum
class DCFModelType(Enum):
    dividend_discount_model = 0
    high_growth_model = 1
    
class DecayType(Enum):
    linear = 0
    exponential = 1
    constant = 2