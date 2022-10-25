from helper_wacc import CostOfCapital
from helper_utils import RnDConverter, OptionValue
import numpy as np
import pandas as pd

############################################################################
# VALUATION OF A BUSINESS
############################################################################
class Valuation:
    def __init__(self, inputs):
        
        self.inputs = inputs
        self.finanicals = self.inputs['finanicals']
        self.assumptions = self.inputs['valuation']
        
        #---- Valuation components
        self.revenue_growth_rate        = []
        self.revenues                   = []
        self.operating_margin           = []
        self.operating_income           = []
        self.tax_rate                   = []
        self.after_tax_operating_income = []
        self.reinvestment               = []
        self.discount_rate              = []
        self.fcff                       = []
        self.nol                        = []    # net operating loss
        self.cum_discount_factor        = []
        self.pv_fcff                    = []
        self.sales_to_capital_ratio     = []
        self.invested_capital           = []
        self.roic                       = []
        
    #-----------------------------------------------------------------------
    # Base year estimates
    #-----------------------------------------------------------------------
    def base_operating_income_estimator(self):
        
        # reported operating income, EBIT
        ebit = self.finanicals['ebit']
        
        # Adjustment for lease commitments
        ebit += self.coc.op_lease_estimator.adjust_ebit
        
        # Adjustment for R&D expenses
        ebit += self.rnd_convertor.adjustment_to_operating_income
        
        return ebit
        
        
    #-----------------------------------------------------------------------
    # Base year estimates
    #-----------------------------------------------------------------------
    def base_year_estimates(self):
        
        self.n_years = self.assumptions['n_years']
        
        ### COST OF CAPITAL ESTIMATOR
        self.coc = CostOfCapital(self.inputs)
        self.coc.estimate()
        
        ### R&D Convertor
        self.rnd_convertor = RnDConverter(self.inputs)
        self.rnd_convertor.estimate()
        
        #---- BASE ESTIMATES
        base_discount_rate = self.coc.wacc
        base_revenue = self.finanicals['revenues']
        base_operating_income = self.base_operating_income_estimator()
        base_operating_margin = base_operating_income / base_revenue
        base_tax_rate = self.inputs['market_drivers']['marginal_tax_rate']      # TODO: should be effective tax rate
        base_after_tax_operating_income = base_operating_margin * (1 - base_tax_rate)
        base_net_operating_loss = self.finanicals['net_operating_loss']
        base_invested_capital = 21479       #TODO: Create a function to estimate invested captial
        
        self.revenues                   = [base_revenue]                    + [0] * 10
        self.operating_income           = [base_operating_income]           + [0] * 10
        self.after_tax_operating_income = [base_after_tax_operating_income] + [0] * 10
        self.nol                        = [base_net_operating_loss]         + [0] * 10
        self.invested_capital           = [base_invested_capital]           + [0] * 10
        self.roic                       = [base_after_tax_operating_income / base_invested_capital] + [0] * 10
        
        # profiles are filled with assumptions in the downstream process
        self.revenue_growth_rate        = [0]
        self.operating_margin           = [base_operating_margin]
        self.tax_rate                   = [base_tax_rate]
        self.discount_rate              = [base_discount_rate]
        self.sales_to_capital_ratio     = [base_revenue / base_invested_capital]
        
        # fill data with 0
        self.reinvestment               = [0] * 11      # includes base year
        self.fcff                       = [0] * 11  
        self.pv_fcff                    = [0] * 11
    
    #-----------------------------------------------------------------------
    # LOAD ASSUMPTIONS MADE IN VALUATION PROFILE
    #-----------------------------------------------------------------------
    def make_profile(self, start, end, mode, n):
        
        if mode == 'constant':
            return [start] * n
        if mode == 'linear':
            delta = (end - start) / n
            return [start + i * delta for i in range(n+1)]
        if mode == 'exponential':
            delta = (end - start) / n
            return [start * (1 + delta)**n for i in range(n+1)]
        
    def _load_discount_rate_assumptions(self):
        
        rates = self.assumptions['discount_rate']
        
        # check if the `target` rate set as WACC,
        # if so, set the value appropriately
        wacc = self.coc.wacc
        for key in ['growth_phase', 'high_growth_phase']:
            rates[key] = [wacc if i == 'wacc' else i for i in rates[key]]
        return self.load_profile_assumptions(rates)
    
    def _load_tax_rate_assumptions(self):
        rates = self.assumptions['tax_rate']
        effective_tax_rate = 0.25               # TODO: effective tax rate should be implied from finanicals
        marginal_tax_rate = self.inputs['market_drivers']['marginal_tax_rate']
        for key in ['growth_phase', 'high_growth_phase']:
            m = []
            for i in rates[key]:
                if i == 'marginal_tax_rate':
                    m.append(marginal_tax_rate)
                elif i == 'effective_tax_rate':
                    m.append(effective_tax_rate)
                else:
                    m.append(i)
            rates[key] = m
        return self.load_profile_assumptions(rates)
            
    def load_profile_assumptions(self, profile_dict):
        
        # 1. Check if the first year growth rate is manually set
        next_year = profile_dict['next_year'] 
        profile = [next_year] if next_year else []
        
        # 2. Make phase 1 profile : Growth Phase
        growth_phase = profile_dict['growth_phase']
        if len(growth_phase) > 0:
            start_year, end_year, trend, start, end = growth_phase
            n = end_year - start_year + 1
            if trend == 'linear': n = n - 1 # TODO: bug in the code
            profile += self.make_profile(start, end, trend, n)
        
        # 2. Make phase 2 profile : High Growth Phase
        high_growth_phase = profile_dict['high_growth_phase']
        if len(high_growth_phase) > 0:
            start_year, end_year, trend, start, end = high_growth_phase
            start = profile[-1]
            n = end_year - start_year + 1
            if trend == 'constant': n = n + 1 # TODO: bug in the code
            profile = profile[:-1] + self.make_profile(start, end, trend, n)
        
        return profile
        
    def estimate_pv_fcff(self):
        
        # --- Aggregate base estimates
        self.base_year_estimates()
        
        # --- Load valuation profile assumptions
        # Revenue growth rate
        self.revenue_growth_rate += self.load_profile_assumptions(self.assumptions['revenue_growth'])
        # Operating margin
        self.operating_margin +=  self.load_profile_assumptions(self.assumptions['operating_margin'])
        self.operating_margin[2] = 0.0198 # REMOVE
        # Tax rate
        self.tax_rate += self._load_tax_rate_assumptions()
        # Cost of captial profile
        self.discount_rate += self._load_discount_rate_assumptions()
        # Cummulative discount factor
        for i, r in enumerate(self.discount_rate):
            self.cum_discount_factor.append(1 if i == 0 else self.cum_discount_factor[i-1] * (1 / (1 + r)))
        # Sales to capital ratio profile
        self.sales_to_capital_ratio += self.load_profile_assumptions(self.assumptions['sales_to_capital_ratio'])

        #--- Compute valuation constituents
        for y in range(1, self.n_years + 1):
            self.revenues[y] = self.revenues[y-1] * (1 + self.revenue_growth_rate[y])
            self.operating_income[y] = self.revenues[y] * self.operating_margin[y]
            self.after_tax_operating_income[y] = self.operating_income[y] * (1-self.tax_rate[y])
            self.reinvestment[y] = max((self.revenues[y] - self.revenues[y-1])/self.sales_to_capital_ratio[y], 0)
            self.fcff[y] = self.after_tax_operating_income[y] - self.reinvestment[y]
            self.nol[y] = self.nol[y-1] - self.operating_income[y] if (self.operating_income[y] < 0) or (self.nol[y-1] > self.operating_income[y]) else 0
            self.pv_fcff[y] = self.fcff[y] * self.cum_discount_factor[y]
            self.invested_capital[y] = self.invested_capital[y-1] + self.reinvestment[y]
            self.roic[y] = self.after_tax_operating_income[y] / self.invested_capital[y]
            
    def terminal_value(self):
        
        terminal = self.assumptions['terminal_year']
        
        # Growth rate in terminal year
        tgr = terminal['growth_rate']
        roic_in_terminal_year = terminal['roic']
        roic_in_terminal_year = self.discount_rate[-1] if roic_in_terminal_year == 'cost_of_capital' else roic_in_terminal_year
        
        # Append terminal year inputs
        self.revenue_growth_rate.append(tgr)
        self.revenues.append(self.revenues[-1] * (1 + tgr))
        self.operating_margin.append(self.operating_margin[-1])     # operating margin in terminal year is same as year 10 / 5
        self.operating_income.append(self.operating_margin[-1] * self.revenues[-1])
        self.tax_rate.append(self.tax_rate[-1])                     # marginal tax rate in terminal year is same as year 10 / 5
        self.after_tax_operating_income.append(self.operating_income[-1] * (1 - self.tax_rate[-1]))
        self.roic.append(roic_in_terminal_year)
        self.reinvestment.append(self.after_tax_operating_income[-1] * (tgr / roic_in_terminal_year))
        self.fcff.append(self.after_tax_operating_income[-1] - self.reinvestment[-1])
        self.nol.append(self.nol[-1] - self.operating_income[-1] if (self.operating_income[-1] < 0) or (self.nol[-1] > self.operating_income[-1]) else 0)
        self.discount_rate.append(self.discount_rate[-1])
        
        #--- PV of terminal cash flows
        self.terminal_cash_flow = self.fcff[-1]
        self.terminal_cost_of_capital = self.discount_rate[-1]
        self.terminal_value = self.terminal_cash_flow / (self.terminal_cost_of_capital - tgr)
        self.pv_terminal_value = self.terminal_value * self.cum_discount_factor[-1]
        self.pv_growth_phase = np.sum(self.pv_fcff)
        self.pv = self.pv_growth_phase + self.pv_terminal_value
        
        # fill dummy
        self.invested_capital.append('')
        self.cum_discount_factor.append('')
        self.pv_fcff.append('')
        self.sales_to_capital_ratio.append('')
        
    def intergrate_loose_ends(self):
        
        self.prob_of_failure = self.assumptions['prob_of_failure']
        self.proceeds_if_firm_fails = 33316.5 #TODO: integrate in calculations
        self.value_of_operating_assests = self.pv * (1 - self.prob_of_failure) + self.prob_of_failure * self.proceeds_if_firm_fails
        self.debt = self.finanicals['book_value_debt'] + self.coc.op_lease_estimator.adjust_total_debt
        self.minority_interest = 0 #TODO: intergrate in calculations
        self.cash = self.finanicals['cash_market_securities']
        self.non_operating_assets = self.finanicals['non_operating_assets']
        option_value = OptionValue(self.inputs)
        option_value.estimate()
        self.value_of_options = option_value.value_of_all_options_outstanding

    def equaity_value(self):
        self.value_of_equity = self.value_of_operating_assests \
                                - self.debt \
                                - self.minority_interest \
                                + self.cash \
                                + self.non_operating_assets
        self.value_of_equity_in_common_shares = self.value_of_equity - self.value_of_options
        self.num_shares = self.inputs['equity_inputs']['shares_outstanding']
        self.estimated_value_per_share = self.value_of_equity / self.num_shares
        self.price = self.inputs['equity_inputs']['current_stock_price']
        self.price_as_value = self.price / self.estimated_value_per_share
        
    def report(self):
        
        items = [self.revenue_growth_rate, 
                 self.revenues,
                 self.operating_margin,
                 self.operating_income,
                 self.tax_rate,
                 self.after_tax_operating_income,
                 self.reinvestment,
                 self.fcff,
                 self.nol,
                 self.discount_rate,
                 self.cum_discount_factor,
                 self.pv_fcff,
                 self.sales_to_capital_ratio,
                 self.invested_capital,
                 self.roic]

        index = ['revenue growth rate', 'revenues', 'operating margin', 'operating income', 'tax rate', 
                 'after-tax operating income', 'reinvestment', 'fcff', 'nol', 'discount rate', 'cumulative discount factor',
                 'PV (FCFF)', 'sales to capital ratio', 'invested capital', 'roic']
        columns = ['base_year'] + list(range(1, self.n_years + 1)) + ['terminal_year']
        self.df = pd.DataFrame(items, index = index, columns = columns)
        
    def estimate(self):
        self.estimate_pv_fcff()
        self.terminal_value()
        self.intergrate_loose_ends()
        self.equaity_value()
        self.report()
