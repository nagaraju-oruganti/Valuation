############################################################################
# VALUATION OF A BUSINESS IN FINANCIAL SERVICE INDUSTRY
'''
    Business model of a company in financial service industry is different from 
    the rest. Below are the three signficant differences:
    1. Financial service company is highly regulated by local government and expected to 
       maintain cash against the customer deposits.
    2. Raw material and loans are same. Customer and cooperates deposits money with the bank
       in excahnge of interest, and the bank lends the same money to its customers and earn
       interest on the loans. In general, difference in the interest rate it earn on advances 
       and the interest rate it pay on the customer deposits is its earnings. Banks do borrow
       from other banks and institutional investors at some interest rates, and largly they are
       significantly low in comparision to deposits.
    3. Blur in reinvestment measurement. Financial service company reinvests on maintaining
       capital adequecy ratio (apart from many other operational and strategic). The investments 
       they make on intangible such as brand name and human capital and normally expensed as operating.
       Estimating working capital is not possible because current assets and liabilites are hard to 
       quantify.
       
    So, we preper to value equity of financial serivce company.
'''
############################################################################


from helper_utils import MakeProfile
import numpy as np
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.5f' % x)

############################################################################
# COST OF EQUITY
# cost of equity = risk free rate + beta x equity risk premium
# risk free rate = 10-yr bond yield - country risk premium
# equity risk premium = mature market risk premium + country risk premium
############################################################################
class Valuation:

    def __init__(self, inputs):
        self.inputs = inputs
        self.mkt_info = self.inputs['market_info']
        self.company_info = self.inputs['company_info']
        self.valuation = self.inputs['valuation']
        self.n_years = self.valuation['n_years']
        
    def base_estimates(self):
        # note that bond yield, country risk premium, and mature market risk premium
        # as inputed in percentage values, now we convert then to decimals
        self.base_bond_yield = self.mkt_info['bond_yield']  / 100
        self.base_country_risk_premium = self.mkt_info['country_risk_premium']  / 100
        self.base_mature_mkt_risk_premium = self.mkt_info['mature_mkt_risk_premium'] / 100
        self.base_risk_free_rate = self.base_bond_yield - self.base_country_risk_premium
        self.base_equity_risk_premium = self.base_mature_mkt_risk_premium + self.base_country_risk_premium
        self.base_beta = self.mkt_info['average_beta']
        
        self.base_cost_of_equity = self.base_risk_free_rate + self.base_beta * self.base_equity_risk_premium
        
        # financials
        self.base_assets = self.inputs['finanicals']['total_assets']
        self.base_bv_of_equity = self.inputs['finanicals']['book_value_of_equity']
        self.base_car = self.inputs['finanicals']['capital_adequacy_ratio'] / 100
        self.base_roe = self.inputs['finanicals']['roe'] / 100
        self.base_cash = self.inputs['finanicals']['cash']
    
    def make_profiles(self):
        # bond yield
        _dict = self.valuation['bond_yield']
        _dict['base'] = self.base_bond_yield
        self.profile_bond_yield = MakeProfile(_dict).run()
        
        # mature market risk premium
        _dict = self.valuation['mature_mkt_erp']
        _dict['base'] = self.base_mature_mkt_risk_premium
        self.profile_mature_market_risk_premium = MakeProfile(_dict).run()
        
        # equity beta
        _dict = self.valuation['beta']
        _dict['base'] = self.base_beta
        self.profile_beta = MakeProfile(_dict).run()
        
        ## RISKFREE RATE AND COST OF EQUITY
        self.riskfree_rate = []
        self.cost_of_equity = []
        for i in range(len(self.profile_bond_yield)):
            bond_yield = self.profile_bond_yield[i]
            mature_mrp = self.profile_mature_market_risk_premium[i]
            beta = self.profile_beta[i]
            crp = self.base_country_risk_premium
            rf = bond_yield - crp
            erp = mature_mrp + crp
            coe = rf + beta * erp
            self.riskfree_rate.append(rf)
            self.cost_of_equity.append(coe)
            
        # asset growth
        _dict = self.valuation['asset_growth']
        _dict['riskfree_rate'] = self.riskfree_rate[-1]
        self.profile_asset_growth = MakeProfile(_dict).run()
        
        # capital adequacy ratio
        _dict = self.valuation['capital_adequacy_ratio']
        _dict['base'] = self.base_car
        self.profile_car = MakeProfile(_dict).run()
        
        # Return on equity
        _dict = self.valuation['roe']
        _dict['base'] = self.base_roe
        _dict['cost_of_equity plus 1'] = self.cost_of_equity[-1] + 0.01
        self.profile_roe = MakeProfile(_dict).run()
        
    def fill_base_dummy(self):
        
        self.asset = [self.base_assets] + [0] * self.n_years
        
        base_reg_cap = self.base_assets * self.base_car
        self.reg_capital = [base_reg_cap] + [0] * self.n_years
        
        self.change_in_regulatory_capital       = [0] * (self.n_years + 1)
        self.net_income                         = [0] * (self.n_years + 1)
        self.investment_in_regulatory_capital   = [0] * (self.n_years + 1)
        self.fcfe                               = [0] * (self.n_years + 1)
        self.cumulative_cost_of_equity          = [1] + [0] * (self.n_years)
        self.pv_of_fcfe                         = [0] * (self.n_years + 1)
        
    def compute(self):
        # ESTIMATE PV of FCFE
        for y in range(1, self.n_years + 1):
            self.asset[y] = (1 + self.profile_asset_growth[y]) * self.asset[y-1]
            self.reg_capital[y] = self.profile_car[y] * self.asset[y]
            self.change_in_regulatory_capital[y] = self.reg_capital[y] - self.reg_capital[y-1]
            self.net_income[y] = self.profile_roe[y] * self.reg_capital[y]
            self.investment_in_regulatory_capital[y] = self.change_in_regulatory_capital[y]
            self.fcfe[y] = self.net_income[y] - self.investment_in_regulatory_capital[y]
            self.cumulative_cost_of_equity[y] = self.cumulative_cost_of_equity[y-1] / (1 + self.cost_of_equity[y])
            self.pv_of_fcfe[y] = self.fcfe[y] * self.cumulative_cost_of_equity[y]
            
        # TERMINAL VALUE
        terminal = self.valuation['terminal_year']
        growth_rate = terminal['growth_rate']
        cost_of_equity = terminal['cost_of_equity']
        if growth_rate == 'riskfree_rate':
            growth_rate = self.riskfree_rate[-1]
        if cost_of_equity == 'cost_of_equity':
            cost_of_equity = self.cost_of_equity[-1]
            
        self.asset.append(self.asset[-1] * (1 + growth_rate))
        self.profile_asset_growth.append(growth_rate)
        self.profile_car.append(self.profile_car[-1])
        self.reg_capital.append(self.asset[-1] * self.profile_car[-1])
        self.change_in_regulatory_capital.append(self.reg_capital[-1] - self.reg_capital[-2])
        self.profile_roe.append(self.profile_roe[-1])
        self.net_income.append(self.profile_roe[-1] * self.reg_capital[-1])
        self.investment_in_regulatory_capital.append(self.change_in_regulatory_capital[-1])
        self.fcfe.append(self.net_income[-1] - self.investment_in_regulatory_capital[-1])
        self.cost_of_equity.append(cost_of_equity)
        self.cumulative_cost_of_equity.append(0)
        self.pv_of_fcfe.append(0)
        
        self.termial_value = self.fcfe[-1] / (cost_of_equity - growth_rate)
        
        self.pv_of_potential_dividends = np.sum(self.pv_of_fcfe)
        self.pv_of_terminal_value = self.termial_value * self.cumulative_cost_of_equity[-2]
        self.value_of_equity = self.pv_of_potential_dividends + self.pv_of_terminal_value
        
        prob_failure = self.valuation['default']['prob_failure']
        proceeds_if_fails = self.valuation['default']['proceeds_if_fails']
        self.value_of_equity = self.value_of_equity * (1 - prob_failure) + prob_failure * proceeds_if_fails
        self.value_of_equity += self.base_cash
        self.shares_outstanding = self.company_info['shares_outstanding']
        self.value_per_share = self.value_of_equity / self.shares_outstanding
        # print(f'value of equity: {self.value_of_equity}')
        # print(f'shares_outstanding: {self.shares_outstanding}')
        # print(f'value_per_share: {self.value_per_share}')
        
    def report(self):
        items = [self.asset,
                 self.profile_asset_growth,
                 self.profile_car,
                 self.reg_capital,
                 self.change_in_regulatory_capital,
                 self.profile_roe,
                 self.net_income,
                 self.investment_in_regulatory_capital,
                 self.fcfe,
                 self.cost_of_equity, 
                 self.cumulative_cost_of_equity,
                 self.pv_of_fcfe
                 ]
        
        index = ['Asset', 'Asset growth rate', 'Capital adequacy ratio',
                 'Regulatory capital', 'Change in regulatory capital', 'Return on Equity',
                 'Net income', 'Investment in regulatory capital',
                 'FCFE (potential dividend)', 'Cost of Equity', 'Cumulative cost of equity',
                 'PV of FCFE']
        columns = ['base_year'] + list(range(1, self.n_years + 1)) + ['terminal_year']
        self.df = pd.DataFrame(items, index = index, columns = columns)
        self.df = self.df.replace(0, '')
        
    def run(self):
        self.base_estimates()
        self.make_profiles()
        self.fill_base_dummy()
        self.compute()
        self.report()
        

if __name__ == '__main__':
    from helper_config import Config
    model_repo = 'models/idfc_first_bank/q2_fy23'
    config = Config(model_repo, is_financial_services = True)
    inputs = config.inputs
    
    valuation = Valuation(inputs)
    valuation.run()