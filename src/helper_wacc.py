from helper_utils import (EquityRiskPremiumEstimator, 
                          UnleveredBetaEstimator, 
                          OperatingLeaseConverter,
                          PreTaxCostOfDebtEstimator)
import pandas as pd
import numpy as np

######################################################################
# WEIGHTED AVERAGE COST OF CAPITAL
# 1) Cost of equity
# 2) Cost of debt
# 3) Cost of preferred stock
######################################################################

class CostOfCapital:
    def __init__(self, inputs):
        self.inputs = inputs
        
        self.risk_free_rate = self.inputs['market_drivers']['risk_free_rate']
        self.marginal_tax_rate = self.inputs['market_drivers']['marginal_tax_rate']
        
    def estimate_components(self):
        
        #---- PRE-TAX COST OF DEBT
        self.cod_estimator = PreTaxCostOfDebtEstimator(self.inputs)        
        self.cod_estimator.estimate()
        self.pre_tax_cost_of_debt = self.cod_estimator.pre_tax_cost_of_debt
        
        #---- OPERATING LEASE TO DEBT CONVERTOR
        self.op_lease_estimator = OperatingLeaseConverter(self.pre_tax_cost_of_debt, self.inputs)
        self.op_lease_estimator.estimate()
        
        #---- UNLEVERED BETA
        self.unlevered_beta_estimator = UnleveredBetaEstimator(self.inputs)
        self.unlevered_beta_estimator.estimate()
        
        #---- MARKET VALUE OF EQUITY
        self.mv_equity = self.inputs['equity_inputs']['current_stock_price'] * self.inputs['equity_inputs']['shares_outstanding']
        
        #---- MARKET VALUE OF DEBT
        # MV of debt = MV of straight debt + Est. value of straight debt in convertible + Value of debt in operating leases
        bv_straight_debt = self.inputs['finanicals']['book_value_debt']
        avg_maturity = self.inputs['debt']['average_maturity']
        
        # interest portion of debt
        interest_portion = self.cod_estimator.interest_expense * (1 - (1 + self.pre_tax_cost_of_debt) ** (-avg_maturity)) / self.pre_tax_cost_of_debt
        principle_portion = bv_straight_debt / (1 + self.pre_tax_cost_of_debt)**(avg_maturity)
        
        mv_straight_debt = interest_portion + principle_portion
        
        convertible_debt = 0 #TODO: write a class to estimate convertible debt
        operating_lease_debt = self.op_lease_estimator.adjust_total_debt
        self.mv_debt = mv_straight_debt + convertible_debt + operating_lease_debt
        
        #---- ESTIMATE LEVERED BETA (EQUITY BETA)
        unlevered_beta = self.unlevered_beta_estimator.unlevered_beta
        self.levered_beta = unlevered_beta * (1 + (1-self.marginal_tax_rate) * (self.mv_debt / self.mv_equity))
        
        #---- VALUE OF PREFERRED STOCK
        self.mv_preferred_stock = self.inputs['finanicals']['preferred_shares'] * self.inputs['equity_inputs']['current_stock_price']
        
        #--- Equity risk premium
        self.erp_estimator = EquityRiskPremiumEstimator(self.inputs)
        self.erp_estimator.estimate()
        self.equity_risk_premium = self.erp_estimator.erp
    
    def estimate(self):
        
        # calculate components to estimate discount rates
        self.estimate_components()
        
        '''
            --- COST OF EQUITY
                cost of equity = riskfree rate + market risk premium x levered beta
            --- COST OF DEBT
                cost of debt  = (1 - marginal tax rate) * pre-tax cost of debt
            --- COST OF PREFREDED STOCK
                cost of preferred stock = annual dividend / current stock price
        '''
        self.cost_of_equity = self.risk_free_rate + self.equity_risk_premium * self.levered_beta
        self.cost_of_debt = (1 - self.marginal_tax_rate) * self.pre_tax_cost_of_debt
        self.cost_of_preferred_stock = self.inputs['finanicals']['annual_dividend'] / self.inputs['equity_inputs']['current_stock_price']
        
        self.df = pd.DataFrame(
            np.transpose([[self.mv_equity, self.mv_debt, self.mv_preferred_stock],
                          [self.cost_of_equity, self.cost_of_debt, self.cost_of_preferred_stock]]),
            columns = ['market_value', 'cost_of_component'])
        self.df['weight'] = self.df['market_value'] / np.sum(self.df['market_value'])
        self.df = self.df.T
        self.df.columns = ['equity', 'debt', 'preferred_stock']
        self.df['capital'] = [
            np.sum(self.df.loc['market_value', :]),
            np.sum(self.df.loc['cost_of_component', :] * self.df.loc['weight', :]),
            np.sum(self.df.loc['weight', :]),
        ]
        
        self.capital = self.df.loc['market_value', 'capital']
        self.wacc = self.df.loc['cost_of_component', 'capital']