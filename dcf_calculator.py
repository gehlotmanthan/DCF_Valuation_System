import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FinancialDataFetcher:
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        self.info = self.stock.info
        
    def get_financial_statements(self):
        #Fetch income statement, balance sheet, and cash flow statement#
        try:
            income_stmt = self.stock.financials
            balance_sheet = self.stock.balance_sheet
            cash_flow = self.stock.cashflow
            
            return {
                'income_statement': income_stmt,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow
            }
        except Exception as e:
            print(f"Error fetching financial statements: {e}")
            return None
    
    def get_key_metrics(self):
        #Extract key financial metrics from statements
        statements = self.get_financial_statements()
        if not statements:
            return None
            
        income_stmt = statements['income_statement']
        balance_sheet = statements['balance_sheet']
        cash_flow = statements['cash_flow']
        
        # Get last 3 years of data
        years = income_stmt.columns[:3]
        
        metrics = {}
        
        # Revenue and profitability metrics
        metrics['revenue'] = self._safe_extract(income_stmt, 'Total Revenue', years)
        metrics['ebit'] = self._safe_extract(income_stmt, 'EBIT', years)
        metrics['ebitda'] = self._safe_extract(income_stmt, 'EBITDA', years)
        metrics['net_income'] = self._safe_extract(income_stmt, 'Net Income', years)
        metrics['tax_expense'] = self._safe_extract(income_stmt, 'Tax Provision', years)
        
        # Cash flow metrics
        metrics['operating_cf'] = self._safe_extract(cash_flow, 'Operating Cash Flow', years)
        metrics['capex'] = self._safe_extract(cash_flow, 'Capital Expenditure', years)
        metrics['depreciation'] = self._safe_extract(cash_flow, 'Depreciation', years)
        
        # Balance sheet metrics
        metrics['total_debt'] = self._safe_extract(balance_sheet, 'Total Debt', years)
        metrics['cash'] = self._safe_extract(balance_sheet, 'Cash And Cash Equivalents', years)
        metrics['working_capital'] = self._calculate_working_capital(balance_sheet, years)
        
        return metrics
    
    def _safe_extract(self, df, metric_name, years):
        #Safely extract metric data, handling missing values
        try:
            possible_names = [
                metric_name,
                metric_name.replace(' ', ''),
                metric_name.title(),
                metric_name.lower()
            ]
            for name in possible_names:
                if name in df.index:
                    return df.loc[name, years].fillna(0).to_dict()
            return {year: 0 for year in years}
        except:
            return {year: 0 for year in years}
    
    def _calculate_working_capital(self, balance_sheet, years):
        #Calculate working capital for each year#
        wc_data = {}
        for year in years:
            try:
                current_assets = balance_sheet.loc['Current Assets', year] if 'Current Assets' in balance_sheet.index else 0
                current_liabilities = balance_sheet.loc['Current Liabilities', year] if 'Current Liabilities' in balance_sheet.index else 0
                wc_data[year] = current_assets - current_liabilities
            except:
                wc_data[year] = 0
        return wc_data
    
    def get_market_data(self):
        #Get market-related data for WACC calculation#
        try:
            market_cap = self.info.get('marketCap', 0)
            shares_outstanding = self.info.get('sharesOutstanding', 0)
            beta = self.info.get('beta', 1.0)
            
            # Get current stock price
            hist = self.stock.history(period='1d')
            current_price = hist['Close'].iloc[-1] if not hist.empty else 0
            
            return {
                'market_cap': market_cap,
                'shares_outstanding': shares_outstanding,
                'current_price': current_price,
                'beta': beta
            }
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return None

class WACCCalculator:
    #Calculates Weighted Average Cost of Capital
    
    def __init__(self, risk_free_rate=0.045, market_risk_premium=0.065):
        self.risk_free_rate = risk_free_rate  # 10-year treasury rate
        self.market_risk_premium = market_risk_premium  # Historical market premium
    
    def calculate_wacc(self, financial_data, market_data, tax_rate=0.25):
        
        # Cost of Equity using CAPM: (Rf + Beta * (Rm - Rf))
        beta = market_data.get('beta', 1.0)
        cost_of_equity = self.risk_free_rate + beta * self.market_risk_premium
        
        # Get latest debt and equity values
        latest_year = list(financial_data['total_debt'].keys())[0]
        total_debt = abs(financial_data['total_debt'][latest_year])
        market_value_equity = market_data['market_cap']
        
        # Calculate weights
        total_value = total_debt + market_value_equity
        
        if total_value == 0:
            return cost_of_equity  # If no debt, WACC = Cost of Equity
        
        weight_equity = market_value_equity / total_value
        weight_debt = total_debt / total_value
        
        # Estimate cost of debt
        cost_of_debt = self._estimate_cost_of_debt(financial_data)
        
        # Calculate WACC
        wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * (1 - tax_rate))
        
        return {
            'wacc': wacc,
            'cost_of_equity': cost_of_equity,
            'cost_of_debt': cost_of_debt,
            'weight_equity': weight_equity,
            'weight_debt': weight_debt,
            'beta': beta
        }
    
    def _estimate_cost_of_debt(self, financial_data):
        try:
            # Please use actual interest rates
            # For now, I'll use a default rate
            return 0.05  # 5% default cost of debt
        except:
            return 0.05

class DCFModel:
    #Discounted Cash Flow valuation model
    
    def __init__(self, ticker, projection_years=5, terminal_growth_rate=0.025):
        self.ticker = ticker
        self.projection_years = projection_years
        self.terminal_growth_rate = terminal_growth_rate
        
        # Initialize components
        self.data_fetcher = FinancialDataFetcher(ticker)
        self.wacc_calculator = WACCCalculator()
        
        # Fetch data
        self.financial_data = self.data_fetcher.get_key_metrics()
        self.market_data = self.data_fetcher.get_market_data()
        
    def calculate_historical_averages(self):
        #Calculate 3-year averages for key growth metrics
        if not self.financial_data:
            return None
        
        averages = {}
        
        # Revenue growth
        revenues = list(self.financial_data['revenue'].values())
        revenue_growth_rates = []
        for i in range(len(revenues)-1):
            if revenues[i+1] != 0:
                growth = (revenues[i] - revenues[i+1]) / abs(revenues[i+1])
                revenue_growth_rates.append(growth)
        
        averages['revenue_growth'] = np.mean(revenue_growth_rates) if revenue_growth_rates else 0.05
        
        # EBIT margin
        ebit_margins = []
        for year in self.financial_data['revenue'].keys():
            revenue = self.financial_data['revenue'][year]
            ebit = self.financial_data['ebit'][year]
            if revenue != 0:
                ebit_margins.append(ebit / revenue)
        
        averages['ebit_margin'] = np.mean(ebit_margins) if ebit_margins else 0.15
        
        # Tax rate
        tax_rates = []
        for year in self.financial_data['revenue'].keys():
            ebit = self.financial_data['ebit'][year]
            tax_expense = self.financial_data['tax_expense'][year]
            if ebit > 0:
                tax_rates.append(abs(tax_expense) / ebit)
        
        averages['tax_rate'] = np.mean(tax_rates) if tax_rates else 0.25
        
        # Capex as % of revenue
        capex_ratios = []
        for year in self.financial_data['revenue'].keys():
            revenue = self.financial_data['revenue'][year]
            capex = abs(self.financial_data['capex'][year])
            if revenue != 0:
                capex_ratios.append(capex / revenue)
        
        averages['capex_ratio'] = np.mean(capex_ratios) if capex_ratios else 0.05
        
        return averages
    
    def project_cash_flows(self):
        #Project future free cash flows#
        averages = self.calculate_historical_averages()
        if not averages:
            return None
        
        # Get base year revenue (most recent)
        base_revenue = list(self.financial_data['revenue'].values())[0]
        
        projections = []
        
        for year in range(1, self.projection_years + 1):
            # Project revenue
            projected_revenue = base_revenue * (1 + averages['revenue_growth']) ** year
            
            # Project EBIT
            projected_ebit = projected_revenue * averages['ebit_margin']
            
            # Calculate NOPAT (Net Operating Profit After Tax)
            nopat = projected_ebit * (1 - averages['tax_rate'])
            
            # Project Capex
            projected_capex = projected_revenue * averages['capex_ratio']
            
            # Estimate depreciation (simplified as % of revenue)
            depreciation = projected_revenue * 0.03  # 3% of revenue
            
            # Estimate change in working capital (simplified)
            change_in_wc = projected_revenue * 0.02  # 2% of revenue growth
            
            # Calculate Free Cash Flow
            fcf = nopat + depreciation - projected_capex - change_in_wc
            
            projections.append({
                'year': year,
                'revenue': projected_revenue,
                'ebit': projected_ebit,
                'nopat': nopat,
                'capex': projected_capex,
                'depreciation': depreciation,
                'change_in_wc': change_in_wc,
                'fcf': fcf
            })
        
        return projections
    
    def calculate_terminal_value(self, final_fcf, wacc):
        #Calculate terminal value using perpetual growth model#
        terminal_value = (final_fcf * (1 + self.terminal_growth_rate)) / (wacc - self.terminal_growth_rate)
        return terminal_value
    
    def calculate_dcf_valuation(self):
        #Calculate the complete DCF valuation#
        if not self.financial_data or not self.market_data:
            return None
        
        # Calculate WACC
        averages = self.calculate_historical_averages()
        wacc_data = self.wacc_calculator.calculate_wacc(
            self.financial_data, 
            self.market_data, 
            averages['tax_rate'] if averages else 0.25
        )
        wacc = wacc_data['wacc']
        
        # Project cash flows
        projections = self.project_cash_flows()
        if not projections:
            return None
        
        # Calculate present value of projected cash flows
        pv_cash_flows = []
        for projection in projections:
            pv = projection['fcf'] / (1 + wacc) ** projection['year']
            pv_cash_flows.append(pv)
        
        # Calculate terminal value
        final_fcf = projections[-1]['fcf']
        terminal_value = self.calculate_terminal_value(final_fcf, wacc)
        pv_terminal_value = terminal_value / (1 + wacc) ** self.projection_years
        
        # Calculate enterprise value
        enterprise_value = sum(pv_cash_flows) + pv_terminal_value
        
        # Calculate equity value
        latest_year = list(self.financial_data['total_debt'].keys())[0]
        net_debt = self.financial_data['total_debt'][latest_year] - self.financial_data['cash'][latest_year]
        equity_value = enterprise_value - net_debt
        
        # Calculate per share value
        shares_outstanding = self.market_data['shares_outstanding']
        value_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0
        
        # Current stock price for comparison
        current_price = self.market_data['current_price']
        
        return {
            'enterprise_value': enterprise_value,
            'equity_value': equity_value,
            'value_per_share': value_per_share,
            'current_price': current_price,
            'upside_downside': (value_per_share - current_price) / current_price if current_price > 0 else 0,
            'wacc': wacc,
            'terminal_value': terminal_value,
            'pv_terminal_value': pv_terminal_value,
            'pv_cash_flows': sum(pv_cash_flows),
            'projections': projections,
            'wacc_breakdown': wacc_data,
            'assumptions': averages
        }

if __name__ == "__main__":
        app.run(debug=True, port=8050) # type: ignore