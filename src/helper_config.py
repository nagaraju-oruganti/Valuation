import yaml

################################################################
# Model configuration
################################################################
class Config:
    def __init__(self, model_repo, is_financial_services = False):
        self.model_repo = model_repo
        self.is_financial_services = is_financial_services
        self.inputs = self.load_inputs()
        
        # # Company info
        # self.load_company_info()
        
        # # Market drivers
        # self.load_market_drivers()
        
    def load_inputs(self):
        '''
            Read inputs in yaml format from the model repository
            and save them in `inputs` variable to decimate information to use in 
            downstream processes
        '''
        path = f'{self.model_repo}/inputs.yaml'
        if self.is_financial_services:
            path = f'{self.model_repo}/fs_inputs.yaml'
        with open(path, 'r') as file_descriptor:
            return yaml.safe_load(file_descriptor)
        
    def load_company_info(self):
        info = self.inputs.get('company_info', None)
        if info is not None:
            self.company_name = info['name']
            self.ticker = info['ticker']
            self.industry = info['industry']
            self.sector = info['sector']
            
    def load_market_drivers(self):
        mkt_drivers = self.inputs['market_drivers']
        self.risk_free_rate = mkt_drivers['risk_free_rate']