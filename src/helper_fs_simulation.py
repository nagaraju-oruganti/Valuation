from helper_utils import RandomDistributions
from helper_fs_valuation import Valuation
from tqdm.auto import tqdm
import random
import copy

#####################################################################
# MONTE-CARLO SIMULATION
#####################################################################    
class MonteCarloSimulation:
    def __init__(self, inputs, params, sample_size = 1000):
        
        '''
            params is dictionary with key (path) and value (distribution properties)
        '''
        
        self.inputs = inputs
        self.params = params
        self.results = []
        self.sample_size = sample_size
        
        self.params = {
            'asset_growth': {'comment': 'Target growth in year 5 between 4 and 4.5 lakh Cr', 
                                    'distribution': 'normal',
                                    'mean': 15,
                                    'std_dev': 2,
                                    'uom': 'percentage'},
            'capital_adequacy_ratio': {'comment': 'Target capital adequacy ratio in year 5 between 8 and 14 %',
                                       'distribution': 'uniform',
                                       'low': 10,
                                       'high': 14,
                                       'uom': 'percentage'},
            'roe': {'comment': 'Target ROE in year 5 is between 10 and 14 %', 
                                    'distribution': 'skewnorm',
                                    'mean': 12,
                                    'std_dev': 2,
                                    'a': 2,
                                    'uom': 'percentage'},
            
            }
        
    def _uniform_distribution(self, properties):
        return self.dist.construct_uniform_samples(
            low = properties['low'], 
            high = properties['high'], 
            sample_size = self.sample_size)
        
    def _normalize_distribution(self, properties):
        return self.dist.construct_normal_samples(
            mean = properties['mean'], 
            std_dev = properties['std_dev'], 
            sample_size = self.sample_size)
        
    def _triang_distribution(self, properties):
        return self.dist.construct_triangular_samples(
            low = properties['low'], 
            high = properties['high'], 
            peak = properties['peak'], 
            sample_size = self.sample_size)
    
    def _skewnorm_distribution(self, properties):
        return self.dist.construct_skewnorm_samples(
            a = properties['a'], 
            mean = properties['mean'], 
            std_dev = properties['std_dev'],
            sample_size = self.sample_size)
        
    def make_distributions(self):
        self.dist = RandomDistributions()
        for path, properties in self.params.items():
            if properties['distribution'] == 'uniform':
                samples = self._uniform_distribution(properties)
            elif properties['distribution'] == 'triang':
                samples = self._triang_distribution(properties)
            elif properties['distribution'] == 'normal':
                samples = self._normalize_distribution(properties)
            elif properties['distribution'] == 'skewnorm':
                samples = self._skewnorm_distribution(properties)
            else:
                samples = []
            
            self.params[path]['samples'] = samples
            
    def construct_amend_inputs_dict(self, inputs):
        self.amend_inputs_list = []
        self.variables = []
        
        path = 'asset_growth'
        scenerio = []
        sample = random.choice(self.params[path]['samples'])
        inputs['valuation']['asset_growth']['growth_phase'][-1] = sample
        inputs['valuation']['asset_growth']['high_growth_phase'][-2] = sample
        scenerio.append(sample)
        self.variables.append(path)
        
        path = 'capital_adequacy_ratio'
        sample = random.choice(self.params[path]['samples'])
        inputs['valuation']['capital_adequacy_ratio']['growth_phase'][-1] = sample
        inputs['valuation']['capital_adequacy_ratio']['high_growth_phase'][-2] = sample
        scenerio.append(sample)
        self.variables.append(path)
        
        path = 'roe'
        sample = random.choice(self.params[path]['samples'])
        inputs['valuation']['roe']['growth_phase'][-1] = sample
        inputs['valuation']['roe']['high_growth_phase'][-2] = sample
        scenerio.append(sample)
        self.variables.append(path)
        
        self.amend_inputs_list = scenerio
        return inputs
    
    def run_a_iteration(self, inputs):
        valuation = Valuation(inputs)
        valuation.run()
        result = self.amend_inputs_list + [valuation.value_per_share]
        self.results.append(result)
    
    def run(self, iterations):
        # make distributions
        self.make_distributions()
        
        # run MC iterations
        for _ in tqdm(range(iterations), total = iterations, desc = 'running MC simulations...'):
            inputs = copy.deepcopy(self.inputs)
            inputs = self.construct_amend_inputs_dict(inputs)
            self.run_a_iteration(inputs)
            
            
if __name__ == '__main__': 
    from helper_config import Config
    model_repo = 'models/idfc_first_bank/q2_fy23'
    config = Config(model_repo, is_financial_services = True)
    inputs = config.inputs
    
    simulation = MonteCarloSimulation(inputs, {})
    simulation.run(100)
    