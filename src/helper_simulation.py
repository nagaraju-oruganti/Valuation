from helper_utils import RandomDistributions
from helper_valuation import Valuation
from tqdm.auto import tqdm
import random

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
            'valuation>revenue_growth>next_year': {
                'distribution': 'normal',
                'mean': -0.1,
                'std_dev': 0.05
                }
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
            
    def construct_amend_inputs_dict(self):
        self.amend_inputs_list = []
        path = 'valuation>revenue_growth>next_year'
        sample = random.choice(self.params[path]['samples'])
        self.inputs['valuation']['revenue_growth']['next_year'] = sample
        self.amend_inputs_list.append(sample)
        
    def run_a_iteration(self):
        valuation = Valuation(self.inputs)
        valuation.estimate()
        result = self.amend_inputs_list + [valuation.estimated_value_per_share]
        self.results.append(result)
    
    def run(self, iterations):
        # make distributions
        self.make_distributions()
        
        # run MC iterations
        for _ in tqdm(range(iterations), total = iterations, desc = 'running MC simulations...'):
            self.construct_amend_inputs_dict()
            self.run_a_iteration()