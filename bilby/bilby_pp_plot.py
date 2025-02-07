import json
import os
import numpy as np
import pandas as pd
from giflow.plot import make_pp_plot

bilby_location = '/data/www.astro/2263373r/giflow/4_paper/bilby/100_testcases/'

keys = [r'$c_x$', r'$c_y$', r'$c_z$', r'$l_x$', r'$l_y$', r'$l_z$', r'$\alpha$']
bilby_samples = []
injection_parameters = []
for i in range(100):
    with open(os.path.join(bilby_location, f"testcase_{i}/inversion_result.json"), 'r') as file:
        bilby_results = json.load(file)
        bilby_posterior_dict = bilby_results['posterior']['content']
        bilby_posterior_dict.pop('log_likelihood')
        bilby_posterior_dict.pop('log_prior')
        print(bilby_posterior_dict.keys())
        b_pos = pd.DataFrame(bilby_posterior_dict)
        bilby_samples.append(b_pos)
        injection_parameters.append(bilby_results['injection_parameters'])
make_pp_plot(bilby_samples, injection_parameters, filename=os.path.join(bilby_location, "pp_plot.png"), labels=keys)
