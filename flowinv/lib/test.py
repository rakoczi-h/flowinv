import scipy.stats
import numpy as np
import torch
import pandas as pd
from lib.plot import make_pp_plot
from scipy.spatial.distance import jensenshannon

# ---------------------------- P-P ----------------------------------------------

def p_p_testing(model, truths, y_test, n_test_samples, n_test_cases=100, saveloc='', filename='pp_plot', keys=None, n_params=10):
    """Draws samples from the flow and constructs a p-p plot.
    Parameters
    ----------
        model: torch model
            The model of the flow.
        truths: array
            True parameter values. [no. of test cases, no. of parameters]
        y_test: torch tensor
            Test conditionals. [no. of test cases, no. of survey points]
        n_test_samples: int
            The number of samples to draw for each test case.
        n_test_cases: int
            The number of test cases to use in the p-p plot. Recommended is at least 100.
        saveloc: str
            The location where the image will be saved.
        fileneme: str
            The file name under which it will be saved
        keys: list of str
            The name of the parameters. If not given, then the names will be automatically generated to be ['q1', 'q2', ...]
        n_params: int
            The number of parameters to plot. Only used if keys is not given.
    Output
    ------
        pp plot image saved at saveloc.
    """
    truths = np.array(truths.cpu())
    if np.shape(truths)[1] > n_params:
        indices = np.random.randint(np.shape(truths)[1], size=n_params)
    else:
        indices = np.arange(0, np.shape(truths)[1])
    print(indices)
    if keys == None:
        keys = [f"q{x}" for x in range(n_params)] # number of parameters to get the posterior for (will be 512)
    posteriors = []
    injections = []
    with torch.no_grad():
        for cnt in range(n_test_cases):
            posterior = dict()
            injection = dict()
            x = []
            # y = torch.repeat_interleave(y_test[cnt:cnt+1,:], n_test_samples, dim=0)
            for j in range(n_test_samples):
                x.append(model.sample(1, conditional=y_test[cnt:cnt+1,:]).cpu().numpy()) # sampling the flow
            x = np.vstack(x)
            for i, key in enumerate(keys):
                posterior[key] = x[:,indices[i]]
                injection[key] = truths[cnt,indices[i]]
            posterior = pd.DataFrame(posterior)
            posteriors.append(posterior)
            injections.append(injection)
    print("Calculated results for p-p...")
    make_pp_plot(posteriors, injections, filename=saveloc+filename+'.png')
    print("Made p-p plot...")

# ---------------------------- DIVERGENCES ---------------------------------------

def KL_divergence_latent(latent_samples):
    """Testing the Gaussianity of the latent space. The measure is the KL divergence between the approximate pdf of the latent space and a unit gaussian pdf.
    Parameters
    ----------
        latent_samples: array
            Samples from the latent space. [no.of samples, no. of parameters]
    Output
    ------
        mean_kldiv: float
            The mean KL-divergence for all dimensions of the latent space
        std_kldiv: float
            The standard deviation of the KL-divergence.
    """
    distance = []
    kdes = []
    x_grids = []
    n = 500 # the number of grid values we are calculating the values of the distributions
    xmin = np.min(latent_samples[:,0])
    xmax = np.max(latent_samples[:,0])
    x_grid = np.arange(xmin, xmax+((xmax-xmin)/n), (xmax-xmin)/n) # the grid values we are using
    q = lambda x: np.exp(-x**2/2)/np.sqrt(2*np.pi) # unit gaussian pdf
    q_x = q(x_grid)
    for i in range(np.shape(latent_samples)[1]): # looping over the latent space dimensions
        # using KDE: kernel density estimation (should trace the edges of the sample hist)
        p = scipy.stats.gaussian_kde(latent_samples[:,i]) # approximate pdf of the latent space
        p_x = p.evaluate(x_grid)
        f = p_x*np.log(p_x/q_x) # kl divergence function
        kldiv = scipy.integrate.simps(f, x_grid)
        if np.isnan(kldiv).any():
            continue
        kdes.append(p_x)
        distance.append(kldiv)
    kdes = np.array(kdes)
    distance = np.array(distance)
    mean_kldiv = np.mean(distance)
    std_kldiv = np.std(distance)
    return mean_kldiv, std_kldiv

def JS_divergence_2samplers(samples_p, samples_q):
    """Function calculating the Jensen-Shannon divergence between two distributions.
    The p(x) and q(x) functions are calculated using a KDE of the input samples.
    This is done for each dimension seperately.
    Parameters
    ----------
        samples_p: array
            Samples from the first sampler. [no. of samples, no. of dimensions]
        samples_q: array
            Samples from the second sampler. [no. of samples, no. of dimensions]
    Output
    ------
        js: list of floats
            The list of JS-divergence values with length of the no. of parameters/dimensions.
    """
    if not np.shape(samples_p)[1] == np.shape(samples_q)[1]:
        raise ValueError('The two sample sets do not have the same number of dimensions.')
    n = 500 # the number of grid values we are calculating the values of the distributions
    js = []
    for dim in range(np.shape(samples_p)[1]):
        xmin = min([np.min(samples_p[:,dim]), np.min(samples_q[:,dim])])
        xmax = max([np.max(samples_p[:,dim]), np.max(samples_q[:,dim])])
        # calculate the minimum and maximum from both
        x_grid = np.arange(xmin, xmax+((xmax-xmin)/n), (xmax-xmin)/n) # the grid values we are using
        p = scipy.stats.gaussian_kde(samples_p[:,dim])
        p_x = p.evaluate(x_grid)
        q = scipy.stats.gaussian_kde(samples_q[:,dim])
        q_x = q.evaluate(x_grid)
        js_pq = np.nan_to_num(np.power(jensenshannon(p_x, q_x), 2))
        js.append(js_pq)
    return js

# ---------------------------- SAMPLING ----------------------------------------

def sample(test_samples_n, conditional, model):
    """Drawing samples from the flow.
    Parameters
    ----------
        test_samples_n: int
            The number of test samples we want to draw (for the same conditional input)
        conditional: array,
            The conditional, a single survey. [no. of survey points]
        model: torch model
            The neural network model (flow)
    Output
    ------
        samples: array
            Samples from the output space of the flow. [no. of samples, no. of parameters]
    """
    samples = []
    with torch.no_grad():
         for i in range(test_samples_n):
             samples.append(np.array(model.sample(1, conditional=conditional).cpu()))
    samples = np.vstack(samples)
    return samples

def sample_and_logprob(test_samples_n, conditional, model):
    """Drawing samples from the flow and returning their corresponding log probabilties too.
    Parameters
    ----------
        test_samples_n: int
            The number of test samples we want to draw (for the same conditional input)
        conditional: array,
            The conditional, a single survey. [no. of survey points]
        model: torch model
            The neural network model (flow)
    Output
    ------
        samples: array
            Samples from the output space of the flow. [no. of samples, no. of parameters]
    """
    samples = []
    log_probs = []
    with torch.no_grad():
        for i in range(test_samples_n):
            s, l = model.sample_and_log_prob(1, conditional=conditional)
            samples.append(np.array(s.cpu()))
            log_probs.append(np.array(l.cpu()))
    samples = np.vstack(samples)
    log_probs = np.vstack(log_probs)
    return samples, log_probs


def forward(data, conditional, model):
    """Drawing samples from the latent space.
    Parameters
    ----------
        data: tensor
            Some density models we use for testing, can come from training set, needs to be a tensor and be on the same device as the model. [no. of samples desired, no. of parameters]
        conditional: tensor
            The corresponding conditionals for each density model.
        model: torch model
            The flow neural network model.
    Output
    ------
         z_: array
            Laten space samples. Same shape as data.
    """
    with torch.no_grad():
         z_, _ = model.forward(data, conditional)
    z_ = z_.cpu().numpy()
    return z_


def forward_and_logprob(x_train, y_train, model):
    """Drawing samples from the latent space and returning their corresponding log probabilties too
    Parameters
    ----------
        data: tensor
            Some density models we use for testing, can come from training set, needs to be a tensor and be on the same device as the model. [no. of samples desired, no. of parameters]
        conditional: tensor
            The corresponding conditionals for each density model.
        model: torch model
            The flow neural network model.
    Output
    ------
        z_: array
            Laten space samples. Same shape as data.
        log_prob: array
            The log probabilties of each sample. [no. of samples]
    """
    with torch.no_grad():
        z_, log_prob = model.forward_and_log_prob(x_train[:20000, :], y_train[:20000, :])
    z_ = z_.cpu().numpy()
    log_prob = log_prob.cpu().numpy()
    return z_, log_prob



