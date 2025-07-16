import bilby
import pandas as pd

num_points = 121

def cast_as_bilby_result(samples, truth, priors):
    
    posterior = dict.fromkeys({'log10_mej', 'log10_vej', 'log10_Xlan'})
    samples_numpy = samples.numpy()
    posterior['log10_mej'] = samples_numpy.T[0].flatten()
    posterior['log10_vej'] = samples_numpy.T[1].flatten()
    posterior['log10_Xlan'] = samples_numpy.T[2].flatten()
    posterior = pd.DataFrame(posterior)
    
    if truth == None:
        result = bilby.result.Result(
            label="test_data",
            posterior=posterior,
            search_parameter_keys=list(posterior.keys()),
            priors=priors
        )
    else: 
        injections = dict.fromkeys({'log10_mej', 'log10_vej', 'log10_Xlan'})
        injections['log10_mej'] = float(truth.numpy()[0])
        injections['log10_vej'] = float(truth.numpy()[1])
        injections['log10_Xlan'] = float(truth.numpy()[2])

        result = bilby.result.Result(
            label="test_data",
            injection_parameters=injections,
            posterior=posterior,
            search_parameter_keys=list(posterior.keys()),
            priors=priors
        )
    
    return result
