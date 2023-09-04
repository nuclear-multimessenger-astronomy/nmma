import argparse
import json
import os
from ast import literal_eval

import bilby
import bilby.core
import matplotlib
import numpy as np
import pandas as pd
from astropy import time
from bilby.core.likelihood import ZeroLikelihood

from .analysis import get_parser

matplotlib.use("agg")

## TODO: make this script use all arguments from analysis.py, ideally in some dynamic way so it doesn't have to be updated every time analysis.py is updated. The only two arguments that would be different are the model and prior files, which would be based on a list of models/priors provided as an argument for this script
## maybe also have bestfit stored as true in this script, so that it can be used to make plots of the bestfit model for each model/prior combination
## also potentially skip the plotting part of analysis.py, since that will be done here (this might preclude the corner plots from being output, but perhaps we can just call the plot argument since it doesn't really matter if the individual model lightcurves are output or not since they're not all that large in size)

def main():
    
    ## call analysis.py:main() multiple times, but with the model and prior files based on a list of models/priors provided as an argument for this script
    
    ## after the analyses are done, make a combined plot of each model along with the data. Essentially the same thing as the plotting done in analysis.py, but with all models on the same plot
    
    ## may benefit from having an additional file that ranks the models based on the difference in their log evidence/likelihood values, so that the "best" model has the highest difference in log evidence/likelihood compared to the other models, the second "best" model has the second highest difference in log evidence/likelihood compared to the other models, etc. This would streamline analysis rankings.