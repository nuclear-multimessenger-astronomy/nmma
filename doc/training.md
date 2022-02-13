
## Training

It is common to have light curves on "grids", for which you have a discrete set of parameters for which the lightcurves were simulated. For example, we may know the lightcurves to expect for specific masses m_1 and m_2, but not for any masses between the two.

We rely on sampling from a grid of modeled lightcurves through the use of Principle Component Analysis (PCA) and an interpolation scheme (either Gaussian process modeling or neural networks). The PCA serves to represent each light curve by a small number of "eigenvalues", rather than the full lightcurve. After performing PCA, you will have a discrete grid of models that relate merger parameters to a few **lightcurve eigenvalues** rather than the whole lightcurve.

At this point, you can model this grid as either a Gaussian process or Neural Network. This will allow you to form a **continuous map** from merger parameters to lightcurve eigenvalues, which are then converted directly to the set of light curve parameters that most likely resulted in this lightcurve.


### NMMA training

There are helper functions within NMMA to support this. In particular, `nmma.em.training.SVDTrainingModel` is designed to take in a grid of models and return an interpolation class.

See nmma.tests.training.py for a simple example
