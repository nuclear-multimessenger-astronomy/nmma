<p align="center">
    <picture>
      <source media="(prefers-color-scheme: dark)" width="600px" height="300px" srcset="https://github.com/nuclear-multimessenger-astronomy/nmma/blob/main/doc/images/dark-logo.svg">
      <source media="(prefers-color-scheme: light)" width="600px" height="300px" srcset="https://github.com/nuclear-multimessenger-astronomy/nmma/blob/main/doc/images/light-logo.svg">
      <img alt="NMMA">
    </picture>
</p>


<div align="center">
   <h1>NMMA</h1>
   <h2>a pythonic library for probing nuclear physics and cosmology with multimessenger analysis</h2>
   <br/><br/>
</div>


[![GitHub Repo stars](https://img.shields.io/github/stars/nuclear-multimessenger-astronomy/nmma?style=flat)](https://github.com/nuclear-multimessenger-astronomy/nmma/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/nuclear-multimessenger-astronomy/nmma?style=flat&color=%2365c563)](https://github.com/nuclear-multimessenger-astronomy/nmma/forks)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/nmma?label=conda%20downloads)](https://anaconda.org/conda-forge/nmma)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/nmma?label=PyPI%20downloads)](https://badge.fury.io/py/nmma)
[![Coverage Status](https://coveralls.io/repos/github/nuclear-multimessenger-astronomy/nmma/badge.svg?branch=main)](https://coveralls.io/github/nuclear-multimessenger-astronomy/nmma?branch=main)
[![CI](https://github.com/nuclear-multimessenger-astronomy/nmma/actions/workflows/continous_integration.yml/badge.svg)](https://github.com/nuclear-multimessenger-astronomy/nmma/actions/workflows/continous_integration.yml)
[![PyPI version](https://badge.fury.io/py/nmma.svg)](https://badge.fury.io/py/nmma)
[![Python version](https://img.shields.io/pypi/pyversions/nmma.svg)](https://badge.fury.io/py/nmma)




Citations to the NMMA code: [Citation record](https://inspirehep.net/literature?sort=mostrecent&size=250&page=1&q=refersto%3Arecid%3A2083145&ui-citation-summary=true)

Read our official documentation: [NMMA Documentation](https://nuclear-multimessenger-astronomy.github.io/nmma/)

Check out our contribution guide: [For contributors](https://nuclear-multimessenger-astronomy.github.io/nmma/contributing.html)

A tutorial on how to produce simulations of lightcurves is given here [tutorial-lightcurve_simulation.ipynb](https://github.com/nuclear-multimessenger-astronomy/nmma/blob/main/tutorials/tutorial-lightcurve_simulation.ipynb)


### Citing NMMA

When using this code for a publication, kindly make a reference to the package by its name, NMMA, and a citation to the companion paper [An updated nuclear-physics and multi-messenger astrophysics framework for binary neutron star mergers](https://www.nature.com/articles/s41467-023-43932-6). The BibTeX entry for the paper is:
```bibtex
@article{Pang:2022rzc,
      title={An updated nuclear-physics and multi-messenger astrophysics framework for binary neutron star mergers},
      author={Peter T. H. Pang and Tim Dietrich and Michael W. Coughlin and Mattia Bulla and Ingo Tews and Mouza Almualla and Tyler Barna and Weizmann Kiendrebeogo and Nina Kunert and Gargi Mansingh and Brandon Reed and Niharika Sravan and Andrew Toivonen and Sarah Antier and Robert O. VandenBerg and Jack Heinzel and Vsevolod Nedora and Pouyan Salehi and Ritwik Sharma and Rahul Somasundaram and Chris Van Den Broeck},
      journal={Nature Communications},
      year={2023},
      month={Dec},
      day={20},
      volume={14},
      number={1},
      pages={8352},
      issn={2041-1723},
      doi={10.1038/s41467-023-43932-6},
      url={https://doi.org/10.1038/s41467-023-43932-6}
}
```
Since NMMA uses bilby as its backend, please also cite the bilby companion paper [BILBY: A user-friendly Bayesian inference library for gravitational-wave astronomy](https://arxiv.org/pdf/1811.02042). The BibTeX entry for the paper is:
```bibtex
@article{Ashton:2018jfp,
    author = "Ashton, Gregory and others",
    title = "{BILBY: A user-friendly Bayesian inference library for gravitational-wave astronomy}",
    eprint = "1811.02042",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    doi = "10.3847/1538-4365/ab06fc",
    journal = "Astrophys. J. Suppl.",
    volume = "241",
    number = "2",
    pages = "27",
    year = "2019"
}
```
If the gravitational-wave component of NMMA has been used, please see [here](https://bilby-dev.github.io/bilby/citing-bilby.html) for the exact citation guideline, depending on your use case.

If you are using the systematics error, please also cite the paper [Data-driven approach for modeling the temporal and spectral evolution of kilonova systematic uncertainties](https://arxiv.org/abs/2410.21978). The BibTeX entry for the paper is:
```bibtex
@article{Jhawar:2024ezm,
    author = "Jhawar, Sahil and Wouters, Thibeau and Pang, Peter T. H. and Bulla, Mattia and Coughlin, Michael W. and Dietrich, Tim",
    title = "{Data-driven approach for modeling the temporal and spectral evolution of kilonova systematic uncertainties}",
    eprint = "2410.21978",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    doi = "10.1103/PhysRevD.111.043046",
    journal = "Phys. Rev. D",
    volume = "111",
    number = "4",
    pages = "043046",
    year = "2025"
}
```

If you are using likelihood free inference, please also cite the paper [Rapid parameter estimation for kilonovae using likelihood-free inference](https://www.arxiv.org/abs/2408.06947). The BibTeX entry for the paper is:
```bibtex
@article{Desai:2024hlp,
    author = "Desai, Malina and Chatterjee, Deep and Jhawar, Sahil and Harris, Philip and Katsavounidis, Erik and Coughlin, Michael",
    title = "{Kilonova Light Curve Parameter Estimation Using Likelihood-Free Inference}",
    eprint = "2408.06947",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    doi = "10.1093/mnras/staf1045",
    month = "8",
    year = "2024"
}
```

Based on the sampler used for the analysis, please also cite the associated papers:

dynesty: Zenodo entry [10.5281/zenodo.3348367](https://zenodo.org/records/17268284) and the associated companion paper as follow:
```bibtex
@article{Speagle:2019ivv,
    author = "Speagle, Joshua S.",
    title = "{dynesty: a dynamic nested sampling package for estimating Bayesian posteriors and evidences}",
    eprint = "1904.02180",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    doi = "10.1093/mnras/staa278",
    journal = "Mon. Not. Roy. Astron. Soc.",
    volume = "493",
    number = "3",
    pages = "3132--3158",
    year = "2020"
}
```

PyMultiNest:
```bibtex
@article{Buchner:2014nha,
    author = "Buchner, J. and Georgakakis, A. and Nandra, K. and Hsu, L. and Rangel, C. and Brightman, M. and Merloni, A. and Salvato, M. and Donley, J. and Kocevski, D.",
    title = "{X-ray spectral modelling of the AGN obscuring region in the CDFS: Bayesian model selection and catalogue}",
    eprint = "1402.0004",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    doi = "10.1051/0004-6361/201322971",
    journal = "Astron. Astrophys.",
    volume = "564",
    pages = "A125",
    year = "2014"
}
```
UltraNest:
```bibtex
@article{Buchner:2021cql,
    author = "Buchner, Johannes",
    title = "{UltraNest -- a robust, general purpose Bayesian inference engine}",
    eprint = "2101.09604",
    archivePrefix = "arXiv",
    primaryClass = "stat.CO",
    reportNumber = "10.21105/joss.03001",
    month = "1",
    year = "2021"
}
```


### Acknowledgments
If you benefited from participating in our community, we ask that you please acknowledge the Nuclear Multi-Messenger Astronomy collaboration, and particular individuals who helped you, in any publications.
Please use the following text for this acknowledgment:
  > We acknowledge the Nuclear Multi-Messenger Astronomy collective as an open community of multi-domain experts and collaborators. This community and \<names of individuals\>, in particular, were important for the development of this project.

### Funding
We gratefully acknowledge previous and current support from the U.S. National Science Foundation (NSF) Harnessing the Data Revolution (HDR) Institute for <a href="https://a3d3.ai">Accelerating AI Algorithms for Data Driven Discovery (A3D3)</a> under Cooperative Agreement No. <a href="https://www.nsf.gov/awardsearch/showAward?AWD_ID=2117997">PHY-2117997</a> and the European Research Council (ERC) under the European Union's Starting Grant (Grant No. <a href="https://doi.org/10.3030/101076369">101076369</a>).

<p align="center">
<img src="https://github.com/nuclear-multimessenger-astronomy/nmma/blob/main/assets/a3d3.png" alt="A3D3" width="200"/>
<img src="https://github.com/nuclear-multimessenger-astronomy/nmma/blob/main/assets/nsf.png" alt="NSF" width="200"/>
<img src="https://github.com/nuclear-multimessenger-astronomy/nmma/blob/main/assets/erc.png" alt="ERC" width="200"/>
