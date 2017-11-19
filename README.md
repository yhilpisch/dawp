# Derivatives Analytics with Python (Wiley Finance)

This repository provides all Python codes and Jupyter Notebooks of the book _Derivatives Analytics with Python_ by Yves Hilpisch.

<img src="http://hilpisch.com/images/derivatives_analytics_front.jpg" width="500">

Oder the book here http://eu.wiley.com/WileyCDA/WileyTitle/productCd-1119037999.html or here www.amazon.com/Derivatives-Analytics-Python-Simulation-Calibration/dp/1119037999/.

There are two code versions available: for **Python 3.6** and **Python 2.7** (not maintained anymore).

## Python Packages

There is now a `yaml` file for the installation of required Python packages in the repository. This is to be used with the `conda` package manager (see https://conda.io/docs/user-guide/tasks/manage-environments.html). If you do not have Miniconda or Anaconda installed, we recommend to install **Miniconda 3.6** first (see https://conda.io/miniconda.html).

After you have cloned the repository, do on the **Linux/Mac** shell:

    cd dawp
    conda env create -f dawp_conda.yml
    source activate dawp
    cd python36
    jupyter notebook

On **Windows**, do:

    cd dawp
    conda env create -f dawp_conda.yml
    activate dawp
    cd python36
    jupyter notebook

Then you can navigate to the Jupyter Notebook files and get started.

## Python Quant Platform

You can immediately use all codes and Jupyter Notebooks by registering on the Quant Platform under http://wiley.quant-platform.com.

## Python for Algorithmic Trading Course & Certificate

<img src="http://hilpisch.com/images/finaince_visual_low.png" width="500">

Check out our **Python for Algorithmic Trading Course** under http://pyalgo.tpq.io.

Check out also our **Python for Algorithmic Trading Certificate Program** under http://certificate.tpq.io.


## Company Information

© Dr. Yves J. Hilpisch \| The Python Quants GmbH

The Quant Platform (http://pqp.io) and all codes/Jupyter notebooks come with no representations or warranties, to the extent permitted by applicable law.

http://tpq.io \| dawp@tpq.io \|
http://twitter.com/dyjh

**Quant Platform** \| http://wiley.quant-platform.com

**Derivatives Analytics with Python (Wiley Finance)** \|
http://derivatives-analytics-with-python.com

**Python for Finance (O'Reilly)** \|
http://python-for-finance.com

**Python for Algorithmic Trading Course** \|
http://pyalgo.tpq.io

**Python for Finance Online Training** \|
http://training.tpq.io
