# Examples

We present two applications of KIM in performing inverse modeling, with Jupyter notebook provided in the repository to guide the package usage. For each case, we developed three types of inverse mappings: (1) the original inverse mapping without knowledge-informed, denoted as $M_0$; (2) the knowledge-informed inverse mapping only using global sensitivity analysis (Step 1), denoted as $M_1$; and (3) the knowledge-informed inverse mapping using both Step 1 and Step 2, denoted as $M_2$. 100 neural networks, $N_e=1$, are trained for each mapping. The remaining configurations can be found in the example jupyter notebook.

## Case 1: Calibrating a cloud chamber model.
Cloud chamber model has been widely applied as a virtual reality of a true cloud chamber to study both turbulence and clouds and reveal aerosol–cloud–turbulence interactions {cite:p}`Wang:2024`. The objective of this example is to estimate two key parameters, i.e., wall fluxes ($\lambda_w$) and collision processes ($\lambda_c$) using inverse mapping. To that, an ensemble of 513 model runs were generated based on a model set up, by varying the values of the two parameters using Sobol sequence. 27 Virtual sensors are configured, each of which 'records' multiple variables including flow properties and cloud properties. The statistics of these variables, calculated over six 5-min periods, are used as the inputs of the inverse mappings, including the temporal standard deviation of vertical velocity ($W$ std), the temporal mean of temperature ($T$ mean), the temporal standard deviation of temperature ($T$ std), the temporal mean of supersaturation ($SS$ mean), the temporal standard deviation of supersaturation ($SS$ std), the mean radius of droplet size distribution (DSD) ($R$ mean, with $R$ representing radius hereafter), the standard deviation of radius ($R$ std), the skewness of radius ($R$ skew), and the kurtosis of radius ($R$ kurt).

- [Train KIM](./im_cloudmodel/kim-holodec.ipynb)
- [Process the training](./im_cloudmodel/postprocessing-holodec.ipynb)

## Case 2: Calibrating an integrated hydrological model.
The Advanced Terrestrial Simulator (ATS) is an integrated hydrological models used to simulate hydrological fluxes across a watershed {cite:p}`Coon:2019`. Here, we calibrated ATS against the streamflow observations at the outlet of Coal Creek watershed, CO, USA. The objective is to estimate eight models parameters categorized into evapotranspiration (ET), snow melting, and subsurface permeability. See {cite:t}`Jiang:2023` for more detailed information.

- [Train KIM](./im_ats/kim.ipynb)
- [Process the training](./im_ats/postprocessing.ipynb)
