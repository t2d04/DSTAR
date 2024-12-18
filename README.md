This repository is a fork of DSTAR from Seoin Back (https://github.com/SeoinBack/DSTAR), and currently under development due to an issue where the finalized models cannot be exported as pickle files.

In addition to the original models in DSTAR, I have integrated DNN, LightGBM, and CatBoost models. Furthermore, 247 simple ensemble models, combining GBR, KRR, ELN, SVR, XGBoost, DNN, LightGBM, and CatBoost, have been added for enhanced performance evaluation.


# DSTAR : **D**ft & STructure free Active motif based Representation

This repository contains codes and notebooks used to create results in our paper.

For more details, check out this paper (https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.1c00726).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Usage](#usage)

## Prerequisites
* Generalized Adsorption Simulator for Python ([GASpy](https://github.com/ulissigroup/GASpy))

* Additional packages required for gaspy enviroment:
- [scikit-learn](http://scikit-learn.org/stable/) (0.24.2)
- [pymatgen](http://pymatgen.org) (2021.3.3)

## Usage of ML model
See DSTAR_Guide_.pdf.

## Application
DSTAR can be utilized to discover catalyst for various electrochemical reactions.
You can reproduce the catalyst discovery for each reaction with the following descriptions.

### CO2RR
#### Usage
To reproduce DSTAR application for CO2RR, please refer to three ipynb files in `application/CO2RR/`. Each ipynb file will do the following: 

`01_Scaler.ipynb` will generates scaler to normalize the productivity.

`02_Heatmap.ipynb` will visualize productivty heatmap.

`03_Selectivity_Plot.ipynb` will plot the productivity for each product corresponding to applied potential, composition and coordination number.

More details are in each ipynb files.

#### Data 
All predicted CO* / H* / OH* binding energies and coordination number of prototype surface used for application can be found in `application/CO2RR/data/energy` and `application/CO2RR/data/CN_dict.pkl`, and the boundary conditions of selectivity map are in `application/CO2RR/script/condition.py`.

