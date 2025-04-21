# TEAM WHO REGULATES THE REGULATORS?<br>Evaluating CO2 Emissions Reduction Programs

Our primary goal was to ...

<!-- ## About Team Who Regulates the Regulators?
Team members: Jared Able, Joshua Jackson, Zachary Brennan, Alexandria Wheeler, Nicholas Geiser
-->

# Table of Contents
1. [Introduction](#Introduction)
2. [Dataset Generation](#Dataset-Generation)
3. [Exploratory Data Analysis](#Exploratory-Data-Analysis)
4. [Modeling Approach](#Modeling-Approach)
5. [Results](#Results)
6. [Future Work](#Future-Work)
7. [Description of Repository](#Description-of-Repository)

## Introduction

The [Regional Greenhouse Gas Initiative](https://www.rggi.org)(RGGI) is an emissions trading system among northeastern US states to reduce carbon dioxide emissions from electric power generation. This project analyzes the causal effect of RGGI on carbon dioxide emissions from 2009 to 2014 using a synthetic control technique for the states in RGGI during this initial period.  

RGGI's causal effect in each state-year is the difference between actual emissions and the potential emissions that would have occurred had RGGI not been implemented. Estimating RGGI's causal effect therefore requires estimating this potential outcome as well as data on actual emissions in RGGI states. Unfortunately, we do not directly observe this counterfactual outcome. The synthetic control method, originally developed by Abadie and Gardeazabal [(2003)](https://doi.org/10.1257/000282803321455188), estimates the counterfactual by constructing a control on the basis of a weighted combination of actual controls that minimizes the difference between the treated unit and the synthetic control on the outcome variable. These weights are chosen to minimize the differences between the synthetic and treated state before the intervention. 

## Dataset Generation

Our dataset combines EPA emissions data (1995-2020) with economic, demographic, and meteorological data—such as GDP, population size, electricity usage, and temperature—as some key covariates of emissions. A complete description of the variables can be found in `total_state_data_info`


## Exploratory Data Analysis




## Modeling Approach

This analysis employs synthetic control methods, which allow one to estimate the effect of an intervention when a direct control group isn’t available. Specifically, a synthetic control is constructed by assigning optimized weights to a set of control states. These weights are chosen to minimize the differences between the synthetic and treated state before the intervention. 

We employ an augmented synthetic control, allowing for both positive and negative weights in order to increase model accuracy, as described in Ben-Michael, Feller, and Rothstein [(2021)](https://doi.org/10.1080/01621459.2021.1929245). With an intervention date of January 1, 2009, the model was trained on data from 1999 to 2009 and evaluated from 2009 to 2014. The model uses 10 transformed features and a donor pool of 33 non-RGGI control states. Model tuning prioritized the alignment of covariates of the real and synthetic across model features and included efforts to reduce skewness of the data. Finally, the 7 selected RGGI states offer robust data for validation and testing.

## Results



## Future Work



## Description of Repository

