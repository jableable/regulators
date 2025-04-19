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


## Dataset Generation

Our dataset combines EPA emissions data (1995-2020) with economic, demographic, and meteorological data—such as GDP, population size, electricity usage, and temperature—as some key covariates of emissions.

## Exploratory Data Analysis



## Modeling Approach

This analysis employs synthetic control methods, which allow one to estimate the effect of an intervention when a direct control group isn’t available. Specifically, a synthetic control is constructed by assigning optimized weights to a set of control states. These weights are chosen to minimize the differences between the synthetic and treated state before the intervention. 

We employ an augmented synthetic control, allowing for both positive and negative weights in order to increase model accuracy. With an intervention date of January 1, 2009, the model was trained on data from 1999 to 2009 and evaluated from 2009 to 2014. The model uses 10 transformed features and a donor pool of 33 non-RGGI control states. Model tuning prioritized the alignment of covariates of the real and synthetic across model features and included efforts to reduce skewness of the data. Finally, the 7 selected RGGI states offer robust data for validation and testing.

## Results



## Future Work



## Description of Repository

