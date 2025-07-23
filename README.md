
---

# `PredictiveModels` contains the codes for three projects in its subfolders: WFAR_code, taxi_market_code, and CFPCA_code.

## Description  
### 1. WFAR_code (main function: WFAR.m – MATLAB code for Warping Functional AutoRegressive (WFAR) Model)  
This MATLAB code proposes a noval time series forecasting model (namely, **WFAR model**) for high dimensional, high frequency and seasonal time series data, in particular, the 1-day ahead electricity price. The code further performs a comprehensive comparison of various time series forecasting models e.g. FAR, VAR, AR*, SAR, ARX* models against the proposed WFAR model.

The electricity price data from two power markets are used: (1) the Nord Pool market from 2013.01.01 to 2017.12.31, (2) the California market from 1999.07.05 to 2001.01.31. This code was originally published with the paper: "Modeling Seasonality and Serial Dependence of Electricity Price Curves with Warping Functional Autoregressive Dynamics" by The Annals of Applied Statistics in 2019.


### 2. taxi_market_code (main function: AnalyseTransMktInefficiency_Stage1to3.Rmd - R code for Taxi Market Inefficiency Analysis: Mismatched Trips 2015-01-01 to 2017-06-30)
This project partitions a day into sub-time-intervals and analyse the taxi drivers’ behaviours to shed light on Singapore local transport market efficiency in each sub-time-interval in a day. For the first time, by establishing the market inefficiency metrics with optimized calculation algorithm this project analyzes taxi operational statuses in each 5-minute intervals (288 intervals/day) across 170,478 unique taxi IDs operating in Singapore during 2015-01-01 to 2017-06-30.

This project was funded and in collabration with LTA. The local taxi operators involved in the analyzed taxi data are: CDG, Premier, SMRT, TransCab.

### 3. CFPCA_code (the CFPCA_codes for 1-day ahead International Yield Curve Prediction)
This project proposes an international yield curve predictive model, where common
factors are identified using the common functional principal component (**CFPC**)
method. For the 1-day ahead out-of-sample forecasts of the US, Sterling, Euro
and Japanese yield curve from 07 April 2014 to 06 April 2015, the CFPC factor
model is compared with several alternative factor models based on the functional principal
component analysis. 

The code was first published with the paper "International Yield Curve Prediction with Common Functional Principal Component Analysis" by Robustness in Econometrics in 2017.

## Notable Features:
The **WFAR model** handles intraday patterns through time warping. the codes implements WFAR and provides a rigorous comparison of functional data approaches and traditional autoregressive/econometric models for electricity price forecasting.

   1. The time warping approach splits the typical daily price pattern (seasonal) variations by day of week from the price level variations.
   2. The function autoregressive model (FAR) is applied on the seasonal-adjusted price values.
   3. The day of week effects (calendar effects) are properly addressed through fixed warping function, as shown by the improvement of forecast accuracy.
   4. The WFAR model is fast and applicable to other seasonal time series in financial market.
  
The **taxi market inefficiency project**. 
   1. Noval Analysis Framework of Taxi Status Types and Market Inefficiency Metrics
   2. Granualar Taxi Status Identification up to each 5-min interval across 55 local markets (planning areas of Singapore) during 2.5 years
   3. Data heavy and computationally heavy
   4. Algorithm optimization via vectorization and profiling tools

The **CFPCA model**.
  1. Computes and compares several yield curve modeling techniques.
    - Dynamic Nelson-Siegel (DNS)
    - Common Principal Components (CPC)
    - Combined Functional Principal Components Analysis (combFPCA)
    - Common Functional Principal Components (CFPC)
 2. Generates 1-day ahead and 5-days-ahead forecasts using AR(1) framework with error elevaluation.
  

## Usage  
Refer to each README.md file in the subfolders.

## Author & Contact  
- **Author**: Jiejie Zhang
- **Date**: 2025-03-30
- **Contact**: jiejiezhangsta@gmail.com

---
