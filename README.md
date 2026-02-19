# Electric Motor Temperature Prediction using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-success)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Predict the **Permanent Magnet (rotor) surface temperature** and other motor temperatures using sensor data from a Permanent Magnet Synchronous Motor (PMSM) test bench.

## Project Overview

This project uses machine learning to estimate hard-to-measure temperatures in an electric motor (especially the rotor temperature `pm`), which is critical for real-time monitoring, thermal protection, and efficiency optimization in electric vehicles and industrial drives.

### Dataset Source
- **Source**: Kaggle – Electric Motor Temperature Dataset  
- **Link**: [https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature](https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature)  
- **Description**: ~1.3 million samples collected over 185 hours at 2 Hz sampling rate from a real PMSM test bench. Includes electrical (d/q voltages & currents), mechanical (speed, torque), and thermal measurements (ambient, coolant, stator & rotor temperatures).  
- **Paper Reference** (recommended citation):  
  Kirchgässner, W. et al. (2020). "Deep Residual Learning for Temperature Estimation of Permanent Magnet Synchronous Motors." IEEE Transactions on Industrial Electronics.

### Key Features
- Ambient temperature
- Coolant temperature
- Voltage d/q components (u_d, u_q)
- Current d/q components (i_d, i_q)
- Motor speed
- Target: Permanent Magnet surface temperature (`pm`) – main prediction target

### Models Trained & Compared
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regression (SVR) / LinearSVR

Model not saved because of github's file size constraints



