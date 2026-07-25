# 🇧🇼 Economic Intelligence Framework for Food Inflation Forecasting

> **Deep Learning IndabaX Botswana 2026 Hackathon**  
> Repository for the **Polyglot Pioneers** team.

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.x-013243?style=for-the-badge&logo=numpy&logoColor=white)

</p>

<p align="center">

![Deep Learning](https://img.shields.io/badge/Deep%20Learning-LSTM-purple?style=for-the-badge)
![Explainable AI](https://img.shields.io/badge/XAI-SHAP-orange?style=for-the-badge)
![Uncertainty](https://img.shields.io/badge/Monte%20Carlo-Dropout-red?style=for-the-badge)
![Forecasting](https://img.shields.io/badge/Forecasting-Time%20Series-00599C?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Hackathon%20Project-success?style=for-the-badge)

</p>

<p align="center">

![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Maintained](https://img.shields.io/badge/Maintained-Yes-brightgreen?style=for-the-badge)

</p>

---

## 🚀 Highlights

- 🇧🇼 Botswana-focused Food Inflation Forecasting
- 📈 Multivariate Time-Series Forecasting
- 🤖 Classical Machine Learning Baseline
- 🧠 Deep Learning using LSTM Networks
- 📉 Monte Carlo Dropout for Predictive Uncertainty
- 🔍 Explainable AI using SHAP
- 🌍 Economic Intelligence Framework
- 📊 Decision-Support Focused

---

# 📖 Overview

Food inflation is influenced by complex interactions between global commodity prices, transportation costs, domestic monetary policy, and broader macroeconomic conditions. Reliable forecasting enables governments, policymakers, researchers, and businesses to make proactive decisions regarding food security, inflation management, and economic planning.

This project presents an **Economic Intelligence Framework** that integrates heterogeneous macroeconomic datasets into a unified forecasting pipeline capable of:

- Forecasting food inflation
- Comparing classical and deep learning approaches
- Quantifying predictive uncertainty
- Explaining model predictions
- Supporting evidence-based economic decision-making

---

# 🎯 Project Objectives

The project aims to:

- Integrate multiple macroeconomic datasets into a unified analytical dataset.
- Develop forecasting models for food inflation.
- Compare classical machine learning and deep learning performance.
- Quantify predictive uncertainty using Monte Carlo Dropout.
- Improve model transparency through SHAP Explainability.
- Build a reusable Economic Intelligence Framework for future research and policy analysis.

---

# 🏗️ System Architecture

```
                    External Economic Data
                              │
                              ▼
                    Data Integration Layer
                              │
                              ▼
                  Data Cleaning & Validation
                              │
                              ▼
                    Feature Engineering
                              │
                              ▼
                   Forecasting Models
                ┌─────────────────────────┐
                │                         │
                ▼                         ▼
     Classical Machine Learning      LSTM Network
                │                         │
                └──────────┬──────────────┘
                           ▼
               Monte Carlo Dropout
                           │
                           ▼
                SHAP Explainability
                           │
                           ▼
              Economic Intelligence Outputs
```

---

# 📂 Repository Structure

```text
.
├── Data_Raw/
│   ├── 01_baltic_dry_index_daily.csv
│   ├── 02_brent_crude_monthly.csv
│   ├── 03_botswana_policy_rate.csv
│   ├── 04_fao_botswana_prices.csv
│   └── 05_human_capital_project.csv
│
├── Data_Clean/
│   └── prediction_data.csv
│
├── notebooks/
│
├── src/
│   ├── data/
│   ├── models/
│   ├── explainability/
│   └── utils/
│
├── models/
│   ├── checkpoints/
│   └── explainability/
│
├── predictions/
│
├── logs/
│
├── README.md
└── requirements.txt
```

---

# 📊 Datasets

The framework integrates multiple macroeconomic datasets to capture factors influencing food inflation.

| Dataset | Description |
|----------|-------------|
| FAO Food Price Index | Food commodity prices |
| Baltic Dry Index | Global shipping costs |
| Brent Crude Oil | International energy prices |
| Botswana Policy Rate | Monetary policy indicator |
| Human Capital Project | Socioeconomic reference dataset |

---

# 🎯 Prediction Target

The forecasting task predicts:

```text
FAO_23014
```

using the following predictor variables:

- Brent_USD_per_barrel
- policy_rate
- BDI_std
- monthly_return_bdi

---

# 🤖 Forecasting Models

## Classical Statistical Model

A classical forecasting model (SARIMAX) serves as the benchmark for comparison with the deep learning approach.

The objective is to evaluate whether deep learning provides additional predictive value over more traditional statistical learning methods.

---

## Deep Learning (LSTM)

The deep learning component employs a **Long Short-Term Memory (LSTM)** neural network designed for multivariate time-series forecasting.

### Model Architecture

| Parameter | Value |
|-----------|------:|
| Input Features | 6 |
| Sequence Length | 12 Months |
| Hidden Units | 64 |
| LSTM Layers | 2 |
| Dropout | 0.20 |

---

# 📉 Uncertainty Quantification

Rather than producing only point forecasts, the framework estimates predictive uncertainty using **Monte Carlo Dropout**.

Generated outputs include:

- Predictive Mean
- Predictive Standard Deviation
- 95% Confidence Intervals
- 95% Prediction Intervals

This provides additional confidence information for economic decision-makers.

---

# 🔍 Explainable AI

Model transparency is achieved using **SHAP (SHapley Additive exPlanations)**.

The explainability module provides:

- Global feature importance
- Feature contribution analysis
- SHAP summary plots
- Improved model interpretability

---

# ⚙️ Training Pipeline

The forecasting workflow consists of:

- Data Integration
- Data Validation
- Feature Engineering
- Feature Scaling
- Chronological Data Splitting
- Sequence Generation
- LSTM Training
- Checkpoint Saving
- Early Stopping
- Learning Rate Scheduling
- Monte Carlo Dropout
- SHAP Explainability
- Forecast Generation

---

# 🛠️ Built With

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| PyTorch | Deep learning |
| Scikit-learn | Classical machine learning & preprocessing |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Matplotlib | Visualisation |
| SHAP | Explainable AI |

---

# 📦 Installation

Clone the repository.

```bash
git clone https://github.com/<organisation>/<repository>.git

cd <repository>
```

Create a virtual environment.

```bash
python -m venv .venv
```

Activate it.

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Train the forecasting model.

```bash
python src/models/trainer.py
```

Generate predictions.

```bash
python src/models/predict.py
```

Run SHAP explainability.

```bash
python src/explainability/shap_analysis.py
```

---

# 📊 Generated Outputs

Running the pipeline produces outputs similar to:

```text
models/
    checkpoints/

predictions/
    lstm_predictions.csv

models/explainability/
    shap_summary.png

logs/
    forecasting_pipeline.log
```

---

# Classical/Statistical Model
For the classical/statistical model, we use SARIMAX. The code is in the "Model_1_Classical" jupyter notebook. Before running this notebook, run the "Data Cleaning" notebook to clean and merge the relevant data. Replace the file paths with your own to run in your local computer. The human capital project data is also cleaned in the "Data Cleaning" notebook. However, it is not merged to the 4 main datasets and it is analyzed separately with external data.

After cleaning the data, save the merged data as "predictions_data" to the Data_Clean folder. Then load this data in the "Model_1_Classical" notebook. Remember to replace the file paths with your own to run on your local computer. The steps for SARIMAX are clearly laid out in the notebook, from data exploration to diagnostic tests and forecasts. 

We find that the classical model outperforms the deep learning model. Therefore, after creating the final forecasts, save the forecasted food price inflation data as "best_model_predictions.csv" to the Predictions folder.

# 📈 Future Improvements

Potential future extensions include:

- Transformer-based forecasting
- Temporal Convolutional Networks (TCNs)
- GRU architectures
- Additional macroeconomic indicators
- Interactive dashboards
- Policy scenario simulation
- Real-time forecasting
- Automated data ingestion
- Web-based decision-support platform

---

# 👥 Team

## Polyglot Pioneers

**Deep Learning IndabaX Botswana 2026 Hackathon**

| Role | Contributor |
|------|-------------|
| Classical Machine Learning | *To be updated* |
| Deep Learning | *To be updated* |
| Data Engineering | *To be updated* |
| Explainable AI | *To be updated* |
| Documentation | *To be updated* |

---

# 🤝 Contributing

This repository was developed as part of the **Deep Learning IndabaX Botswana 2026 Hackathon**.

Contributions, improvements, and suggestions are welcome.

If you would like to contribute:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push the branch
5. Open a Pull Request

---

# 📄 License

This project is released under the **MIT License**.

See the `LICENSE` file for details.

---

# 🙏 Acknowledgements

This project would not have been possible without the datasets and support provided by:

- Deep Learning IndabaX Botswana
- Food and Agriculture Organization (FAO)
- Bank of Botswana
- Baltic Exchange
- World Bank Human Capital Project
- The open-source Python community

---

# 📬 Contact

For questions, suggestions, or collaboration, please contact the **Polyglot Pioneers** team through GitHub.

---

<p align="center">
Built with ❤️ by <strong>Polyglot Pioneers</strong> for the <strong>Deep Learning IndabaX Botswana 2026 Hackathon</strong>.
</p>
