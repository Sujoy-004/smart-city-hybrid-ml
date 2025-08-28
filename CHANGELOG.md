# Changelog

All notable changes to this project will be documented in this file.  
This project follows a progressive roadmap: starting with a demo-quality hybrid ML system (2 days) and scaling into a research-quality smart city optimizer.

---

## [v0.1.0] - 2025-08-28
### Added
- **Supervised Learning Module**  
  - Implemented AQI/traffic prediction using Random Forest and XGBoost.  
  - Cleaned dataset and engineered basic features (time, weather, traffic density).  
  - Added regression metrics (MAE, RMSE, R²).  

- **Unsupervised Learning Module**  
  - Applied KMeans clustering to detect traffic/pollution hotspots.  
  - Visualized clusters using scatter plots and heatmaps.  
  - Basic interpretation of zone-wise risk levels.  

- **Reinforcement Learning Simulation (Lite)**  
  - Built a toy RL environment for traffic signal control using Q-Learning.  
  - Defined reward function to minimize wait time + reduce idle pollution.  
  - Plotted reward vs episodes graph to show learning curve.  

- **Integration Notebook**  
  - Unified supervised, unsupervised, and RL modules in a single Colab notebook.  
  - Added structured sections: *Prediction → Clustering → RL Simulation*.  

### Documentation
- Created `README.md` describing the hybrid ML concept.  
- Added this `CHANGELOG.md`.  

---

## [v0.2.0] - Planned
### Enhancements
- Replace toy RL with **DQN-based signal optimization**.  
- Add map-based visualization (e.g., Folium/Plotly) for AQI hotspots.  
- Streamlit dashboard integration for interactive demo.  

---

## [v1.0.0] - Future Research Release
### Roadmap
- Integrate **real traffic simulation (SUMO)**.  
- Extend supervised learning with **LSTM/GNN** for time-series + network data.  
- Deploy full hybrid pipeline as a **Smart City SaaS prototype**.  
- Add **explainability (SHAP/LIME)** for supervised models.  

---
