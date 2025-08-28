# 🚦 Smart City Hybrid ML: Traffic & Pollution Optimizer  

> **Hybrid Machine Learning project** combining **Supervised Learning**, **Unsupervised Learning**, and **Reinforcement Learning** to analyze and optimize urban traffic congestion and air pollution.  

---

## 📌 Problem Statement  
Traffic congestion and rising air pollution are two of the most critical challenges in modern cities.  
Most solutions handle either **prediction** (supervised learning) or **pattern discovery** (unsupervised learning), but rarely integrate them with **adaptive control** (reinforcement learning).  

This project demonstrates a **hybrid ML pipeline** that:  
1. **Predicts** traffic congestion & AQI (Supervised).  
2. **Discovers** hidden traffic/pollution hotspots (Unsupervised).  
3. **Optimizes** traffic signal policies in a toy environment (Reinforcement Learning).  

---

## 🛠️ Approach  

### 🔹 Supervised Learning
- Predict Air Quality Index (AQI) or traffic congestion level.  
- Models: Random Forest, XGBoost.  
- Evaluation: MAE, RMSE, R².  

### 🔹 Unsupervised Learning
- Apply **KMeans clustering** to identify high-risk traffic/pollution zones.  
- Visualize hotspots via scatter plots & heatmaps.  

### 🔹 Reinforcement Learning (Lite Simulation)
- Custom toy environment simulating traffic signals.  
- **Q-Learning Agent** minimizes average vehicle wait time + idle pollution.  
- Reward curve demonstrates learning progress.  

---

## 📊 Results (Sample Outputs)  

| Module | Output |
|--------|--------|
| **Supervised** | AQI prediction plot (Predicted vs Actual) |
| **Unsupervised** | Clustered traffic/pollution hotspots |
| **Reinforcement** | RL reward curve showing improvement |

📌 Figures are stored in [`/results`](results/) and embedded below:  

![Supervised Results](results/supervised_results.png)  
![Cluster Map](results/clustering_map.png)  
![RL Rewards](results/rl_rewards_curve.png)  

---

## 🚀 Getting Started  

### 1. Clone Repository
```bash
git clone https://github.com/<your-username>/smart-city-hybrid-ml.git
cd smart-city-hybrid-ml
