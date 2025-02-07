# reinforcement-learning-contextual-bandits

# Multi-Armed Bandit Algorithms for Personalized Recommendations and Warfarin Dosing

This project demonstrates the application of **multi-armed bandit algorithms** in two real-world scenarios:
1. **Personalized content recommendations** for users on platforms like YouTube.
2. **Optimal warfarin dose prediction** based on patient data to improve medical outcomes.

---

# How to Use the Project

Everything you need is included in the `reinforcement_learning.zip` file:
- Fully functioning project
- Environment setup (`environment.yml`)
- Scripts and Jupyter notebooks
- Datasets and outputs

Simply download and extract the zip file to get started!

---

## Project Overview

### Objective
To showcase how contextual bandit algorithms can solve decision-making problems in dynamic environments with incomplete information.

---

## Real-World Scenarios

1. **Warfarin Dose Prediction**  
   Warfarin dosing is complex due to varying patient characteristics. This task involved:
   - Predicting optimal dosages using patient demographic and genetic data.
   - Implementing baseline and advanced models such as LinUCB and Thompson Sampling.
   - Evaluating models based on the accuracy of prescribed dosages.

2. **Video Recommendation System**  
   In this scenario, contextual bandit algorithms were applied to simulate personalized video recommendations:
   - Users have static or evolving preferences.
   - New "arms" (videos) were dynamically added to the system.
   - Multiple strategies for updating arms were tested, including:
     - Popularity-based
     - Corrective (error-driven)
     - Counterfactual optimization

---

### **Algorithms Implemented**

#### **Warfarin Dose Prediction**
- **Fixed-Dose Baseline**: Assigns a static dose to all patients.
- **Clinical Linear Model**: Uses patient features (age, weight, race, etc.) to predict warfarin dosage.
- **Disjoint Linear UCB (LinUCB)**: Balances exploration and exploitation by estimating upper confidence bounds for optimal dosing.
- **Thompson Sampling**: Uses Bayesian inference to dynamically update dosage predictions.
- **ε-Greedy**: Randomly explores new dose options while primarily exploiting known best choices.

#### **Recommendation System**
- **Disjoint Linear UCB (LinUCB-None)**: Adapts to user preferences and balances exploration and exploitation for better recommendations.
- **Popularity-Based Strategy**: Creates new arms (content/videos) based on the most popular recent arms.
- **Corrective Strategy**: Generates new arms to correct recommendation errors by learning from past user interactions.
- **Counterfactual Optimization Strategy**: Simulates new arms that could have yielded better outcomes in past recommendations.

---

## Key Insights

- In medical dosing, even small improvements in prediction accuracy can significantly reduce adverse effects.
- In recommendation systems, strategies like counterfactual optimization can dynamically adapt to changing user and content creator behaviors.
- Feedback loops in AI systems can reinforce user preferences, which may have ethical implications if not properly monitored.

---

## Tools and Technologies

- **Python**: Primary programming language.
- **Reinforcement Learning (RL)** frameworks.
- **Matplotlib**: Visualization of algorithm performance.
- **Jupyter Notebook**: Interactive exploration and documentation.

---

## About Me

I'm a data scientist with experience in applying machine learning and reinforcement learning to solve complex problems. This project showcases my skills in:
- **Algorithm development** for contextual bandit problems.
- **Data-driven decision-making** with a focus on healthcare and user personalization.
- **Communication of insights** through visualizations and reports.
