# reinforcement-learning-contextual-bandits

# Multi-Armed Bandit Algorithms for Personalized Recommendations and Warfarin Dosing

This project demonstrates the application of **multi-armed bandit algorithms** in two real-world scenarios:
1. **Personalized content recommendations** for users on platforms like YouTube.
2. **Optimal warfarin dose prediction** based on patient data to improve medical outcomes.

The project highlights various reinforcement learning techniques for decision-making under uncertainty, including:
- Upper Confidence Bound (UCB)
- Thompson Sampling
- ε-Greedy

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

## Algorithms Implemented

- **Fixed-Dose Baseline**: Assigns a static dose or recommendation.
- **Clinical Linear Model**: Incorporates patient features to predict outcomes.
- **Disjoint Linear UCB (LinUCB)**: Balances exploration and exploitation by estimating upper confidence bounds.
- **Thompson Sampling**: Utilizes Bayesian inference to dynamically update predictions.
- **ε-Greedy**: Randomly explores new options while mostly exploiting known best choices.

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
