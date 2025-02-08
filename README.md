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

### **Performance Results**

#### **Recommendation System**

1. **Fraction Incorrect vs. Users Seen** (First Graph):  
   - The **red curve (Counterfactual)** achieves the lowest error rate, improving steadily as more users are seen.
   - The **blue curve (None)** steadily improves with users seen, eventually outperforming both **green (Corrective)** and **orange (Popular)**.
   - **Popular** and **Corrective** are **exactly identical**, with both curves overlapping entirely and converging to a higher error rate than **None** and **Counterfactual**.

2. **Total Fraction Correct vs. Number of Arms (`K`)** (Second Graph):  
   - The **red curve (Counterfactual)** maintains the highest correct fraction across all values of `K`.
   - The **blue curve (None)** steadily improves with more arms seen, maintaining better performance than **Popular** and **Corrective**.
   - **Popular** and **Corrective** are **exactly identical**, showing no variation in performance even as the number of arms increases.

---

### **What Does This Mean?**

1. **Counterfactual Optimization**:
   - This strategy is highly effective because it simulates alternative outcomes to guide learning. It helps the model continuously optimize based on what "could have been better," leading to significantly improved performance in dynamic environments with evolving user interactions.

2. **None Strategy**:
   - Surprisingly, the **None** strategy (no new arms added) performs better than both **Popular** and **Corrective**.  
   - This suggests that introducing new arms can create unnecessary exploration noise or instability, limiting the algorithm’s ability to refine its understanding of existing arms.

3. **Identical Performance of Popular and Corrective**:
   - The fact that **Popular** and **Corrective** are indistinguishable suggests that their implementations or the underlying data they rely on might be structured in a similar way. Neither approach introduces any significant advantage over the other in this scenario.

4. **Convergence at Higher K**:
   - As the number of arms increases, all strategies show signs of performance convergence. This implies that with more arms, the performance differences between strategies narrow, although **Counterfactual** continues to maintain an edge.

---

### **Performance Results**

#### **Warfarin Dose Prediction**

1. **Fixed Strategy**:
   - **Total Fraction Correct:** Consistently around 0.61  
   - **Performance Insight:** The **Fixed** strategy has deterministic behavior and stabilizes at the lowest correct fraction, indicating its lack of adaptability to new data or patient characteristics.

2. **Clinical Model**:
   - **Total Fraction Correct:** Consistently around 0.64  
   - **Performance Insight:** The **Clinical** model is also deterministic and produces the same result across multiple runs. It leverages static patient-specific features to achieve better performance than **Fixed**, but without exploration, its performance is capped.

3. **LinUCB**:
   - **Total Fraction Correct:** Ranges between 0.64 and 0.66 across runs  
   - **Performance Insight:** **LinUCB** is a stochastic algorithm with random exploration, resulting in slight variations in total fraction correct across different runs. It outperforms both **Fixed** and **Clinical** through dynamic learning and confidence-bound optimization.

4. **ε-Greedy**:
   - **Total Fraction Correct:** Ranges between 0.64 and 0.65 across runs  
   - **Performance Insight:** The **ε-Greedy** strategy introduces randomness to balance exploration and exploitation. It converges to similar performance as LinUCB, though it may be less efficient in exploration.

5. **Thompson Sampling**:
   - **Total Fraction Correct:** Ranges between 0.64 and 0.66 across runs  
   - **Performance Insight:** **Thompson Sampling** uses Bayesian inference, producing slightly different outcomes on each run due to random sampling. It performs comparably to LinUCB and ε-Greedy, demonstrating strong exploration capabilities.

---

### **What Does This Mean?**

1. **Deterministic vs. Stochastic Models**:  
   - **Fixed** and **Clinical** models produce consistent results across runs because they lack stochastic exploration. While **Clinical** performs better than **Fixed**, it is still limited in adaptability.
   - In contrast, **LinUCB**, **Thompson Sampling**, and **ε-Greedy** exhibit random variations but converge to similar high performance.

2. **Exploration Matters**:  
   - Stochastic models outperform deterministic ones because they continually explore and adapt to new data. This is especially important in dynamic environments like medical dosing, where patient characteristics vary widely.

3. **Fixed Model's Weakness**:  
   - The consistently low performance of the **Fixed** model highlights the importance of personalized dosing strategies in reducing incorrect outcomes.

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
