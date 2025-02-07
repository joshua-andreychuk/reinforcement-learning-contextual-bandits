import numpy as np
import csv
import os

from collections import Counter

from abc import ABC, abstractmethod
from utils.data_preprocessing import load_data, dose_class, LABEL_KEY


# Base classes
class BanditPolicy(ABC):
    @abstractmethod
    def extract_features(self, x, features):
        pass

    @abstractmethod
    def choose(self, x):
        pass

    @abstractmethod
    def update(self, x, a, r):
        pass


class StaticPolicy(BanditPolicy):
    def extract_features(self, x, features):
        pass

    def update(self, x, a, r):
        pass


class RandomPolicy(StaticPolicy):
    def __init__(self, probs=None):
        self.probs = probs if probs is not None else [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

    def choose(self, x):
        return np.random.choice(range(len(self.probs)), p=self.probs)


############################################################
# Problem 1: Estimation of Warfarin Dose
############################################################

############################################################
# Problem 1a: baselines


class FixedDosePolicy(StaticPolicy):
    def choose(self, x):
        """
        Args:
                x: numpy array of features
        Returns:
                output: index of the chosen action

        TODO:
                - Please implement the fixed dose which is to assign medium dose
                  to all patients.
        """
        ### START CODE HERE ###
        # Always return the index for the medium dose (index 1 in this case).
        return 1
        ### END CODE HERE ###


class ClinicalDosingPolicy(StaticPolicy):

    def extract_features(self, x):
        """
        Args:
                x (dict): dictionary containing the possible patient features.

        Returns:
                x (float): containing the square root of the weekly warfarin dose

        TODO:
                - Prepare the features to be used in the clinical model
                  (consult section 1f of appx.pdf for feature definitions)

        Hint:
                - Look at the utils/data_preprocessing.py script to see the key values
                  of the features you can use. The age in decades is implemented for
                  you as an example.
                - You can treat Unknown race as missing or mixed race.

        """
        weekly_dose_sqrt = None

        age_in_decades = x["Age in decades"]

        ### START CODE HERE ###
        # Initialize the coefficients as per the clinical model from section 1f
        intercept = 4.0376
        age_coeff = -0.2546
        height_coeff = 0.0118
        weight_coeff = 0.0134
        asian_coeff = -0.6752
        black_coeff = 0.4060
        mixed_coeff = 0.0443
        enzyme_inducer_coeff = 1.2799
        amiodarone_coeff = -0.5695

        # Retrieve fields with defaults for optional features
        height_cm = x["Height (cm)"]
        weight_kg = x["Weight (kg)"]
        race = x.get("Race", "Missing or Mixed race")  # Default to "Missing or Mixed race"
        enzyme_inducer_status = x.get("Enzyme inducer status", 0)  # Default to 0 if missing
        amiodarone_status = x.get("Amiodarone", 0)  # Default to 0 if missing

        # Compute race-specific coefficient
        race_coeff = 0
        if race == "Asian":
            race_coeff = asian_coeff
        elif race == "Black or African American":
            race_coeff = black_coeff
        elif race == "Missing or Mixed race":
            race_coeff = mixed_coeff

        # Compute the square root of the weekly dose
        weekly_dose_sqrt = (intercept +
                            age_coeff * age_in_decades +
                            height_coeff * height_cm +
                            weight_coeff * weight_kg +
                            race_coeff +
                            enzyme_inducer_coeff * enzyme_inducer_status +
                            amiodarone_coeff * amiodarone_status)
        ### END CODE HERE ###

        return weekly_dose_sqrt

    def choose(self, x):
        """
        Args:
                x (dict): dictionary containing the possible patient features.
        Returns:
                output: index of the chosen action

        TODO:
                - Create a linear model based on the values in section 1f
                  and return its output based on the input features

        Hint:
                - Use dose_class() implemented for you.
        """

        weekly_dose_sqrt = self.extract_features(x)
        ### START CODE HERE ###
        # Convert the predicted square root of the dose into one of the three dose classes
        dose = weekly_dose_sqrt ** 2  # Convert back to weekly dose
        return dose_class(dose)
        ### END CODE HERE ###


############################################################
# Problem 1b: upper confidence bound linear bandit


class LinUCB(BanditPolicy):
    def __init__(self, num_arms, features, alpha=1.0):
        """
        See Algorithm 1 from paper:
                "A Contextual-Bandit Approach to Personalized News Article Recommendation"

        Args:
                num_arms (int): the initial number of different arms / actions the algorithm can take
                features (list of str): contains the features to use
                alpha (float): hyperparameter for step size.

        TODO:
                - Please initialize the following internal variables for the Disjoint Linear UCB Bandit algorithm:
                        * self.features
                        * self.d
                        * self.alpha
                        * self.A
                        * self.b
                  These terms align with the paper, please refer to the paper to understand what they are.
                  Feel free to add additional internal variables if you need them, but they are not necessary.

        Hint:
                Keep track of a seperate A, b for each action (this is what the Disjoint in the algorithm name means)
        """
        ### START CODE HERE ###
        self.features = features
        self.d = len(features)
        self.alpha = alpha
        # Store per‐arm parameters in lists (indexed by arm ID).
        self.A = [np.eye(self.d) for _ in range(num_arms)]
        self.b = [np.zeros(self.d) for _ in range(num_arms)]
        ### END CODE HERE ###

    def extract_features(self, x):
        """
        Args:
                x (dict): dictionary containing the possible features.

        Returns:
                out: numpy array of features
        """

        return np.array([x[f] for f in self.features])

    def choose(self, x):
        """
        See Algorithm 1 from paper:
                "A Contextual-Bandit Approach to Personalized News Article Recommendation"

        Args:
                x:
                 - (dict): dictionary containing the possible features.
                 or
                 - (numpy array): array of features
        Returns:
                output: index of the chosen action

        Please implement the "forward pass" for Disjoint Linear Upper Confidence Bound Bandit algorithm.
        """

        xvec = x
        if isinstance(x, dict):
            xvec = self.extract_features(x)

        ### START CODE HERE ###
        max_pta = -np.inf
        best_action = 0
        # Iterate over all arms in our list
        for a in range(len(self.A)):
            A_inv = np.linalg.inv(self.A[a])
            theta_a = A_inv @ self.b[a]
            pta = theta_a.dot(xvec) + self.alpha * np.sqrt(xvec @ A_inv @ xvec)
            if pta > max_pta:
                max_pta = pta
                best_action = a

        return best_action
        ### END CODE HERE ###

    def update(self, x, a, r):
        """
        Args:
                x:
                 - (dict): dictionary containing the possible features.
                 or
                 - (numpy array): array of features
                a: integer, indicating the action your algorithm chose
                r: the reward you received for that action
        Returns:
                Nothing

        Please implement the update step for Disjoint Linear Upper Confidence Bound Bandit algorithm.
        """

        xvec = x
        if isinstance(x, dict):
            xvec = self.extract_features(x)

        ### START CODE HERE ###
        self.A[a] += np.outer(xvec, xvec)
        self.b[a] += r * xvec
        ### END CODE HERE ###


############################################################
# Problem 1c: eGreedy linear bandit


class eGreedyLinB(LinUCB):
    def __init__(self, num_arms, features, alpha=1.0):
        super(eGreedyLinB, self).__init__(num_arms, features, alpha)
        self.time = 0

    def choose(self, x):
        """
        Args:
                x (dict): dictionary containing the possible features.
        Returns:
                output: index of the chosen action

        TODO:
                - Instead of using the Upper Confidence Bound to find which action to take,
                  compute the payoff of each action using a simple dot product between Theta & the input features.
                  Then use an epsilon greedy algorithm to choose the action.
                  Use the value of epsilon provided and np.random.uniform() in your implementation.
        """

        self.time += 1
        epsilon = float(1.0 / self.time) * self.alpha
        xvec = self.extract_features(x)

        ### START CODE HERE ###
        # Compute the estimated payoff for each arm
        payoffs = []
        for a in range(len(self.A)):
            A_inv = np.linalg.inv(self.A[a])
            theta_a = A_inv @ self.b[a]
            payoffs.append(theta_a.dot(xvec))

        # Epsilon‐greedy choice
        if np.random.uniform() < epsilon:
            chosen_action = np.random.randint(len(self.A))  # random arm
        else:
            chosen_action = np.argmax(payoffs)

        return chosen_action
        ### END CODE HERE ###


############################################################
# Problem 1d: Thompson sampling


class ThomSampB(BanditPolicy):
    def __init__(self, num_arms, features, alpha=1.0):
        """
        See Algorithm 1 and section 2.2 from paper:
                "Thompson Sampling for Contextual Bandits with Linear Payoffs"

        Args:
                num_arms (int): the initial number of different arms / actions the algorithm can take
                features (list of str): contains the features to use
                alpha (float): hyperparameter for step size.

        TODO:
                - Please initialize the following internal variables for the Thompson sampling bandit algorithm:
                        * self.features
                        * self.num_arms
                        * self.d
                        * self.v2 (please set this term equal to alpha)
                        * self.B
                        * self.mu
                        * self.f
                These terms align with the paper, please refer to the paper to understand what they are.
                Please feel free to add additional internal variables if you need them, but they are not necessary.

        Hints:
                - Keep track of a separate B, mu, f for each arm (this is what the Disjoint in the algorithm name means)
                - Unlike in section 2.2 in the paper where they sample a single mu_tilde, we'll sample a mu_tilde for each arm
                  based on our saved B, f, and mu values for each arm. Also, when we update, we only update the B, f, and mu
                  values for the arm that we selected
                - What the paper refers to as b in our case is the medical features vector
                - The paper uses a summation (from time =0, .., t-1) to compute the model parameters at time step (t),
                  however if you can't access prior data how might one store the result from the prior time steps.

        """
        ### START CODE HERE ###
        self.features = features
        self.num_arms = num_arms
        self.d = len(features)
        self.v2 = alpha  # variance scaling

        # Store each arm's parameters in lists
        self.B = [np.eye(self.d) for _ in range(num_arms)]
        self.mu = [np.zeros(self.d) for _ in range(num_arms)]
        self.f = [np.zeros(self.d) for _ in range(num_arms)]
        ### END CODE HERE ###

    def extract_features(self, x):
        """
        Args:
                x (dict): dictionary containing the possible features.

        Returns:
                out: numpy array of features
        """

        return np.array([x[f] for f in self.features])

    def choose(self, x):
        """
        See Algorithm 1 and section 2.2 from paper:
                "Thompson Sampling for Contextual Bandits with Linear Payoffs"

        Args:
                x (dict): dictionary containing the possible features.
        Returns:
                output: index of the chosen action

        TODO:
                - Please implement the "forward pass" for Disjoint Thompson Sampling Bandit algorithm.
                - Please use np.random.multivariate_normal to simulate the multivariate gaussian distribution in the paper.
        """

        xvec = self.extract_features(x)

        ### START CODE HERE ###
        max_score = -np.inf
        best_action = 0
        for a in range(len(self.B)):
            B_inv = np.linalg.inv(self.B[a])
            # Sample from N(mu[a], v2 * B_inv)
            mu_tilde = np.random.multivariate_normal(self.mu[a], self.v2 * B_inv)
            score = mu_tilde.dot(xvec)
            if score > max_score:
                max_score = score
                best_action = a
        return best_action
        ### END CODE HERE ###

    def update(self, x, a, r):
        """
        See Algorithm 1 and section 2.2 from paper:
                "Thompson Sampling for Contextual Bandits with Linear Payoffs"

        Args:
                x (dict): dictionary containing the possible features.
                a: integer, indicating the action your algorithm chose
                r (int): the reward you received for that action

        TODO:
                - Please implement the update step for Disjoint Thompson Sampling Bandit algorithm.

        Hint:
                Which parameters should you update?
        """

        xvec = self.extract_features(x)

        ### START CODE HERE ###
        self.B[a] += np.outer(xvec, xvec)
        self.f[a] += r * xvec
        self.mu[a] = np.linalg.inv(self.B[a]) @ self.f[a]
        ### END CODE HERE ###


############################################################
# Problem 2: Recommender system simulator
############################################################

############################################################
# Problem 2a: LinUCB with increasing number of arms
############################################################


class DynamicLinUCB(LinUCB):

    def add_arm_params(self):
        """
        Add a new A and b for the new arm we added.
        Initialize them in the same way you did in the __init__ method
        """
        ### START CODE HERE ###
        # Instead of storing an arm under key -1 in a dictionary, just append.
        self.A.append(np.eye(self.d))
        self.b.append(np.zeros(self.d))
        ### END CODE HERE ###


class Simulator:
    """
    Simulates a recommender system setup where we have say A arms corresponding to items and U users initially.
    The number of users U cannot change but the number of arms A can increase over time
    """
    def __init__(
        self,
        num_users=10,
        num_arms=5,
        num_features=10,
        update_freq=20,
        update_arms_strategy="none",
    ):
        self.num_users = num_users
        self.num_arms = num_arms
        self.num_features = num_features
        self.update_freq = update_freq
        self.update_arms_strategy = update_arms_strategy
        self.arms = {}  ## arm_id: np.array
        self.users = {}  ## user_id: np.array
        self._init(means=np.arange(-5, 6), scale=1.0)
        self.steps = 0  ## number of steps since last arm update
        self.logs = []  ## each element is of the form [user_id, arm_id, best_arm_id]

    def _init(self, means, scale):
        for i in range(self.num_users):
            v = []
            for _ in range(self.num_features):
                v.append(np.random.normal(loc=np.random.choice(means), scale=scale))
            self.users[i] = np.array(v).reshape(-1)
        for i in range(self.num_arms):
            v = []
            for _ in range(self.num_features):
                v.append(np.random.normal(loc=np.random.choice(means), scale=scale))
            self.arms[i] = np.array(v).reshape(-1)

    def reset(self):
        user_ids = list(self.users.keys())
        user = np.random.choice(user_ids)
        return user, self.users[user]

    def get_reward(self, user_id, arm_id):
        user_context = self.users[user_id]
        best_arm_id, best_score = None, None
        for a_id, arm in self.arms.items():
            score = arm.dot(user_context)
            if not best_arm_id:
                best_arm_id = a_id
                best_score = score
                continue
            if best_score < score:
                best_arm_id = a_id
                best_score = score
        self.logs.append([user_id, arm_id, best_arm_id])
        if arm_id == best_arm_id:
            return 0
        else:
            return -1

    def update_arms(self):
        if self.update_arms_strategy == "none":
            return False
        if self.update_arms_strategy == "popular":
            from collections import Counter
            #######################################################
            #########  ~8 lines.   #############
            ### START CODE HERE ###
            counts = Counter([log[1] for log in self.logs])
            if len(counts) < 2:
                return False

            most_common = counts.most_common(2)
            arm1, arm2 = most_common[0][0], most_common[1][0]

            new_theta = (self.arms[arm1] + self.arms[arm2]) / 2
            new_arm_id = len(self.arms)
            self.arms[new_arm_id] = new_theta
            self.num_arms += 1
            return True
            ### END CODE HERE ###
        if self.update_arms_strategy == "corrective":
            from collections import Counter
            #######################################################
            #########  ~7 lines.   #############
            ### START CODE HERE ###
            incorrect_logs = [log for log in self.logs if log[1] != log[2]]
            if not incorrect_logs:
                return False

            weights = Counter([log[2] for log in incorrect_logs])
            total_weight = sum(weights.values())

            new_theta = sum(weights[arm_id] * self.arms[arm_id] for arm_id in weights) / total_weight
            new_arm_id = len(self.arms)
            self.arms[new_arm_id] = new_theta
            self.num_arms += 1
            return True
            ### END CODE HERE ###
        if self.update_arms_strategy == "counterfactual":
            #######################################################
            #########  ~9 lines.   #############
            ### START CODE HERE ###
            theta_new = np.zeros(self.num_features)
            eta = 0.1

            grad_sum = np.zeros(self.num_features)
            for (user_id, chosen_arm_id, best_arm_id) in self.logs:
                x_i = self.users[user_id]
                chosen_theta = self.arms[chosen_arm_id]
                diff = (theta_new @ x_i) - (chosen_theta @ x_i)
                grad_sum += diff * x_i

            theta_new += eta * grad_sum

            new_arm_id = len(self.arms)
            self.arms[new_arm_id] = theta_new
            self.num_arms += 1
            return True
            ### END CODE HERE ###
        return True

    def step(self, user_id, arm_id):
        self.steps += 1
        reward = self.get_reward(user_id, arm_id)
        arm_added = False
        if self.steps % self.update_freq == 0:
            arm_added = self.update_arms()
            self.logs = []
            self.steps = 0
        user_ids = list(self.users.keys())
        user = np.random.choice(user_ids)
        return user, self.users[user], reward, arm_added
