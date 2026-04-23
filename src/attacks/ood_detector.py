"""OOD detectors for LIME and SHAP perturbations.

Direct port of perturbation generation from Slack et al.'s adversarial_models.py,
adapted for our geospatial housing setting.
"""

import numpy as np
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_lime_ood_detector(X, perturbation_std=0.3, perturbation_multiplier=30,
                            rf_estimators=100, random_state=42):
    """Train RF to distinguish real data from LIME-style Gaussian perturbations.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    perturbation_std : float, std of Gaussian noise (LIME default is 0.3)
    perturbation_multiplier : int, how many copies of perturbed data to generate
    rf_estimators : int
    random_state : int

    Returns
    -------
    detector : RandomForestClassifier
    diagnostics : dict with accuracy, n_real, n_perturbed
    """
    np.random.seed(random_state)

    all_x, all_y = [], []
    for _ in range(perturbation_multiplier):
        perturbed = X + np.random.normal(0, perturbation_std, size=X.shape)
        combined_x = np.vstack((X, perturbed))
        combined_y = np.concatenate((np.ones(len(X)), np.zeros(len(X))))
        all_x.append(combined_x)
        all_y.append(combined_y)

    all_x = np.vstack(all_x)
    all_y = np.concatenate(all_y)

    xtrain, xtest, ytrain, ytest = train_test_split(
        all_x, all_y, test_size=0.2, random_state=random_state
    )

    detector = RandomForestClassifier(
        n_estimators=rf_estimators, random_state=random_state
    )
    detector.fit(xtrain, ytrain)

    ypred = detector.predict(xtest)
    accuracy = accuracy_score(ytest, ypred)

    diagnostics = {
        "accuracy": float(accuracy),
        "n_real": int(np.sum(all_y == 1)),
        "n_perturbed": int(np.sum(all_y == 0)),
    }

    return detector, diagnostics


def train_shap_ood_detector(X, n_kmeans=10, perturbation_multiplier=10,
                            n_samples=20000, rf_estimators=100,
                            random_state=42):
    """Train RF to distinguish real data from SHAP-style feature substitutions.

    Generates OOD samples by randomly substituting features from k-means
    background distribution, mimicking KernelSHAP's internal perturbation.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    n_kmeans : int, number of k-means clusters for background
    perturbation_multiplier : int, repetitions of real data
    n_samples : int, number of substituted samples to generate
    rf_estimators : int
    random_state : int

    Returns
    -------
    detector : RandomForestClassifier
    diagnostics : dict with accuracy, n_real, n_perturbed
    """
    import shap

    np.random.seed(random_state)

    # Background distribution from k-means
    background = shap.kmeans(X, n_kmeans).data

    # Repeat real data
    repeated_X = np.repeat(X, perturbation_multiplier, axis=0)

    # Generate substituted samples
    new_instances = []
    for _ in range(n_samples):
        i = np.random.choice(len(X))
        point = deepcopy(X[i, :])
        # Randomly substitute features from background
        for _ in range(X.shape[1]):
            j = np.random.choice(X.shape[1])
            point[j] = background[np.random.choice(len(background)), j]
        new_instances.append(point)

    substituted = np.vstack(new_instances)

    # Label: 1 if point exists in original data, 0 otherwise
    xlist = X.tolist()
    sub_labels = np.array([
        1 if substituted[i].tolist() in xlist else 0
        for i in range(len(substituted))
    ])

    all_x = np.vstack((repeated_X, substituted))
    all_y = np.concatenate((np.ones(len(repeated_X)), sub_labels))

    xtrain, xtest, ytrain, ytest = train_test_split(
        all_x, all_y, test_size=0.2, random_state=random_state
    )

    detector = RandomForestClassifier(
        n_estimators=rf_estimators, random_state=random_state
    )
    detector.fit(xtrain, ytrain)

    ypred = detector.predict(xtest)
    accuracy = accuracy_score(ytest, ypred)

    diagnostics = {
        "accuracy": float(accuracy),
        "n_real": int(np.sum(all_y == 1)),
        "n_perturbed": int(np.sum(all_y == 0)),
    }

    return detector, diagnostics
