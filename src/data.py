from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from config import RANDOM_STATE

def load_data():
    """
    Load the California Housing dataset and return the features and target values.
    The dataset is downloaded on first use and cached in ~/scikit_learn_data.
    Returns:
        X (numpy.ndarray): The 8 features of each census block group.
        y (numpy.ndarray): Median house value in units of $100,000.
    """
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    return X, y

def split_data(X, y):
    """
    Split the data into training and testing sets.
    Args:
        X (numpy.ndarray): The features of the dataset.
        y (numpy.ndarray): The target values of the dataset.
    Returns:
        X_train, X_test, y_train, y_test (tuple): The split dataset.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    return X_train, X_test, y_train, y_test
