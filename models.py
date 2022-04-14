import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


# function to classify income levels 
def income_classifier(df):
    lower = np.percentile(df['ATINC'], 25)
    lower_middle = np.percentile(df['ATINC'], 50)
    middle = np.percentile(df['ATINC'], 75)
    upper_middle = np.percentile(df['ATINC'], 95)

    if df['ATINC'] <= lower: return "Low"
    elif (df['ATINC'] > lower) & (df['ATINC'] <= lower_middle): return "Lower-middle"
    elif (df['ATINC'] > lower_middle) & (df['ATINC'] <= middle): return "Middle"
    elif (df['ATINC'] > middle) & (df['ATINC'] <= upper_middle): return "Upper-middle"
    elif (df['ATINC'] > upper_middle): return "Upper"
# Reference: https://www.ictsd.org/what-income-class-are-you-canada/


# feature importance
def feature_imp(model, X_test, y_test):
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=0, n_jobs=-1
    )
    features = X_test.columns
    forest_importances = pd.Series(result.importances_mean, index=features)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()