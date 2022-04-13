import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


# function to classify income levels 
def income_classifier(df):
    if df['ATINC'] <= 32048: return "Low"
    elif (df['ATINC'] > 32048) & (df['ATINC'] <= 53413): return "Lower-middle"
    elif (df['ATINC'] > 53413) & (df['ATINC'] <= 106827): return "Middle"
    elif (df['ATINC'] > 106827) & (df['ATINC'] <= 373894): return "Upper-middle"
    elif (df['ATINC'] > 373894): return "Upper"
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