import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


# function to classify income levels 
def income_classifier(df):
    if (df['ATINC'] <= 35290.0): return "Low"
    elif (df['ATINC'] > 35290.0) & (df['ATINC'] <= 49230.0): return "Lower-middle"
    elif (df['ATINC'] > 49230.0) & (df['ATINC'] <= 69692.5): return "Middle"
    elif (df['ATINC'] > 69692.5) & (df['ATINC'] <= 114920.0): return "Upper-middle"
    elif (df['ATINC'] > 114920.0): return "Upper"

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