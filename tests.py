import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns

ALPHA = 0.05

def chi_test(data, column1, column2):
    contingency = pd.crosstab(data[column1], data[column2])
    print(contingency)
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f'Chi-Squared Test p-value for {column1} and {column2}: {p}')


# 3. There is a difference in mean incomes of those who are landed immigrants vs those who are not
def imm_income(data):
    # seperate the data into 2 sets, one for immigrants and another for non-immigrants
    imm_data = data[data['IMMST'] == 'Immigrant']
    imm_data = imm_data['ATINC']
    non_data = data[data['IMMST'] == 'Non-Immigrant']
    non_data = non_data['ATINC']
    pval = stats.levene(imm_data, non_data).pvalue
    equal_var = pval < ALPHA
    ttest = stats.ttest_ind(imm_data, non_data, equal_var=equal_var)
    print('T-test p-value:', ttest.pvalue)


# ANOVA on incomes 
def anova(data):
    data = data[['MARSTP', 'ATINC']]
    # seperate by maritial status
    married = data[data['MARSTP'] == 'Married']
    married = married['ATINC']
    common_law = data[data['MARSTP'] == 'Common-Law']
    common_law = common_law['ATINC']
    seperated = data[data['MARSTP'] == 'Separated']
    seperated = seperated['ATINC']
    never_married = data[data['MARSTP'] == 'Single']
    never_married = never_married['ATINC']

    # check if the datasets have equal variance
    pval = stats.levene(married, common_law, seperated, never_married).pvalue
    if pval < ALPHA:
        anova = stats.f_oneway(married, common_law, seperated, never_married)
        print('ANOVA p-value:', anova.pvalue)
        # prepare for post hoc analysis
        married = pd.DataFrame(married)
        married[1] = 'married'
        common_law = pd.DataFrame(common_law)
        common_law[1] = 'common_law'
        seperated = pd.DataFrame(seperated)
        seperated[1] = 'seperated'
        never_married = pd.DataFrame(never_married)
        never_married[1] = 'never_married'
        ph_data = pd.concat((married, common_law, seperated, never_married), axis=0)
        ph_data.columns = ('value', 'variable')
        post_hoc(ph_data)
    else:
        print(pval)


def post_hoc(data):
    posthoc = pairwise_tukeyhsd(
        data['value'].astype(float), data['variable'],
        alpha = ALPHA
    )
    print(posthoc)


# Test whether a person's income is affected by their gender.
def income_gender(dflog):
    # splitting data
    mf_income = dflog[['SEX','ATINC']] 
    male_income = mf_income.loc[mf_income['SEX'] == 'Male'] # male after tax income
    female_income = mf_income.loc[mf_income['SEX']== 'Female'] # female after tax income

    pval = stats.levene(male_income['ATINC'], female_income['ATINC']).pvalue
    equal_var = pval < ALPHA
    ttest = stats.ttest_ind(male_income['ATINC'], female_income['ATINC'],equal_var=equal_var)
    print('T-Test p-value:', ttest.pvalue)
   
    sns.histplot(data=mf_income, x='ATINC', hue="SEX")
 


# Test to determine peoples major source of income difference 
def major_source(dflog):
    # splitting data
    maj_income = dflog[['MAJRI','ATINC']] 
    inc2 = maj_income.loc[maj_income['MAJRI'] == 'Wages and Salary'] 
    inc3 = maj_income.loc[maj_income['MAJRI'] == 'Self-Employment']  
    inc4 = maj_income.loc[maj_income['MAJRI'] == 'Government Transfers'] 
    inc5 = maj_income.loc[maj_income['MAJRI'] == 'Investment'] 
    inc6 = maj_income.loc[maj_income['MAJRI'] == 'Private Retirement Pensions']
    inc7 = maj_income.loc[maj_income['MAJRI'] == 'Other']  

    # ANOVA test to determine if the means of any of the groups differ
    anova = stats.f_oneway(inc2['ATINC'], inc3['ATINC'], inc4['ATINC'], inc5['ATINC'], inc6['ATINC'], inc7['ATINC'])
    print("\nANOVA p-value:",anova.pvalue,"\n") 

    # post hoc Tukey Test
    posthoc = pairwise_tukeyhsd(maj_income['ATINC'], maj_income['MAJRI'], alpha=0.05)
    print(posthoc)



# Adapted from: https://seaborn.pydata.org/examples/jitter_stripplot.html
def stripplot(melt_data, num_feature, cat_feature):
    num_vars = melt_data[cat_feature].nunique()
    sns.set_theme(style="whitegrid")

    # "Melt" the dataset to "long-form" or "tidy" representation
    melt_data = pd.melt(melt_data, cat_feature, var_name=num_feature)

    # Initialize the figure
    f, ax = plt.subplots()
    sns.despine(bottom=True, left=True)

    # Show each observation with a scatterplot
    sns.stripplot(x="value", y=num_feature, hue=cat_feature,
                data=melt_data, dodge=True, alpha=.25, zorder=1)

    # Show the conditional means, aligning each pointplot in the
    # center of the strips by adjusting the width allotted to each
    # category (.8 by default) by the number of hue levels
    sns.pointplot(x="value", y=num_feature, hue=cat_feature,
                data=melt_data, dodge=.8 - .8 / num_vars,
                join=False, palette="dark",
                markers="d", scale=.75, ci=None)

    # Improve the legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[num_vars:], labels[num_vars:], title=cat_feature,
            handletextpad=0, columnspacing=1,
            loc="lower right", ncol=num_vars, frameon=False)
