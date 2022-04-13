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
    print(f'Chi squared contingency tests for {column1} and {column2}: {p}')


def t_test(data1, data2, equal_var):
    ttest = stats.ttest_ind(data1, data2, equal_var=equal_var)
    if ttest.pvalue < ALPHA:
        print(f'The means of {data1.__name__} and {data2.__name__} are different.')
    else:
        print(f'The means of {data1.__name__} and {data2.__name__} are not different.')


def plot_hist(data1, data2, num_bins):
    figure, (axis1, axis2) = plt.subplots(1, 2)
    axis1.hist(data1, bins=num_bins)
    axis1.set_title(data1.__name__)
    axis2.hist(data2, bins=num_bins)
    axis2.set_title(data2.__name__)
    plt.show()

# 3. There is a difference in mean incomes of those who are landed immigrants vs those who are not
def hyp_3(data):
    # seperate the data into 2 sets, one for immigrants and another for non-immigrants
    imm_data = data[data['IMMST'] == 1]
    imm_data = imm_data['ATINC']
    imm_data.__name__ = 'After-tax income of immigrants'
    non_data = data[data['IMMST'] == 2]
    non_data = non_data['ATINC']
    non_data.__name__ = 'After-tax income of non-immigrants'
    # plot histograms for each of the 2 datasets to see if it is normally distributed (enough)
    plot_hist(imm_data, non_data, 50)
    # it's not at all normally distributed, it's quite right skewed so transform it
    log_imm = np.log(imm_data[imm_data > 0])
    log_imm.__name__ = 'Transformed after-tax income of immigrants'
    log_non = np.log(non_data[non_data > 0])
    log_non.__name__ = 'Transformed after-tax income of non-immigrants'
    # check again if it is normally distributed (enough)
    plot_hist(log_imm, log_non, 50)
    # check if the datasets have equal variance
    pval = stats.levene(log_imm, log_non).pvalue
    if pval < ALPHA:
        t_test(log_imm, log_non, True)
    else:
        t_test(log_imm, log_non, False)
    # sanity check?
    print(imm_data.mean(), non_data.mean())


# ANOVA on incomes 
def anova(data):
    data = data[['MARSTP', 'ATINC']]
    # seperate by maritial status
    married = data[data['MARSTP'] == 1]
    married = married['ATINC']
    common_law = data[data['MARSTP'] == 2]
    common_law = common_law['ATINC']
    seperated = data[data['MARSTP'] == 3]
    seperated = seperated['ATINC']
    never_married = data[data['MARSTP'] == 4]
    never_married = never_married['ATINC']

    married.__name__ = 'married after-tax income' 
    common_law.__name__ = 'common-law after-tax income'
    seperated.__name__ = 'seperated after-tax income' 
    never_married.__name__ = 'never married after-tax income' 

    plot_hist(married, common_law, 50)
    plot_hist(seperated, never_married, 50)
    
    # it's not at all normally distributed, it's quite right skewed so transform it
    married = np.log(married[married > 0])
    common_law = np.log(common_law[common_law > 0])
    seperated = np.log(seperated[seperated > 0])
    never_married = np.log(never_married[never_married > 0])
    
    married.__name__ = 'Log transformed married after-tax income' 
    common_law.__name__ = 'Log transformed common-law after-tax income'
    seperated.__name__ = 'Log transformed seperated after-tax income' 
    never_married.__name__ = 'Log transformed never married after-tax income'

    plot_hist(married, common_law, 50)
    plot_hist(seperated, never_married, 50)

    # check if the datasets have equal variance
    pval = stats.levene(married, common_law, seperated, never_married).pvalue
    if pval < ALPHA:
        anova = stats.f_oneway(married, common_law, seperated, never_married)
        print('p-value: ', anova.pvalue)
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
        alpha = 0.05
    )
    print(posthoc)


# Test whether a person's income is affected by their gender.
def income_gender(dflog):
    # splitting data
    mf_income = dflog[['SEX','ATINC']] 
    male_income = mf_income.loc[mf_income['SEX'] == 'Male'] # male after tax income
    female_income = mf_income.loc[mf_income['SEX']== 'Female'] # female after tax income

    # ttest
    print('T-test p-value:',stats.ttest_ind(male_income['ATINC'], female_income['ATINC'],equal_var=False).pvalue)
    # since the p-value < 0.05, there is sufficient evidence to reject the null hypothesis that both male and female after tax income are the same 
    # therefore we conclude the income for male and females are different. 

    #print(male_income['ATINC'].mean())
    #print(female_income['ATINC'].mean())

    sns.histplot(data=mf_income, x='ATINC', hue="SEX")
    #male_income['ATINC'].hist()
    #female_income['ATINC'].hist()

    # fails equal variance and normality test even after log transforming the data?
    print('Levene-test p-value:', stats.levene(male_income['ATINC'], female_income['ATINC']).pvalue) # pvalue < 0.05 
    print('Normality-test for male income p-value:', stats.normaltest(male_income['ATINC']).pvalue) # p-value < 0.05 
    print('Normality-test for female income p-value;',stats.normaltest(female_income['ATINC']).pvalue) # p-value < 0.05 


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
    print("\nanova p-value:",anova.pvalue,"\n") 
    # alpha < 0.05 hence we have sufficient evidence to reject the null hypothesis that the means of the groups are the same.
    # to conclude, the means of the groups are different. 

    # post hoc Tukey Test
    posthoc = pairwise_tukeyhsd(maj_income['ATINC'], maj_income['MAJRI'], alpha=0.05)
    print(posthoc)

    #fig = posthoc.plot_simultaneous();
    #plt.show()


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