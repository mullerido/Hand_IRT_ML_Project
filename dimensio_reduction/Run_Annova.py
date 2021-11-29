import numpy as np
import scipy.stats as stats
from bioinfokit.analys import stat

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from bioinfokit.analys import stat
from statsmodels.formula.api import ols
import statsmodels.api as sm

if __name__ == "__main__":

    result_df = pd.read_excel(r'G:\My Drive\Thesis\Project\Results\Two-D Laplacian\Dist Method Comparison.xlsx', header=0)

    dist_type = 'ave_dist_model'
    # Normality Assumption Check: Q-Q plot
    unique_majors = result_df['type'].unique()
    p = 0
    for major in unique_majors:
        plt.subplot(221+p)
        stats.probplot(result_df[result_df['type'] == major][dist_type], dist="norm", plot=plt)
        plt.title("Probability Plot - " + major)
        if p == 0 or p == 1:
            plt.xlabel('')
        plt.show()
        p+=1
    plt.suptitle('Normality Assumption Check', fontsize=20)

    '''
    ax0 = plt.subplot(221)
    res = stats.probplot(result_df[unique_majors[0]], plot=plt)
    plt.title("Probability Plot - " + unique_majors[0])
    plt.xlabel('')
    ax1 = plt.subplot(222)
    res = stats.probplot(result_df[unique_majors[1]], plot=plt)
    plt.title("Probability Plot - " + unique_majors[1])
    plt.xlabel('')
    ax2 = plt.subplot(223)
    res = stats.probplot(result_df[unique_majors[2]], plot=plt)
    plt.title("Probability Plot - " + unique_majors[2])
    ax3 = plt.subplot(224)
    res = stats.probplot(result_df[unique_majors[3]], plot=plt)
    plt.title("Probability Plot - " + unique_majors[3])
    '''
    # calculate ratio of the largest to the smallest sample standard deviation
    ratio = result_df.groupby('type').std().max() / result_df.groupby('type').std().min()
    print('Ratio = ' + str(ratio[dist_type]) + '  (Threshold < 2)')

    model = ols('ave_dist_model ~ C(type)', data=result_df).fit()
    w, pvalue = stats.shapiro(model.resid)
    print(w, pvalue)

    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_table

    print('note: if the data is balanced (equal sample size for each group), Type 1, 2, and 3 sums of squares '
          '(typ parameter) will produce similar results.')
    # generate a boxplot to see the data distribution by treatments. Using boxplot, we can
    # easily detect the differences between different treatments

    ax = sns.boxplot(x='type', y=dist_type, data=result_df, color='#99c2a2')
    ax = sns.swarmplot(x="type", y=dist_type, data=result_df, color='#7d0013')
    plt.title('ANOVA- Box plot', fontsize=20)
    plt.show()

    # Create ANOVA backbone table
    data = [['Between Groups', '', '', '', '', '', ''], ['Within Groups', '', '', '', '', '', ''],
            ['Total', '', '', '', '', '', '']]
    anova_table = pd.DataFrame(data, columns=['Source of Variation', 'SS', 'df', 'MS', 'F', 'P-value', 'F crit'])
    anova_table.set_index('Source of Variation', inplace=True)

    # calculate SSTR and update anova table
    x_bar = result_df[dist_type].mean()
    SSTR = result_df.groupby('type').count() * (result_df.groupby('type').mean() - x_bar) ** 2
    anova_table['SS']['Between Groups'] = SSTR[dist_type].sum()

    # calculate SSE and update anova table
    SSE = (result_df.groupby('type').count() - 1) * result_df.groupby('type').std() ** 2
    anova_table['SS']['Within Groups'] = SSE[dist_type].sum()

    # calculate SSTR and update anova table
    SSTR = SSTR[dist_type].sum() + SSE[dist_type].sum()
    anova_table['SS']['Total'] = SSTR

    # update degree of freedom
    anova_table['df']['Between Groups'] = result_df['type'].nunique() - 1
    anova_table['df']['Within Groups'] = result_df.shape[0] - result_df['type'].nunique()
    anova_table['df']['Total'] = result_df.shape[0] - 1

    # calculate MS
    anova_table['MS'] = anova_table['SS'] / anova_table['df']

    # calculate F
    F = anova_table['MS']['Between Groups'] / anova_table['MS']['Within Groups']
    anova_table['F']['Between Groups'] = F

    # p-value
    anova_table['P-value']['Between Groups'] = 1 - stats.f.cdf(F, anova_table['df']['Between Groups'],
                                                               anova_table['df']['Within Groups'])

    # F critical
    alpha = 0.05
    # possible types "right-tailed, left-tailed, two-tailed"
    tail_hypothesis_type = "two-tailed"
    if tail_hypothesis_type == "two-tailed":
        alpha /= 2
    anova_table['F crit']['Between Groups'] = stats.f.ppf(1 - alpha, anova_table['df']['Between Groups'],
                                                          anova_table['df']['Within Groups'])

    # Final ANOVA Table
    anova_table


########################################################################################################################

    # Multiple pairwise comparison
    # we will use bioinfokit (v1.0.3 or later) for performing tukey HSD test
    # check documentation here https://github.com/reneshbedre/bioinfokit

    # perform multiple pairwise comparison (Tukey's HSD)
    # unequal sample size data, tukey_hsd uses Tukey-Kramer test
    res = stat()
    res.tukey_hsd(df=result_df, res_var=dist_type, xfac_var='type', anova_model='dist ~ C(type)')
    res.tukey_summary
    # output