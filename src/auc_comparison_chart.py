import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import random


random.seed(92)

#   Results dataframe
df = pd.DataFrame([['development data (test set)', 96.38, 98.71, 98.95, 99.02, 99.23], 
                   ['PPMI dataset', 97.51, 99.12, 99.23, 99.31, 99.38], 
                   ['MPH dataset', 93.46, 92.42, 95.73, 96.12, 96.24]], 
                   columns=['dataset', 'SBR', 'PCA-RFC', 'CNN-MVT', 'CNN-RLT', "CNN-Regression"]) 

palette = sns.color_palette("tab10", 7)
random.shuffle(palette)

target_figsize = (6, 4.5)

ax = df.plot(x='dataset', 
            kind='bar', 
            stacked=False, 
            xlabel='',
            ylabel='AUC (%) of Balanced Accuracy over PIncObs',
            ylim=(90,100),
            color=palette,
            edgecolor='white', 
            width=0.7,
            linewidth=2,
            figsize=target_figsize, 
            zorder=5) 

"""
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=10)
"""

ax.grid(axis='y', zorder=0)
ax.set_yticks(range(90, 101, 1))
ax.set_xticklabels(df['dataset'], rotation=0)


plt.tight_layout()
plt.savefig("/Users/aleksej/IdeaProjects/master-thesis-kucerenko/src/results/evaluations/auc_main_comparison.png",
            dpi=300)