import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from main import dd

def load_pickle(obj):
    with open(obj, 'rb') as source:
        data = pickle.load(source)
    return data

results = load_pickle('results.pkl')

resdfs = []
columns = ['model', 'Splitting by', 'iteration', 'X_size']
for model, model_res in results.items():
    for split, split_res in model_res.items():
        for i, iter_res in split_res.items():            
            metrics = list(iter_res.keys())
            
            subdf = pd.DataFrame(columns=columns + metrics)
            X_size, _ = tuple(np.array(iter_res[metrics[-1]]).T)
            
            subdf['X_size'] = X_size
            subdf['model'] = np.repeat(model, len(X_size))
            subdf['Splitting by'] = np.repeat(split, len(X_size))
            subdf['iteration'] = np.repeat(i, len(X_size))
            
            for metric, metric_list in iter_res.items():
                subdf[metric] = [v[1] for v in metric_list]
            
            resdfs.append(subdf)

resdf = pd.concat(resdfs)

# plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

data_melt = resdf.melt(columns)
data_melt.replace('Random', 'Random sampling', inplace=True)
data_melt.replace('Stratified', 'Stratified sampling', inplace=True)

grouped = data_melt.groupby(['model', 'variable'])
hue_order = data_melt['Splitting by'].unique().tolist()
targets = zip(grouped.groups.keys(), axes.flatten())
for i, (group, ax) in enumerate(targets):
    ax.set_title(group[0])
    gdata = grouped.get_group(group).copy()
    
    legend=None
    if i + 1 == len(grouped):
        legend='brief'
    
    sns.lineplot(
        data = gdata,
        x = 'X_size',
        y = 'value',
        hue='Splitting by',
        ax=ax,
        legend=legend,
        hue_order=hue_order
    )
    
    ax.set_ylabel(group[1].split('_')[-1].title())
    ax.set_xlabel('Size of X_train')
    
fig.tight_layout() 
fig.savefig('results.png')