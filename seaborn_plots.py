import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def pretty_plot(df, plottitle):
    df_stacked = df[['Dataset', 'Model Cost', 'Idx Cost', 'Residual Cost']]
    sns.set(style="whitegrid")
    palette = {
        'Model Cost': (0.3, 0.6, 0.2, 1.0),    # lightgreen with 60% opacity
        'Idx Cost': (0.5, 0.8, 0.3, 0.8),    # lightgreen with 60% opacity
        'Residual Cost': (1.0, 0.5, 0.2, 0.6) # orange with 60% opacity
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    df_stacked.set_index('Dataset').plot(kind='bar', stacked=True, color=[palette['Model Cost'], palette['Idx Cost'], palette['Residual Cost']], ax=plt.gca())

    for i, row in df.iterrows():
        lcc_y_pos = row['Model Cost'] + row['Idx Cost']
        plt.text(i, lcc_y_pos, f"{row['LCCScore']:.2f}", color='darkgreen', ha='center', size=10 if len(df)>6 else 12)

    plt.title(plottitle, fontsize=16)
    plt.xlabel('Dataset', fontsize=14)
    plt.ylabel('Bits', fontsize=15)
    plt.xticks(rotation=45, size=10 if len(df)>6 else 12)
    plt.yticks(fontsize=10)
    ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='y')
    plt.tight_layout()
    plt.legend(title='Cost Type', labels=['Model Cost', 'Idx Cost', 'Residual Cost'], loc='upper left')
    plt.savefig('results/plots/' + plottitle.lower().replace(' ','-') + '.png')
    plt.show()
    #plt.clf()

# Provided data
#text_df = pd.DataFrame({
#    'Dataset': ['text-en', 'text-de', 'text-ie', 'simp-en', 'rand', 'repeat2', 'repeat5', 'repeat10'],
#    'Model Cost': [1630.11, 1446.82, 1421.84, 20.26, 0.00, 2.00, 11.61, 30.53],
#    'Idx Cost': [12413.92, 12868.71, 12346.86, 5799.01, 0.00, 0.00, 0.00, 0.00],
#    'Residual Cost': [8460.33, 9747.60, 10132.59, 0.00, 46811.18, 0.00, 0.00, 0.00],
#    'LCCScore': [14044.03, 14315.53, 13768.70, 5819.28, 0.00, 2.00, 11.61, 30.53]
#})
text_df = pd.read_csv('results/text/mean-results.csv', index_col=0)
text_df['Dataset'] = text_df.index
#text_df_main = text_df[text_df['Dataset'].isin(['text-en', 'text-de', 'text-ie', 'simp-en', 'rand'])]
text_df_main = text_df.T[['text-en', 'text-de', 'text-ie', 'simp-en', 'rand']].T
text_df_main.index = list(range(len(text_df_main)))
#text_df_repeat = text_df[text_df['Dataset'].isin(['repeat2', 'repeat5', 'repeat10'])]
text_df_repeat = text_df.T[['repeat2', 'repeat5', 'repeat10']].T
text_df_repeat.index = list(range(len(text_df_repeat)))
breakpoint()
pretty_plot(text_df_main, 'Description Lengths and LCC Score for Natural Language and Random Text')
pretty_plot(text_df_repeat, 'Description Lengths and LCC Score for Artificial Repetitive Text')

img_dsets = ['im', 'cifar', 'stripes', 'halves', 'rand']
#img_df = pd.DataFrame({d:pd.read_csv(f'results/images/{d}_results.csv', index_col=0).loc['means'] for d in img_dsets}).T
img_df = pd.concat([pd.read_csv(f'results/images/{d}_results.csv', index_col=0).loc['means'] for d in img_dsets], axis=1).T
img_df.index = list(range(len(img_df)))
img_df = img_df.drop(['proc_time', 'total'], axis=1)
img_df = img_df.rename({'lccscore': 'LCCScore', 'model_cost': 'Model Cost', 'idx_cost': 'Idx Cost', 'residuals': 'Residual Cost'}, axis=1)
img_df['Dataset'] = ['imagenet', 'cifar10', 'stripes', 'halves', 'rand']
#pretty_plot(img_df, 'Description Lengths and LCC Score for Images Raw')
img_df['Residual Cost'] = img_df['Residual Cost']/5
#pretty_plot(img_df, 'Description Lengths and LCC Score for Images')

audio_df = pd.read_csv('results/audio/main-mean-results.csv', index_col=0).T
audio_df = audio_df.drop(['orcavocs', 'orcavocs-background', 'birdsong', 'birdsong-background', 'gaussian-noise'])
audio_df['Dataset'] = audio_df.index
audio_df.index = list(range(len(audio_df)))
print(audio_df)
pretty_plot(audio_df, 'Description Lengths and LCC Score for Audio Raw')
audio_df['Residual Cost'] = audio_df['Residual Cost']/5
pretty_plot(audio_df, 'Description Lengths and LCC Score for Audio')
