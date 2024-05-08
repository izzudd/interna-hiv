import pandas as pd
import numpy as np
import deepchem as dc

def load_main_data():
  main_df = pd.read_csv('datasets/HIV.csv', encoding='unicode_escape')
  main_df['HIV_active'] = main_df['HIV_active'] == 1
  main_df = main_df.drop_duplicates()
  
  count = main_df['smiles'].value_counts()
  smiles_grouped = main_df[main_df['smiles'].isin(count.index[count > 1])].groupby('smiles').agg({
      'activity': lambda x: ', '.join(sorted(x)),
      'HIV_active': 'sum'
    }).reset_index()
  
  for _, row in smiles_grouped.iterrows():
    smiles, activity = row['smiles'], row['activity']
    rep_act, rep_active = ('CM', True) if activity == 'CA, CM' else ('CI', False)
    main_df.loc[main_df['smiles'] == smiles, 'activity'] = rep_act
    main_df.loc[main_df['smiles'] == smiles, 'HIV_active'] = rep_active

  main_df = main_df.drop_duplicates()
  return main_df
  
def scaffold_split(dataset):
  Xs = dataset['smiles'].values
  Ys = dataset['HIV_active'].values
  
  dataset = dc.data.NumpyDataset (X=Xs, y=Ys, ids=Xs)
  scaffoldsplitter = dc.splits.ScaffoldSplitter()
  return scaffoldsplitter.train_test_split(dataset)