from pathlib import Path
from glob import glob
import pandas as pd

dfs = []
ids = []
for f in glob('11_*.tsv'):
    id_ = Path(f).stem.split('11_')[1]
    ids.append(id_)
    dfs.append(pd.read_csv(f, sep='\t'))

df = pd.concat(dfs, keys=ids)
sampled = df.groupby(level=0).sample(100, random_state=141717)
sampled['human_sentence'] = sampled.apply(lambda row: row['sentence'].replace(row['pronoun_type'], '___'), axis=1)
sampled.to_csv('sampled_for_humans.tsv', sep='\t', index=None)
