from datasets import load_dataset
import json, os

os.makedirs('data', exist_ok=True)
ds = load_dataset('openlifescienceai/medmcqa')
for split, name in [('train', 'train'), ('validation', 'dev'), ('test', 'test')]:
    with open(f'data/{name}.json', 'w') as f:
        for row in ds[split]:
            f.write(json.dumps(row) + '\n')
    print(f'{name}.json done')