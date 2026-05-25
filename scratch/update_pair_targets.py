import json

with open('biomedical_embedding_pipeline_ayan_v1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'train_pairs_target' in src and 'CONFIG' in src and 'test_pairs_target' in src:
        print(f"Found CONFIG in cell {i}")

        # Update train and test pair targets
        old_train = '"train_pairs_target": 2000,'
        new_train = '"train_pairs_target": 6000,'
        old_test  = '"test_pairs_target": 500,'
        new_test  = '"test_pairs_target": 800,'

        new_src = src
        if old_train in new_src:
            new_src = new_src.replace(old_train, new_train)
            print("Updated train_pairs_target: 2000 -> 6000")
        else:
            print("WARNING: train_pairs_target pattern not found")

        if old_test in new_src:
            new_src = new_src.replace(old_test, new_test)
            print("Updated test_pairs_target: 500 -> 800")
        else:
            print("WARNING: test_pairs_target pattern not found")

        nb['cells'][i]['source'] = new_src.splitlines(keepends=True)
        break

with open('biomedical_embedding_pipeline_ayan_v1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Saved.")
