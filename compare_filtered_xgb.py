"""Compare tuned weighted XGBoost with vs without suspicious training rows.

Put this file, test.csv, steam_price_experimental_filtered_2026.csv,
and tune_model.py in the same folder. Run: python compare_filtered_xgb.py

The SAME validation games are used for both models. The final test set
is not evaluated; no model files are overwritten.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tune_model import FeaturePreprocessor, LIST_COLUMNS, model_from_config, sample_weights, metrics

# Winning configuration from tune_model.py, trial 35.
CONFIG = dict(
    n_estimators=700, max_depth=5, learning_rate=0.04,
    min_child_weight=2, subsample=0.85, colsample_bytree=0.65,
    reg_lambda=1.0, reg_alpha=1.0, gamma=2.0,
    weight20=3.0, weight40=6.0,
)

original = pd.read_csv('test.csv')
filtered_file = pd.read_csv('steam_price_experimental_filtered_2026.csv')

# Ensure that the CSV supplied really is a filtered COPY of this dataset.
if original.app_id.duplicated().any() or filtered_file.app_id.duplicated().any():
    raise ValueError('Duplicate Steam app_id detected; check the cleaning step.')
if set(original.columns) != set(filtered_file.columns):
    raise ValueError('Dataset columns differ. Rebuild test.csv using clean_dataset.py.')
source = original.set_index('app_id')
subset = filtered_file.set_index('app_id')
if not subset.index.isin(source.index).all():
    raise ValueError('Filtered CSV contains games missing from test.csv.')
if len(original) != 10959 or len(filtered_file) != 10842:
    raise ValueError('Unexpected dataset version. Expected 10959 / 10842 rows. Run the latest cleaner.')

# Identical split to train_model.py and tune_model.py.
train, remainder = train_test_split(original, test_size=0.30, random_state=42)
validation, final_test = train_test_split(remainder, test_size=0.50, random_state=42)
del final_test, remainder  # NEVER train on or score the final test set.

# Verify shared prices only in the train/validation portions, not the test set.
seen = pd.concat([train, validation]).set_index('app_id')
matching_ids = subset.index.intersection(seen.index)
if not np.allclose(subset.loc[matching_ids, 'price'], seen.loc[matching_ids, 'price']):
    raise ValueError('Some training/validation prices changed across CSV versions.')

allowed = set(filtered_file.app_id)
filtered_train = train[train.app_id.isin(allowed)]
clean_validation = validation.app_id.isin(allowed).to_numpy()
feature_columns = [col for col in LIST_COLUMNS if col in original.columns]

def evaluate(train_rows):
    prep = FeaturePreprocessor(feature_columns)
    X_train = prep.fit_transform(train_rows)
    X_val = prep.transform(validation)
    model = model_from_config(CONFIG, threads=4)
    y_train = train_rows.price.to_numpy(dtype=float)
    model.fit(X_train, y_train,
              sample_weight=sample_weights(y_train, CONFIG['weight20'], CONFIG['weight40']))
    return model.predict(X_val)

print(f'Training: {len(train)} original / {len(filtered_train)} filtered', flush=True)
print(f'Validation: {len(validation)} identical games for both models', flush=True)
print(f'Flagged validation games: {(~clean_validation).sum()} (tracked separately)', flush=True)
print('Final TEST set: untouched\n', flush=True)

print('Training original model...', flush=True)
original_pred = evaluate(train)
print('Training filtered model...', flush=True)
filtered_pred = evaluate(filtered_train)

# MAE is in USD; bias = predicted - actual (negative means underprediction).
y = validation.price.to_numpy(dtype=float)
results = []
for sample_name, mask in [('ALL validation', np.ones(len(y), dtype=bool)),
                          ('Unflagged validation', clean_validation)]:
    for model_name, pred in [('Original training', original_pred),
                             ('Filtered training', filtered_pred)]:
        result = metrics(y[mask], pred[mask])
        results.append(dict(sample=sample_name, model=model_name, **result))

report = pd.DataFrame(results)
columns = ['sample', 'model', 'overall_mae', 'mae_20_plus',
           'mae_20_to_40', 'mae_40_plus', 'bias_40_plus',
           'mae_20_to_60', 'within_5_20_to_60_pct']
print('\nCOMPARISON — lower MAE is better:', flush=True)
print(report[columns].round(2).to_string(index=False), flush=True)
report.to_csv('filtered_xgb_comparison.csv', index=False)

# Pairwise predictions to inspect any changed games.
predictions = validation[['app_id', 'name', 'price']].copy()
predictions['is_flagged'] = ~clean_validation
predictions['original_prediction'] = original_pred
predictions['filtered_prediction'] = filtered_pred
predictions.to_csv('filtered_xgb_predictions.csv', index=False)
print('\nSaved filtered_xgb_comparison.csv and filtered_xgb_predictions.csv')
