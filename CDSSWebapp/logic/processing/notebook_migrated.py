from IPython.core.display_functions import display
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import re
import json
import os
import pandas as pd
import numpy as np
from pycaret.classification import *
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn import metrics as mt
from sklearn.metrics import confusion_matrix

import config

data_folder = config.TMP_DATA_FOLDER
models_folder = config.TMP_DATA_FOLDER + '/models'
tuned_models_folder = config.TMP_DATA_FOLDER + '/tuned_models'

# TODO JONA: Pasar al principio o a un archivo aparte (para validar variables) el análisis exploratorio.
#  Así se va a poder mostrar los gráficos por frontend.


# BLOQUE 4
def get_dataset(username, filename):
    return pd.read_csv(config.UPLOAD_FOLDER + '/' + username + '/' + filename)


def clone_dataset(dataset):
    return pd.DataFrame(dataset, columns=dataset.columns)


def reduce_features(dataset, threshold, dataset_target):
    dataset_reduced = dataset.copy()

    while True:
        columns_to_remove = []

        correlation = dataset_reduced.corr(numeric_only=True)
        correlation_keys = correlation.keys()

        for column in correlation:
            for i in range(correlation_keys.get_loc(column) + 1, len(correlation_keys)):
                column_to_remove = correlation.columns[i]

                if correlation[column][column_to_remove] >= threshold:
                    if column_to_remove != dataset_target and column_to_remove not in columns_to_remove:
                        columns_to_remove.append(column_to_remove)

        if not len(columns_to_remove):
            break
        else:
            dataset_reduced.drop(columns=columns_to_remove, inplace=True)

    return dataset_reduced


def load_json_file(name, default={}):
    try:
        f = open(data_folder + '/' + name)

        data = json.load(f)

        f.close()
    except:
        data = default

    return data


def save_json_file(name, data):
    with open(data_folder + '/' + name, 'w') as f:
        json.dump(data, f)


def load_models(folder):
    models = {}

    if os.path.exists(folder) and os.path.isdir(folder):
        files = os.listdir(folder)

        for f in files:
            if os.path.isfile(folder + '/' + f):
                basename = re.sub(r'\.pkl$', '', f)
                parts = basename.split('-')

                if len(parts) == 3:
                    model = load_model(
                        folder + '/' + basename,
                        verbose=False
                    )

                    if model:
                        dataset_id = parts[0]
                        train_size = parts[1]
                        model_id = parts[2]

                        if not dataset_id in models:
                            models[dataset_id] = {}

                        if not train_size in models[dataset_id]:
                            models[dataset_id][train_size] = {}

                        models[dataset_id][train_size][model_id] = model

    return models


def save_model_custom(folder, model, dataset_id, train_size, model_id):
    if not os.path.exists(folder):
        os.makedirs(folder)

    save_model(
        model,
        folder + '/' + dataset_id + '-' + train_size + '-' + model_id,
        model_only=True,
        verbose=False,
    )


def get_metric_display_names():
    return get_metrics()['Display Name'].values


def setup_dataset(dataset_id, train_size):
    setup(
        datasets[dataset_id],
        target=config.DATASET_TARGET,
        session_id=config.SESSION_ID,
        preprocess=False,
        use_gpu=True,
        train_size=train_size,
        verbose=False,
    )
    add_metric('specificity', 'Specificity', calculate_specificity)


def metric_is_better(value, original_value, metric_name):
    if metric_name == 'TT (Sec)':
        is_better = value < original_value
    else:
        is_better = value > original_value

    return is_better


def calculate_specificity(y, y_pred):
    tp, tn, fn, fp = 0.0, 0.0, 0.0, 0.0

    for i, target in enumerate(y):
        pred = y_pred[i]

        if target == pred:
            if target == 1:
                tp += 1
            else:
                tn += 1
        else:
            if target == 1:
                fn += 1
            else:
                fp += 1

    return tn / (tn + fp) if tn + tp != 0.0 else 0.0


# BLOQUE 5
datasets = {'original': get_dataset(username="admin", filename="survey.csv")}  # reemplazar por el archivo cargado

# BLOQUE 6
categoricals = []
numericals = []

for column_name in datasets['original'].columns:
    column = datasets['original'][column_name].dropna()

    if pd.api.types.is_numeric_dtype(column.dtype):
        numericals.append(column_name)
    elif column.dtype == 'object':
        uniques = column.unique()

        uniques.sort()

        if len(uniques) == 2 and uniques[0] == False and uniques[1] == True:
            datasets['original'][column_name] = column.astype(int)
        else:
            categoricals.append(column_name)

standard_scaler = StandardScaler()
min_max_scaler = MinMaxScaler()

datasets['original_standard_scaled'] = pd.DataFrame(datasets['original'], columns=datasets['original'].columns)
datasets['original_standard_scaled'][numericals] = standard_scaler.fit_transform(
    datasets['original_standard_scaled'][numericals])

datasets['original_min_max_scaled'] = pd.DataFrame(datasets['original'], columns=datasets['original'].columns)
datasets['original_min_max_scaled'][numericals] = min_max_scaler.fit_transform(
    datasets['original_min_max_scaled'][numericals])

if len(categoricals):
    datasets['onehot'] = pd.get_dummies(datasets['original'], columns=categoricals)
    datasets['onehot_standard_scaled'] = pd.get_dummies(datasets['original_standard_scaled'], columns=categoricals)
    datasets['onehot_min_max_scaled'] = pd.get_dummies(datasets['original_min_max_scaled'], columns=categoricals)

# BLOQUE 7 Verificar Duplicados
for dataset_id in datasets:
    dataset = datasets[dataset_id]
    print(dataset_id + ': ' + str(dataset.shape[0] - dataset.drop_duplicates().shape[0]))

# BLOQUE 8 Eliminación de nulos
for dataset_id in datasets:
    datasets[dataset_id].dropna(inplace=True)

# BLOQUE 9 Reducción de características
dataset_ids = datasets.copy().keys()

custom_limits = [0.4, 0.6, 0.8]
select_k_best_estimators = {
    'chi2': {
        'estimator': chi2,
        'validator': lambda dataset: (dataset.values >= 0).all(),
    },
    'f_classif': {
        'estimator': f_classif,
    }
}

for dataset_id in dataset_ids:
    if not re.search(r"_reduced_.+$", dataset_id):
        for limit in custom_limits:
            datasets[dataset_id + '_reduced_custom_' + str(limit)] = reduce_features(datasets[dataset_id], limit,
                                                                                     config.DATASET_TARGET)

        # We only use SelectKBest over one hot versions since the algorithm does not work with string values
        if re.search(r"^onehot_", dataset_id):
            for name, settings in select_k_best_estimators.items():
                dataset = datasets[dataset_id]

                validator = settings.get('validator')

                if not validator or validator(dataset):
                    estimator = SelectKBest(settings.get('estimator'))

                    estimator.fit_transform(dataset.drop(config.DATASET_TARGET, axis=1), dataset[config.DATASET_TARGET])

                    features = estimator.get_feature_names_out()

                    datasets[dataset_id + '_reduced_' + name] = datasets[dataset_id].loc[:,
                                                                np.append(features, [config.DATASET_TARGET])]

for dataset_id in datasets:
    print(dataset_id + ': ' + str(len(datasets[dataset_id].columns)))

# BLOQUE 10 Verificación de datasets duplicados

for dataset_id in datasets:
    for other_dataset_id in datasets:
        if dataset_id != other_dataset_id and datasets[dataset_id].equals(datasets[other_dataset_id]):
            print(dataset_id + ' = ' + other_dataset_id)

# BLOQUE 11 Grabación de datasets

datasets_folder = data_folder + '/datasets'

if not os.path.exists(datasets_folder):
    os.makedirs(datasets_folder)

for dataset_id in datasets:
    datasets[dataset_id].to_csv(datasets_folder + '/' + dataset_id + '.csv')

# BLOQUE 12

validation_datasets = {}
validation_ratio = config.VALIDATION_RATIO

for dataset_id in datasets:
    dataset = datasets[dataset_id].sample(frac=1, random_state=123)

    validation_limit = int(dataset.shape[0] * (1 - validation_ratio))

    train_test_dataset = dataset[0:validation_limit]
    validation_dataset = dataset[validation_limit:]

    datasets[dataset_id] = train_test_dataset
    validation_datasets[dataset_id] = validation_dataset

# BLOQUE 13

metrics = load_json_file('metrics.json')
train_sizes = [0.7, 0.8]
models = load_models(models_folder)
total = len(datasets) * len(train_sizes)
index = 0

for dataset_id in datasets:
    if not dataset_id in metrics:
        metrics[dataset_id] = {}

    if not dataset_id in models:
        models[dataset_id] = {}

    for train_size in train_sizes:
        train_size_string = str(train_size)

        index += 1

        print("Dataset ID " + dataset_id + ' (train size: ' + train_size_string + ', ' + str(index) + '/' + str(
            total) + ')')

        if not train_size_string in metrics[dataset_id]:
            setup_dataset(dataset_id, train_size)

            compare_models(
                verbose=False,
            )

            model_metrics = pull()

            metric_names = get_metric_display_names()

            metrics[dataset_id][train_size_string] = {}

            if not train_size_string in models[dataset_id]:
                models[dataset_id][train_size_string] = {}

            for model_id in model_metrics.index:
                if not model_id in metrics[dataset_id][train_size_string]:
                    metrics[dataset_id][train_size_string][model_id] = {}

                for metric_name in metric_names:
                    metrics[dataset_id][train_size_string][model_id][metric_name] = model_metrics[metric_name][model_id]

                save_json_file('metrics.json', metrics)

                if not model_id in models[dataset_id][train_size_string]:
                    model = create_model(
                        model_id,
                        verbose=False,
                    )

                    models[dataset_id][train_size_string][model_id] = model

                    save_model_custom(models_folder, model, dataset_id, train_size_string, model_id)

# BLOQUE 14
print(metrics)

# BLOQUE 15

tuned_models = load_models(tuned_models_folder)
tuned_metrics = load_json_file('tuned_metrics.json')
index = 0
total = 0

for dataset_id in models:
    for train_size in models[dataset_id]:
        for model_id in models[dataset_id][train_size]:
            total += 1

for dataset_id in models:
    if not dataset_id in tuned_metrics:
        tuned_metrics[dataset_id] = {}

    if not dataset_id in tuned_models:
        tuned_models[dataset_id] = {}

    for train_size in train_sizes:
        train_size_string = str(train_size)

        if not train_size_string in tuned_metrics[dataset_id]:
            tuned_metrics[dataset_id][train_size_string] = {}

        for model_id in models[dataset_id][train_size_string]:
            index += 1

            print(
                "Dataset ID " + dataset_id + ', model ' + model_id + ' (train_size: ' + train_size_string + ', ' + str(
                    index) + '/' + str(total) + ')')

            if model_id != 'dummy' and (model_id not in tuned_metrics[dataset_id][train_size_string] or
                                        tuned_metrics[dataset_id][train_size_string][model_id] == {}):
                setup_dataset(dataset_id, train_size)

                tuned_metrics[dataset_id][train_size_string][model_id] = {}

                if not train_size_string in tuned_models[dataset_id]:
                    tuned_models[dataset_id][train_size_string] = {}

                if not model_id in tuned_models[dataset_id][train_size_string]:
                    try:
                        tuned_model = tune_model(
                            models[dataset_id][train_size_string][model_id],
                            verbose=False,
                        )

                        tuned_models[dataset_id][train_size_string][model_id] = tuned_model

                        save_model_custom(tuned_models_folder, tuned_model, dataset_id, train_size_string, model_id)

                        tuned_metrics_for_model = pull()

                        metric_names = get_metric_display_names()

                        tuned_metrics[dataset_id][train_size_string][model_id] = {}

                        for metric_name in metric_names:
                            tuned_metrics[dataset_id][train_size_string][model_id][metric_name] = \
                                tuned_metrics_for_model[metric_name]['Mean']
                    except Exception as error:
                        print('Tuning failed: ', error)

                save_json_file('tuned_metrics.json', tuned_metrics)

# BLOQUE 16
# Compara métricas originales vs. tuneadas. Muestra una tabla con comparación de modelos.
# ESTO DEBE VER EL USUARIO QUE CARGÓ EL CSV, DESPUÉS DEL ANÁLISIS EXPLORATORIO

results = {}
total_metrics = 0
total_tuned_metrics = 0
total_tuned_metrics_better = 0

for dataset_id in metrics:
    for train_size in metrics[dataset_id]:
        train_size_string = str(train_size)

        for model_id in metrics[dataset_id][train_size_string]:
            if model_id != 'dummy':
                if model_id not in results:
                    results[model_id] = {}

                for metric_name in metrics[dataset_id][train_size_string][model_id]:
                    value = metrics[dataset_id][train_size_string][model_id][metric_name]

                    total_metrics += 1

                    if (
                            dataset_id in tuned_metrics
                            and train_size_string in tuned_metrics[dataset_id]
                            and model_id in tuned_metrics[dataset_id][train_size_string]
                            and metric_name in tuned_metrics[dataset_id][train_size_string][model_id]):
                        tuned_value = tuned_metrics[dataset_id][train_size_string][model_id][metric_name]

                        total_tuned_metrics += 1
                    else:
                        tuned_value = None

                    tuned = False

                    if tuned_value != None:
                        if metric_is_better(tuned_value, value, metric_name):
                            value = tuned_value
                            tuned = True
                            total_tuned_metrics_better += 1

                    include = False

                    if metric_name not in results[model_id]:
                        include = True
                    else:
                        current_value = results[model_id][metric_name].get('value')

                        if metric_is_better(value, current_value, metric_name):
                            include = True

                    if include:
                        results[model_id][metric_name] = {
                            'value': value,
                            'dataset_id': dataset_id,
                            'train_size': train_size,
                            'tuned': tuned
                        }

results_df = pd.DataFrame(results).transpose()
style = results_df.style

style.format(
    lambda cell: cell.get('dataset_id') + ' (' + str(cell.get('train_size')) + '): ' + str(cell.get('value')) + (
        '*' if cell.get('tuned') else ''))
style.format_index(str.upper)

# MOSTRAR ESTE CUADRO EN EL FRONTEND
display(style.apply(lambda row: ['background: yellow' if cell.get('value') == (
    row.map(lambda cell: cell.get('value')).max() if 'TT (Sec)' != row.name else row.map(
        lambda cell: cell.get('value')).min()) else '' for cell in row]))

best_datasets = {}
best_train_sizes = {}
model_metrics = {}

for model_id in results:
    for metric_name in results[model_id]:
        cell = results[model_id][metric_name]

        dataset_id = cell.get('dataset_id')
        train_size = cell.get('train_size')

        if dataset_id not in best_datasets:
            best_datasets[dataset_id] = 0

        if train_size not in best_train_sizes:
            best_train_sizes[train_size] = 0

        best_datasets[dataset_id] += 1
        best_train_sizes[train_size] += 1

        if model_id not in model_metrics:
            model_metrics[model_id] = 0

        model_metrics[model_id] += cell.get('value')

display(best_datasets)
display(best_train_sizes)
display('Tuned metrics: ' + str(total_tuned_metrics) + '/' + str(total_metrics) + ', better: ' + str(
    total_tuned_metrics_better))
display(sorted(model_metrics.items(), key=lambda x: x[1]))

# BLOQUE 17
# Se eligió un modelo específico (onehot_min_max_scaled_reduced_chi2) para seguir analizando.
# Esto se debe poder ver al hacer clic sobre una línea de la tabla del bloque 16.

dataset_id = 'onehot_min_max_scaled_reduced_chi2'
train_size = 0.7
train_size_string = str(train_size)
results = {}

for model_id in metrics[dataset_id][train_size_string]:
    if model_id != 'dummy':
        model_title = model_id.upper()

        results[model_title] = {}

        for metric_name in metrics[dataset_id][train_size_string][model_id]:
            value = tuned_metrics[dataset_id][train_size_string][model_id][metric_name]

            if value == 0:
                callback = None

                if metric_name == 'AUC':
                    callback = mt.roc_auc_score
                elif metric_name == 'Recall':
                    callback = mt.recall_score
                elif metric_name == 'Prec.':
                    callback = mt.precision_score
                elif metric_name == 'F1':
                    callback = mt.f1_score
                elif metric_name == 'Kappa':
                    callback = mt.cohen_kappa_score
                elif metric_name == 'MCC':
                    callback = mt.matthews_corrcoef

                if callback:
                    setup_dataset(dataset_id, train_size)
                    tuned_model = tuned_models[dataset_id][train_size_string][model_id]
                    dataset = datasets[dataset_id]
                    predicted = predict_model(tuned_model, datasets[dataset_id])
                    value = mt.roc_auc_score(dataset['depression_diagnosis'], predicted['prediction_label'])
                    tuned_metrics[dataset_id][train_size_string][model_id][metric_name] = value

            results[model_title][metric_name] = round(value, 2)

results_df = pd.DataFrame(results).transpose().sort_values(['AUC', 'F1'], ascending=False)

style = results_df.style

style.format_index(str.upper)
display(style.apply(lambda row: ['background: yellow' if cell == (
    row.map(lambda cell: cell).max() if 'TT (Sec)' != row.name else row.map(lambda cell: cell).min()) else '' for cell
                                 in row]))


# BLOQUE 18 AUXILIAR
def chart(dataset_id, model_id, plot):
    train_size = 0.8

    setup_dataset(dataset_id, train_size)

    plot_model(models[dataset_id][str(train_size)][model_id], plot)


# chart( 'onehot', 'xgboost', 'auc' )
# chart( 'onehot_reduced_custom_0.4', 'xgboost', 'auc' )
# chart( 'onehot_standard_scaled_reduced_f_classif', 'xgboost', 'auc' )
# chart( 'onehot_min_max_scaled_reduced_chi2', 'xgboost', 'auc' )
# chart( 'onehot_min_max_scaled_reduced_chi2', 'xgboost', 'feature_all' )
display(datasets['onehot_min_max_scaled_reduced_chi2'])

# BLOQUE 19 AUXILIAR
for dataset_id in datasets:
    print(dataset_id + ": ", end="")

    for column in datasets[dataset_id].columns:
        print(column + ", ", end="")

    print("")
    print("")

# BLOQUE 20
# Para un dataset específico, se hace el predict_model y la confusion Matrix,
# también haciendo clic en la tabla del bloque 16.

# print(metrics['onehot_min_max_scaled_reduced_chi2']['0.8']['ridge'])
# print(datasets['onehot_min_max_scaled_reduced_chi2'])

dataset = datasets['onehot']

setup_dataset('onehot', 0.7)
model = models['onehot']['0.7']['lda']

# model = model = create_model(
#    'xgboost'
# )

predict_model(model, dataset)

tuned_model = tuned_models['onehot']['0.7']['lda']

predicted = predict_model(tuned_model, dataset)

confusion_matrix(dataset['depression_diagnosis'], predicted['prediction_label'])

# BLOQUE 21

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

fig, _ = plt.subplots(nrows=1, figsize=(10, 10))
ax = plt.subplot(1, 1, 1)
ax.grid(False)

cm = confusion_matrix(dataset['depression_diagnosis'], predicted['prediction_label'])

disp = ConfusionMatrixDisplay(cm, display_labels=['No', 'Yes'])

disp.plot(ax=ax)

# BLOQUE 22

dataset_id = 'onehot_min_max_scaled_reduced_chi2'
train_size = 0.7
model_id = 'lda'

train_size_string = str( train_size )
dataset = validation_datasets[ dataset_id ]

setup_dataset( dataset_id, train_size )

tuned_model = tuned_models[ dataset_id ][ train_size_string ][ model_id ]

predicted = predict_model( tuned_model, dataset )

# BLOQUE 23
# Esto es parte del análisis exploratorio del inicio, para visualizar.

import seaborn as sns

#datasets['original']
sns.heatmap(datasets['original'].drop(['id', 'gender', 'who_bmi', 'depression_severity', 'anxiety_severity'], axis=1).corr())