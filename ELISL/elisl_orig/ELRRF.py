# -*- coding: utf-8 -*-
"""ELISL's ELRRF model, vendored from the authors' repository.

Source: https://github.com/joanagoncalveslab/ELISL, commit
eaf5f8c502a6db580722d73ceef8b41f4e4fe1b7 (2024-04-22), files
  src/models/ELRRF.py    (sha256 f5ad1664c265...) : class ELRRF, combine_preds
  src/experiment.py      (sha256 6c1011a1bc7d...) : fit_predict_cross_validation,
                                                    balance_cancer_by_index,
                                                    balance_by_index
Tepeli YI, Seale C, Goncalves JP. ELISL: early-late integrated synthetic
lethality prediction in cancer. Bioinformatics 40(1):btad764 (2024).

Copyright (C) ytepeli. Distributed under the GNU General Public License v3
(ELISL/LICENSE, copied unchanged from the repository; its README badge says
LGPL v3, its LICENSE file is the GPL v3 text).

Everything below the import block is the authors' code, byte for byte, except
the lines marked `# HARNESS:`. There are five such changes, none of which
alters a fitted model or a prediction:
  1. the import block: `from src import config` is dropped (it creates
     data/ and results/ directories at import) along with the module-level
     path hack; the rest is the original's list;
  2./3. `clf.fit(..., verbose=False)` -> `clf.fit(...)` in fit_predict_report
     and fit_report. LightGBM >= 4 removed fit()'s `verbose` argument (it
     only silenced logging; the classifier is built with verbose=-1), and
     there it raises TypeError;
  4. fit_predict_report re-raises a training exception instead of returning
     all-zero probabilities. With (2) unpatched, every fit on LightGBM 4
     failed into that branch and the "model" silently scored 0 everywhere;
  5. evaluate_all_models: `DataFrame.append` (removed in pandas 2) ->
     pd.concat, the same frame.
"""
# HARNESS: import block (see 1. above)
import sys
import collections as coll
from collections import defaultdict
import pandas as pd
import numpy as np
import copy
from sklearn.metrics import auc, roc_auc_score, average_precision_score, make_scorer, precision_recall_curve, \
    matthews_corrcoef
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold
import lightgbm as lgb
import time

RANDOMIZING=False

class ELRRF:
    def __init__(self, boosting_type='rf', num_leaves=165, max_depth=- 1, learning_rate=0.1, n_estimators=400,
                 subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.0,
                 min_child_weight=5, min_child_samples=10, subsample=0.632, subsample_freq=1,
                 colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=5.0, random_state=124, n_jobs=- 1,
                 verbose=False, thold=0.5, process=True, is_ready_data=False,
                 use_single=True, use_comb=True, grid_searched=False, sep_train_test=False, balance=True,
                 balance_strategy='class_weight_train', return_model=False, pretrained_model=None, cancer='BRCA'):
        self.boosting_type = boosting_type
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample_for_bin = subsample_for_bin
        self.objective = objective
        self.class_weight = class_weight
        self.min_split_gain = min_split_gain
        self.min_child_weight = min_child_weight
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.thold = thold
        self.datasets_dict = coll.OrderedDict()
        self.train_datasets_dict = coll.OrderedDict()
        self.test_datasets_dict = coll.OrderedDict()
        self.both_datasets_dict = coll.OrderedDict()
        self.no_of_datasets = 0
        self.process = process
        self.is_ready_data = is_ready_data
        self.ready_datasets = coll.OrderedDict()
        self.no_of_ready_datasets = 0
        self.use_single = use_single
        self.use_comb = use_comb
        self.grid_searched = grid_searched
        self.sep_train_test = sep_train_test
        self.balance = balance
        self.balance_strategy = balance_strategy
        self.return_model=return_model
        self.pretrained_model = pretrained_model
        self.cancer=cancer
        self.contr = {}
        self.best_params = {}

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in dir(self):
            value = getattr(self, key)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=False)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def add_dataset(self, dataset_name, dataset, dataset_te=None):
        """
        Parameters
        ----------
        dataset_name:
            Name of the dataset
        dataset:
            pandas dataframe with columns = [sample_name, label, features0, ..., featuresn]
        """
        if dataset_te is None:
            self.datasets_dict[dataset_name] = dataset
            self.no_of_datasets = self.no_of_datasets + 1
        else:
            self.both_datasets_dict[dataset_name] = {'train': dataset, 'test':dataset_te}
            self.no_of_datasets = self.no_of_datasets + 1

    def add_datasets(self, datasets_to_add=coll.OrderedDict()):
        """
        Parameters
        ----------
        datasets_to_add:
            Ordered dict where
                key=dataset_name,
                value= pandas dataframe with columns = [sample_name, label, features0, ..., featuresn]
        """
        for ds_name, ds in datasets_to_add.items():
            self.add_dataset(ds_name, ds)

    def balance_set(self, x, y, names, n=None, rand_state=-1):
        if rand_state == -1:
            rand_state = self.random_state
        n = np.unique(y, return_counts=True)[1].min() if n is None else n
        neg_indices = np.where(y == 0)[0]
        pos_indices = np.where(y == 1)[0]

        np.random.seed(rand_state)
        neg_sample_ind = np.random.choice(np.unique(y, return_counts=True)[1][0], n, replace=False)
        pos_sample_ind = np.random.choice(np.unique(y, return_counts=True)[1][1], n, replace=False)

        chosen_neg_ind = neg_indices[neg_sample_ind]
        chosen_pos_ind = pos_indices[pos_sample_ind]

        x_neg = x[chosen_neg_ind, :]
        x_pos = x[chosen_pos_ind, :]
        new_x = np.concatenate((x_neg, x_pos))

        y_neg = y[chosen_neg_ind]
        y_pos = y[chosen_pos_ind]
        new_y = np.concatenate((y_neg, y_pos))

        names_neg = names[chosen_neg_ind]
        names_pos = names[chosen_pos_ind]
        new_names = np.concatenate((names_neg, names_pos))

        return new_x, new_y, new_names

    def find_comb(self, choice=None):
        comb_dict = coll.OrderedDict()
        for ds1_name, ds1_data in self.datasets_dict.items():
            if self.use_single:
                comb_dict[ds1_name] = ds1_data
            if self.use_comb:
                for ds2_name in self.datasets_dict.keys():
                    ds_pair_name = min(ds1_name, ds2_name) + '|' + max(ds1_name, ds2_name)
                    if ds1_name != ds2_name and ds_pair_name not in comb_dict.keys():
                        comb_dict[ds_pair_name] = self.comb_2datasets(ds_pair_name)
        return comb_dict

    def comb_2datasets(self, ds_pair_name):
        df1 = self.datasets_dict[ds_pair_name.split('|')[0]]
        df2 = self.datasets_dict[ds_pair_name.split('|')[1]]
        df2 = df2.drop(columns='class')
        comb_df = pd.merge(df1, df2, how='inner', on=['pair_name'])
        return comb_df

    def prepare_train_test(self, df, train_test_ind=[], return_process_fits=False):
        train_ind = train_test_ind['train']
        is_test_included=True
        if 'test' in train_test_ind.keys():
            test_ind = train_test_ind['test']
        else:
            is_test_included=False

        if type(df) is dict:
            train_x = df['train'].iloc[train_ind].drop(columns=['pair_name', 'class']).values
            if is_test_included:
                test_x = df['test'].iloc[test_ind].drop(columns=['pair_name', 'class']).values
        else:
            train_x = df.iloc[train_ind].drop(columns=['pair_name', 'class']).values
            if is_test_included:
                test_x = df.iloc[test_ind].drop(columns=['pair_name', 'class']).values

        # test_names = df.loc[test_ind]['pair_name'].values
        # test_y = df.loc[test_ind]['SL'].values

        if RANDOMIZING:
            for idx in range(len(train_x)):
                np.random.seed(idx)
                np.random.shuffle(train_x[idx])
            lst_idx = idx
            if is_test_included:
                for idx in range(len(test_x)):
                    np.random.seed(lst_idx+idx)
                    np.random.shuffle(test_x[idx])


        if self.process:
            fitted_scale = preprocessing.StandardScaler().fit(train_x)
            train_x = fitted_scale.transform(train_x)
            if is_test_included and not return_process_fits:
                test_x = fitted_scale.transform(test_x)
            try:
                train_fits = VarianceThreshold(threshold=1e-04).fit(train_x)
                train_x = train_fits.transform(train_x)
                if is_test_included and not return_process_fits:
                    test_x = train_fits.transform(test_x)
            except:
                print('Variance could not be applied')
            # train_x = VarianceThreshold(threshold=1e-04).fit_transform(train_x)
            # test_x = VarianceThreshold(threshold=1e-04).fit_transform(test_x)

        # return train_x, train_y, test_x, test_y, train_names, test_names
        if is_test_included and not return_process_fits:
            return train_x, test_x
        elif is_test_included and return_process_fits:
            return train_x, test_x, fitted_scale, train_fits
        else:
            return train_x

    def precision_recall_scorer(self, truth, pred):
        precision, recall, thresholds = precision_recall_curve(truth, pred)
        return auc(recall, precision)

    def get_tree_search_grid(self, feature_size=None):
        n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        colsample_bytree = [0.5, 0.8, 1]
        if feature_size is not None:
            sqrt_size = np.sqrt(feature_size) / feature_size
            log_size = np.log2(feature_size) / feature_size
            colsample_bytree = [1, sqrt_size, log_size]
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=5)]
        max_depth.append(-1)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_child_samples = [1, 2, 4,10]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]  # Create the random grid
        reg_lambda = [5, 10]
        subsample = [0.632, 0.8, 0.99]
        random_grid = {'n_estimators': n_estimators,
                       'colsample_bytree': colsample_bytree,  # 'max_features': max_features,
                       'max_depth': max_depth,
                       # 'min_samples_split': min_samples_split,
                       'min_child_samples': min_child_samples,
                       # 'bootstrap': bootstrap,
                       'subsample': subsample,
                       'reg_lambda': reg_lambda}
        return random_grid

    def fit_predict_report(self, train_x, train_y, test_x, init_model=None):
        pos = sum(train_y)
        neg = len(train_y) - pos
        ratio = neg / pos
        is_unbalance = False
        if 'class_weight' in self.balance_strategy:
            is_unbalance = True
        if False:#self.grid_searched:
            clf = lgb.LGBMClassifier(boosting_type="rf",
                                     num_leaves=165,
                                     max_depth=self.max_depth,
                                     min_child_samples=self.min_samples_leaf,
                                     subsample=self.max_samples,
                                     subsample_freq=1,
                                     min_split_gain=0,  # min impurity split
                                     reg_alpha=0,  # L1 regularization
                                     random_state=self.random_state,
                                     n_jobs=-1,
                                     is_unbalance=is_unbalance,
                                     # bagging_freq=1,
                                     verbose=-1, silent=True)
            scorer = make_scorer(self.precision_recall_scorer, greater_is_better=True, needs_threshold=True)
            cv = 5
            clf = RandomizedSearchCV(estimator=clf, param_distributions=self.get_tree_search_grid(train_x.shape[1]),
                                     random_state=self.random_state, n_jobs=-1, verbose=-1, n_iter=5, cv=cv,
                                     scoring=scorer)
        else:
            clf = lgb.LGBMClassifier(boosting_type=self.boosting_type,
                                     num_leaves=self.num_leaves,
                                     colsample_bytree=self.colsample_bytree,
                                     n_estimators=self.n_estimators,
                                     min_child_weight=self.min_child_weight,
                                     min_child_samples=self.min_child_samples,
                                     subsample=self.subsample,
                                     subsample_freq=self.subsample_freq,
                                     min_split_gain=self.min_split_gain,
                                     reg_alpha=self.reg_alpha,
                                     reg_lambda=self.reg_lambda,  # L2 regularization
                                     is_unbalance=is_unbalance,
                                     n_jobs=self.n_jobs, verbose=-1)
        try:
            clf.fit(train_x, train_y, init_model=init_model)  # HARNESS: 2. no verbose= (LightGBM >= 4)
            zero_ind = np.where(clf.classes_ == 0)[0][0]
            one_ind = np.where(clf.classes_ == 1)[0][0]
            tr_probs = clf.predict_proba(train_x)[:, one_ind]
            tr_auc = average_precision_score(train_y, tr_probs)
            probs = clf.predict_proba(test_x)[:, one_ind]
            preds = clf.predict(test_x)
        except Exception as e_training:
            print(f'Exception happened: {e_training}')
            raise  # HARNESS: 4. never score a failed fit as all-zero
            clf=None
            tr_auc=0
            probs= np.zeros(len(test_x))
            preds= np.zeros(len(test_x))
        return probs, preds, tr_auc, clf

    def fit_report(self, train_x, train_y):
        pos = sum(train_y)
        neg = len(train_y) - pos
        ratio = neg / pos
        is_unbalance = False
        if 'class_weight' in self.balance_strategy:
            is_unbalance = True
        clf = lgb.LGBMClassifier(boosting_type=self.boosting_type,
                                 num_leaves=self.num_leaves,
                                 colsample_bytree=self.colsample_bytree,
                                 n_estimators=self.n_estimators,
                                 min_child_weight=self.min_child_weight,
                                 min_child_samples=self.min_child_samples,
                                 subsample=self.subsample,
                                 subsample_freq=self.subsample_freq,
                                 min_split_gain=self.min_split_gain,
                                 reg_alpha=self.reg_alpha,
                                 reg_lambda=self.reg_lambda,  # L2 regularization
                                 is_unbalance=is_unbalance,
                                 n_jobs=self.n_jobs, verbose=-1)
        clf.fit(train_x, train_y)  # HARNESS: 3. no verbose= (LightGBM >= 4)
        zero_ind = np.where(clf.classes_ == 0)[0][0]
        one_ind = np.where(clf.classes_ == 1)[0][0]
        tr_probs = clf.predict_proba(train_x)[:, one_ind]
        tr_auc = average_precision_score(train_y, tr_probs)
        return clf, one_ind, tr_probs, tr_auc

    def predict_report(self, clf, one_ind, test_x):
        probs = clf.predict_proba(test_x)[:, one_ind]
        preds = clf.predict(test_x)
        return probs, preds

    def evaluate_all_models(self, all_models):
        label_dict = coll.OrderedDict()
        pred_df = pd.DataFrame(columns=['pair_name', 'model', 'prediction', 'probability'])
        for model_name, model_res in all_models.items():
            if 'time' in model_name:
                continue
            model_pred_df = pd.DataFrame(columns=['pair_name', 'model', 'prediction', 'probability'])
            model_pred_df['pair_name'] = model_res['test_names']
            model_pred_df['prediction'] = model_res['predictions']
            model_pred_df['probability'] = model_res['probabilities']
            model_pred_df['tr_auc'] = model_res['tr_auc']
            model_pred_df['model'] = model_name
            pred_df = pd.concat([pred_df, model_pred_df], ignore_index=True)  # HARNESS: 5. DataFrame.append removed in pandas 2
            model_labels = dict(zip(model_res['test_names'], model_res['labels']))
            label_dict.update(model_labels)
        #grouped_preds = pred_df.groupby(['pair_name'])['probability'].mean().reset_index()
        grouped_preds = pred_df.groupby(['pair_name']).apply(combine_preds).reset_index()
        grouped_preds.loc[grouped_preds['probability'] >= self.thold, 'pred'] = 1
        grouped_preds.loc[grouped_preds['probability'] < self.thold, 'pred'] = 0
        final_preds = grouped_preds.rename(columns={'probability': 'prob'})
        final_preds['label'] = final_preds['pair_name'].map(label_dict)
        scores = {}
        scores['AUROC'] = roc_auc_score(final_preds['label'], final_preds['prob'])
        scores['AUPRC'] = average_precision_score(final_preds['label'], final_preds['prob'])
        scores['MC'] = matthews_corrcoef(final_preds['label'], final_preds['pred'])

        return scores

    def evaluate_folds(self, folds_prediction, report=True):
        AUROC = []
        AUPRC = []
        MC = []
        for fold, models in folds_prediction.items():
            try:
                fold_int = int(fold)
                scores = self.evaluate_all_models(models)
                AUROC.append(scores['AUROC'])
                AUPRC.append(scores['AUPRC'])
                MC.append(scores['MC'])
            except:
                print(sys.exc_info()[0])
            # print(f'Fold={fold}, AUROC={scores["AUROC"]}, AUPRC={scores["AUPRC"]}')
        AUROC = np.array(AUROC)
        AUPRC = np.array(AUPRC)
        MC = np.array(MC)
        if report:
            print(f'Average AUROC score: {AUROC.mean()}, with Std.Dev.: {np.std(AUROC)}')
            print(f'Average AUPRC score: {AUPRC.mean()}, with Std.Dev.: {np.std(AUPRC)}')
            print(f'Average MC score: {MC.mean()}, with Std.Dev.: {np.std(MC)}')
        return AUROC, AUPRC, MC

    def calculate_data_weights(self, folds_prediction, fold_main=0):
        if fold_main not in self.contr.keys() and str(fold_main) not in self.contr.keys():
            self.contr[fold_main] = {}
        for fold, models in folds_prediction.items():
            try:
                fold_int = int(fold)
                for model_name, model_res in models.items():
                    if 'time' in model_name:
                        continue
                    auprc_tmp = average_precision_score(model_res['labels'], model_res['probabilities'])
                    if fold_main in self.contr.keys() and model_name in self.contr[fold_main]:
                        self.contr[fold_main][model_name].append(auprc_tmp)
                    else:
                        self.contr[fold_main][model_name] = [auprc_tmp]
            except:
                pass
            # print(f'Fold={fold}, AUROC={scores["AUROC"]}, AUPRC={scores["AUPRC"]}')
        for cnt_name, cnt_val in self.contr[fold_main].items():
            self.contr[fold_main][cnt_name]=np.mean(self.contr[fold_main][cnt_name])

    def fit_predict_all_datasets(self, sample_dict=[], samples=None, init_model=None, cv=0):
        fold=cv
        combinations = self.find_comb()
        if not combinations:
            combinations = self.datasets_dict
        predictions = {}
        train_samples = samples.iloc[sample_dict['train']]
        test_samples = samples.iloc[sample_dict['test']]
        for comb_name, comb_data in combinations.items():
            train_x, test_x = self.prepare_train_test(comb_data, sample_dict)
            # train_x, train_y, train_names = self.balance_set(train_x, train_y, train_names)
            predictions[comb_name] = {'train_names': train_samples['pair_name'].values,
                                      'test_names': test_samples['pair_name'].values}
            if init_model is not None:
                comb_probs, comb_preds, tr_auc, model = self.fit_predict_report(train_x, train_samples['class'].values,
                                                                            test_x, init_model[comb_name]['model'])
            else:
                comb_probs, comb_preds, tr_auc, model = self.fit_predict_report(train_x, train_samples['class'].values,
                                                                            test_x)
            predictions[comb_name]['probabilities'] = comb_probs
            predictions[comb_name]['predictions'] = comb_preds
            predictions[comb_name]['labels'] = test_samples['class'].values
            if fold in self.contr.keys() and comb_name in self.contr[fold].keys():
                predictions[comb_name]['tr_auc'] = self.contr[fold][comb_name]
            else:
                predictions[comb_name]['tr_auc'] = tr_auc
            if self.return_model:
                predictions[comb_name]['model'] = model
        return predictions


    def fit_predict_cv(self, sample_dict=[], samples=None, train_idx=None):
        predictions = {}
        train_samples = samples.iloc[sample_dict['train']]
        test_samples = samples.iloc[sample_dict['test']]
        if self.both_datasets_dict:
            sets = self.both_datasets_dict.items()
        elif self.datasets_dict:
            sets = self.datasets_dict.items()
        for set_name, set_both_data in sets:
            if self.both_datasets_dict:
                comb_data = set_both_data['train'].iloc[train_idx]
            elif self.datasets_dict:
                comb_data = set_both_data.iloc[train_idx,:]
            train_x, test_x = self.prepare_train_test(comb_data, sample_dict)
            # train_x, train_y, train_names = self.balance_set(train_x, train_y, train_names)
            predictions[set_name] = {'train_names': train_samples['pair_name'].values,
                                      'test_names': test_samples['pair_name'].values}
            comb_probs, comb_preds, tr_auc, model = self.fit_predict_report(train_x, train_samples['class'].values,
                                                                            test_x)
            predictions[set_name]['probabilities'] = comb_probs
            predictions[set_name]['predictions'] = comb_preds
            predictions[set_name]['labels'] = test_samples['class'].values
            predictions[set_name]['tr_auc'] = tr_auc
            if self.return_model:
                predictions[set_name]['model'] = model
        return predictions


    def fit_predict_all_datasets_2sets(self, sample_dict, samples_tr=None, samples_te=None, fold=0):
        predictions = {}
        train_samples = samples_tr.iloc[sample_dict['train']]
        test_samples = samples_te.iloc[sample_dict['test']]
        for set_name, set_both_data in self.both_datasets_dict.items():
            train_x, test_x = self.prepare_train_test(set_both_data, sample_dict)
            # train_x, train_y, train_names = self.balance_set(train_x, train_y, train_names)
            predictions[set_name] = {'train_names': train_samples['pair_name'].values,
                                      'test_names': test_samples['pair_name'].values}
            comb_probs, comb_preds, tr_auc, model = self.fit_predict_report(train_x, train_samples['class'].values, test_x)
            predictions[set_name]['probabilities'] = comb_probs
            predictions[set_name]['predictions'] = comb_preds
            predictions[set_name]['labels'] = test_samples['class'].values
            if fold in self.contr.keys() and set_name in self.contr[fold].keys():
                predictions[set_name]['tr_auc'] = self.contr[fold][set_name]
            elif str(fold) in self.contr.keys() and set_name in self.contr[str(fold)].keys():
                predictions[set_name]['tr_auc'] = self.contr[str(fold)][set_name]
            else:
                #print('Contribution not found')
                predictions[set_name]['tr_auc'] = tr_auc
            if self.return_model:
                predictions[set_name]['model'] = model
        return predictions


    def calc_detail_dataset_importance(self, sample_dict, samples_tr=None, samples_te=None, fold=0):
        predictions = {}
        permute_predictions = {}
        train_samples = samples_tr.iloc[sample_dict['train']]
        test_samples = samples_te.iloc[sample_dict['test']]
        all_combined_dataset_name = ''
        for set_name, set_both_data in self.both_datasets_dict.items():
            permute_predictions[set_name]=[]
            train_x, raw_test_x, fitted_scale, train_fits = self.prepare_train_test(set_both_data, sample_dict, return_process_fits=True)
            test_x = fitted_scale.transform(raw_test_x)
            test_x = train_fits.transform(test_x)
            # train_x, train_y, train_names = self.balance_set(train_x, train_y, train_names)
            predictions[set_name] = {'train_names': train_samples['pair_name'].values,
                                      'test_names': test_samples['pair_name'].values}
            model, one_ind, train_probs, tr_auc = self.fit_report(train_x, train_samples['class'].values)
            comb_probs, comb_preds = self.predict_report(model, one_ind, test_x)
            #comb_probs, comb_preds, tr_auc, model = self.fit_predict_report(train_x, train_samples['class'].values, test_x)
            predictions[set_name]['probabilities'] = comb_probs
            predictions[set_name]['predictions'] = comb_preds
            predictions[set_name]['labels'] = test_samples['class'].values
            if fold in self.contr.keys() and set_name in self.contr[fold].keys():
                predictions[set_name]['tr_auc'] = self.contr[fold][set_name]
            elif str(fold) in self.contr.keys() and set_name in self.contr[str(fold)].keys():
                predictions[set_name]['tr_auc'] = self.contr[str(fold)][set_name]
            else:
                #print('Contribution not found')
                predictions[set_name]['tr_auc'] = tr_auc

            if '&' not in set_name:
                for i in range(20):
                    np.random.seed(self.random_state+i)
                    permuted_test_x = np.random.permutation(raw_test_x)
                    permuted_test_x = fitted_scale.transform(permuted_test_x)
                    permuted_test_x = train_fits.transform(permuted_test_x)
                    p_comb_probs, p_comb_preds = self.predict_report(model, one_ind, permuted_test_x)
                    permute_predictions[set_name].append({'probabilities': p_comb_probs,
                                                          'predictions': p_comb_preds})
            if '&' in set_name:
                all_combined_dataset_name = set_name
                col_id=0
                for set_name2, set_both_data2 in self.both_datasets_dict.items():
                    permute_predictions[set_name2+'_all'] = []
                    col_id_end = col_id + set_both_data2['test'].shape[1]-2
                    for i in range(20):
                        permuted_test_x = raw_test_x.copy()
                        np.random.seed(self.random_state+i)
                        permuted_test_x[:,col_id:col_id_end] = np.random.permutation(permuted_test_x[:,col_id:col_id_end])
                        permuted_test_x = fitted_scale.transform(permuted_test_x)
                        permuted_test_x = train_fits.transform(permuted_test_x)
                        p_comb_probs, p_comb_preds = self.predict_report(model, one_ind, permuted_test_x)
                        permute_predictions[set_name2+'_all'].append({'probabilities': p_comb_probs,
                                                              'predictions': p_comb_preds})
                    col_id = col_id_end

        perturb_res_dict = {}

        for set_name, set_both_data in self.both_datasets_dict.items():
            if '&' in set_name:
                continue
            perturb_res_dict[set_name] = []
            perturb_set_name = set_name
            all_models = copy.deepcopy(predictions)#{key: val for key, val in predictions.items()}
            original_score = self.evaluate_all_models(all_models)['AUPRC']
            original_error = 1.0 - original_score
            #original_set_names = list(set_both_data.keys())-[perturb_set_name]
            for i in range(20):
                all_pert_models = copy.deepcopy(predictions)#{key: val for key, val in predictions.items()}
                all_pert_models[set_name]['probabilities']=copy.deepcopy(permute_predictions[perturb_set_name][i]['probabilities'])
                all_pert_models[set_name]['predictions']=copy.deepcopy(permute_predictions[perturb_set_name][i]['predictions'])
                all_pert_models[all_combined_dataset_name]['probabilities']=copy.deepcopy(permute_predictions[perturb_set_name+'_all'][i]['probabilities'])
                all_pert_models[all_combined_dataset_name]['predictions']=copy.deepcopy(permute_predictions[perturb_set_name+'_all'][i]['predictions'])
                perturb_score = self.evaluate_all_models(all_pert_models)['AUPRC']
                perturb_error = 1.0 - perturb_score
                ratio = perturb_error / original_error
                perturb_res_dict[set_name].append(ratio)
            #perturb_res_dict[set_name] = np.mean(perturb_res_dict[set_name])

            if self.return_model:
                predictions[set_name]['model'] = model
        return perturb_res_dict


    def calc_dataset_importance(self, sample_dict, samples_tr=None, samples_te=None, fold=0):
        predictions = {}
        permute_predictions = {}
        train_samples = samples_tr.iloc[sample_dict['train']]
        test_samples = samples_te.iloc[sample_dict['test']]
        all_combined_dataset_name = ''
        for set_name, set_both_data in self.both_datasets_dict.items():
            permute_predictions[set_name]=[]
            train_x, raw_test_x, fitted_scale, train_fits = self.prepare_train_test(set_both_data, sample_dict, return_process_fits=True)
            test_x = fitted_scale.transform(raw_test_x)
            test_x = train_fits.transform(test_x)
            # train_x, train_y, train_names = self.balance_set(train_x, train_y, train_names)
            predictions[set_name] = {'train_names': train_samples['pair_name'].values,
                                      'test_names': test_samples['pair_name'].values}
            model, one_ind, train_probs, tr_auc = self.fit_report(train_x, train_samples['class'].values)
            comb_probs, comb_preds = self.predict_report(model, one_ind, test_x)
            #comb_probs, comb_preds, tr_auc, model = self.fit_predict_report(train_x, train_samples['class'].values, test_x)
            predictions[set_name]['probabilities'] = comb_probs
            predictions[set_name]['predictions'] = comb_preds
            predictions[set_name]['labels'] = test_samples['class'].values
            if fold in self.contr.keys() and set_name in self.contr[fold].keys():
                predictions[set_name]['tr_auc'] = self.contr[fold][set_name]
            elif str(fold) in self.contr.keys() and set_name in self.contr[str(fold)].keys():
                predictions[set_name]['tr_auc'] = self.contr[str(fold)][set_name]
            else:
                #print('Contribution not found')
                predictions[set_name]['tr_auc'] = tr_auc

            if '&' not in set_name:
                for i in range(20):
                    np.random.seed(self.random_state+i)
                    permuted_test_x = np.random.permutation(raw_test_x)
                    permuted_test_x = fitted_scale.transform(permuted_test_x)
                    permuted_test_x = train_fits.transform(permuted_test_x)
                    p_comb_probs, p_comb_preds = self.predict_report(model, one_ind, permuted_test_x)
                    permute_predictions[set_name].append({'probabilities': p_comb_probs,
                                                          'predictions': p_comb_preds})
            if '&' in set_name:
                all_combined_dataset_name = set_name
                col_id=0
                for set_name2, set_both_data2 in self.both_datasets_dict.items():
                    permute_predictions[set_name2+'_all'] = []
                    col_id_end = col_id + set_both_data2['test'].shape[1]-2
                    for i in range(20):
                        permuted_test_x = raw_test_x.copy()
                        np.random.seed(self.random_state+i)
                        permuted_test_x[:,col_id:col_id_end] = np.random.permutation(permuted_test_x[:,col_id:col_id_end])
                        permuted_test_x = fitted_scale.transform(permuted_test_x)
                        permuted_test_x = train_fits.transform(permuted_test_x)
                        p_comb_probs, p_comb_preds = self.predict_report(model, one_ind, permuted_test_x)
                        permute_predictions[set_name2+'_all'].append({'probabilities': p_comb_probs,
                                                              'predictions': p_comb_preds})
                    col_id = col_id_end

        perturb_res_dict = {}
        for set_name, set_both_data in self.both_datasets_dict.items():
            if '&' in set_name:
                continue
            perturb_res_dict[set_name] = []
            perturb_set_name = set_name
            #original_set_names = list(set_both_data.keys())-[perturb_set_name]
            all_models={key: val for key, val in predictions.items()}
            original_score = self.evaluate_all_models(all_models)
            for i in range(20):
                all_pert_models = {key: val for key, val in predictions.items()}
                all_pert_models[set_name]['probabilities']=permute_predictions[perturb_set_name][i]['probabilities']
                all_pert_models[set_name]['predictions']=permute_predictions[perturb_set_name][i]['predictions']
                all_pert_models[all_combined_dataset_name]['probabilities']=permute_predictions[perturb_set_name+'_all'][i]['probabilities']
                all_pert_models[all_combined_dataset_name]['predictions']=permute_predictions[perturb_set_name+'_all'][i]['predictions']
                perturb_score = self.evaluate_all_models(all_pert_models)['AUPRC']
                diff = original_score['AUPRC'] - perturb_score
                perturb_res_dict[set_name].append(diff)
            perturb_res_dict[set_name] = np.mean(perturb_res_dict[set_name])

            if self.return_model:
                predictions[set_name]['model'] = model
        return perturb_res_dict


    def calc_feature_importance(self, sample_dict, samples_tr=None, samples_te=None, fold=0):
        predictions = {}
        permute_predictions = {}
        train_samples = samples_tr.iloc[sample_dict['train']]
        test_samples = samples_te.iloc[sample_dict['test']]
        all_combined_dataset_name = ''
        for set_name, set_both_data in self.both_datasets_dict.items():
            train_x, raw_test_x, fitted_scale, train_fits = self.prepare_train_test(set_both_data, sample_dict, return_process_fits=True)
            test_x = fitted_scale.transform(raw_test_x)
            test_x = train_fits.transform(test_x)
            # train_x, train_y, train_names = self.balance_set(train_x, train_y, train_names)
            predictions[set_name] = {'train_names': train_samples['pair_name'].values,
                                      'test_names': test_samples['pair_name'].values}
            model, one_ind, train_probs, tr_auc = self.fit_report(train_x, train_samples['class'].values)
            comb_probs, comb_preds = self.predict_report(model, one_ind, test_x)
            #comb_probs, comb_preds, tr_auc, model = self.fit_predict_report(train_x, train_samples['class'].values, test_x)
            predictions[set_name]['probabilities'] = comb_probs
            predictions[set_name]['predictions'] = comb_preds
            predictions[set_name]['labels'] = test_samples['class'].values
            if fold in self.contr.keys() and set_name in self.contr[fold].keys():
                predictions[set_name]['tr_auc'] = self.contr[fold][set_name]
            elif str(fold) in self.contr.keys() and set_name in self.contr[str(fold)].keys():
                predictions[set_name]['tr_auc'] = self.contr[str(fold)][set_name]
            else:
                #print('Contribution not found')
                predictions[set_name]['tr_auc'] = tr_auc

            if '&' not in set_name:
                for feature_idx in range(raw_test_x.shape[1]):
                    permute_predictions[set_name+str(feature_idx)] = []
                    for i in range(20):
                        permuted_test_x = raw_test_x.copy()
                        np.random.seed(self.random_state+feature_idx+i)
                        permuted_test_x[:,feature_idx] = np.random.permutation(permuted_test_x[:,feature_idx])
                        permuted_test_x = fitted_scale.transform(permuted_test_x)
                        permuted_test_x = train_fits.transform(permuted_test_x)
                        p_comb_probs, p_comb_preds = self.predict_report(model, one_ind, permuted_test_x)
                        permute_predictions[set_name+str(feature_idx)].append({'probabilities': p_comb_probs,
                                                              'predictions': p_comb_preds})
            if '&' in set_name:
                all_combined_dataset_name = set_name
                col_id=0
                for set_name2, set_both_data2 in self.both_datasets_dict.items():
                    if '&' in set_name2:
                        continue
                    col_id_end = col_id + set_both_data2['test'].shape[1]-2
                    for feature_idx in range(set_both_data2['test'].shape[1]-2):
                        permute_predictions[set_name2+'_all_'+str(feature_idx)] = []
                        for i in range(20):
                            permuted_test_x = raw_test_x.copy()
                            np.random.seed(self.random_state+i+feature_idx+col_id)
                            try:
                                permuted_test_x[:,feature_idx+col_id] = np.random.permutation(permuted_test_x[:,feature_idx+col_id])
                            except:
                                print()
                            permuted_test_x = fitted_scale.transform(permuted_test_x)
                            permuted_test_x = train_fits.transform(permuted_test_x)
                            p_comb_probs, p_comb_preds = self.predict_report(model, one_ind, permuted_test_x)
                            permute_predictions[set_name2+'_all_'+str(feature_idx)].append({'probabilities': p_comb_probs,
                                                                  'predictions': p_comb_preds})
                    col_id = col_id_end

        perturb_res_dict = {}
        for set_name, set_both_data in self.both_datasets_dict.items():
            if '&' in set_name:
                continue
            for feature_idx in range(set_both_data['test'].shape[1]-2):
                perturb_res_dict[set_name+str(feature_idx)] = []
                perturb_set_name = set_name
                #original_set_names = list(set_both_data.keys())-[perturb_set_name]
                all_models={key: val for key, val in predictions.items()}
                original_score = self.evaluate_all_models(all_models)
                for i in range(20):
                    all_pert_models = {key: val for key, val in predictions.items()}
                    try:
                        all_pert_models[set_name]['probabilities']=permute_predictions[perturb_set_name+str(feature_idx)][i]['probabilities']
                    except:
                        print()
                    all_pert_models[set_name]['predictions']=permute_predictions[perturb_set_name+str(feature_idx)][i]['predictions']
                    all_pert_models[all_combined_dataset_name]['probabilities']=permute_predictions[perturb_set_name+'_all_'+str(feature_idx)][i]['probabilities']
                    all_pert_models[all_combined_dataset_name]['predictions']=permute_predictions[perturb_set_name+'_all_'+str(feature_idx)][i]['predictions']
                    perturb_score = self.evaluate_all_models(all_pert_models)['AUPRC']
                    diff = original_score['AUPRC'] - perturb_score
                    perturb_res_dict[set_name+str(feature_idx)].append(diff)
                perturb_res_dict[set_name+str(feature_idx)] = np.mean(perturb_res_dict[set_name+str(feature_idx)])

            if self.return_model:
                predictions[set_name]['model'] = model
        return perturb_res_dict


    def fit_all_datasets(self, sample_dict, samples_tr=None):
        predictions = {}
        train_samples = samples_tr.iloc[sample_dict['train']]
        for set_name, set_data in self.datasets_dict.items():
            train_x = self.prepare_train_test(set_data, sample_dict)
            # train_x, train_y, train_names = self.balance_set(train_x, train_y, train_names)
            predictions[set_name] = {'train_names': train_samples['pair_name'].values}
            model, one_ind, tr_auc = self.fit_report(train_x, train_samples['class'].values)
            predictions[set_name]['tr_auc'] = tr_auc
            if self.return_model:
                predictions[set_name]['model'] = model
                predictions[set_name]['one_ind'] = one_ind
        return predictions


    def fit_predict_cross_validation(self, samples, fold_type, n_split):
        if 'stratified' in fold_type and 'shuffle' in fold_type:
            skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=self.random_state)
        elif 'stratified' in fold_type:
            skf = StratifiedKFold(n_splits=n_split, shuffle=False, random_state=self.random_state)
        elif 'shuffle' in fold_type:
            skf = KFold(n_splits=n_split, shuffle=True, random_state=self.random_state)
        else:
            skf = KFold(n_splits=n_split, shuffle=False, random_state=self.random_state)
        indices = skf.split(samples, samples['class'].values)

        fold_ind = coll.OrderedDict()
        fold_samples = coll.OrderedDict()

        for i, (tr_indices, te_indices) in enumerate(indices):
            if 'undersample' in self.balance_strategy and 'train' in self.balance_strategy:
                tr_indices = balance_by_index(samples, tr_indices, rand_seed=124)
            if 'undersample' in self.balance_strategy and 'test' in self.balance_strategy:
                te_indices = balance_by_index(samples, te_indices, rand_seed=124)
            fold_ind[i] = {'train': tr_indices, 'test': te_indices}
            fold_samples[i] = {'train': samples.iloc[tr_indices], 'test': samples.iloc[te_indices]}

        results = coll.OrderedDict()
        for fold, sample_dict in fold_ind.items():
            fold_start_time = time.time()
            if self.pretrained_model is not None:
                results[fold] = self.fit_predict_all_datasets(sample_dict, samples, init_model=self.pretrained_model[fold])
            else:
                results[fold] = self.fit_predict_all_datasets(sample_dict, samples)
            lasted = str(time.time() - fold_start_time)
            results[fold]['time']=lasted

        return results


    def fit_predict_ho_cross_validation(self, samples, fold_type, n_split, ho=None):
        fold_ind = coll.OrderedDict()
        fold_samples = coll.OrderedDict()

        for i in range(n_split):
            genes = samples['pair_name'].str.split('|', expand=True)
            genes['SL'] = samples['class']
            all_genes = np.union1d(genes[0].values, genes[1].values)
            adj_genes = pd.DataFrame(index=all_genes, columns=all_genes)
            for ind, row in genes.iterrows():
                adj_genes.loc[row[0], row[1]] = row['SL']
                adj_genes.loc[row[1], row[0]] = row['SL']
            train_size, test_size = 1.0, 1.0
            train_genes, test_genes = [], []
            seed_id = i*len(all_genes)
            while (len(all_genes) > 0):
                if (train_size / test_size <= 4) & (len(all_genes) > 0):
                    np.random.seed(seed_id)
                    chosen_gene = np.random.choice(all_genes, 1)
                    all_genes = np.delete(all_genes, np.where(all_genes == chosen_gene))
                    train_genes.append(chosen_gene[0])
                    train_size = adj_genes.loc[train_genes, train_genes].notna().sum().sum() / 2
                if (train_size / test_size > 4) & (len(all_genes) > 0):
                    np.random.seed(seed_id)
                    chosen_gene = np.random.choice(all_genes, 1)
                    all_genes = np.delete(all_genes, np.where(all_genes == chosen_gene))
                    test_genes.append(chosen_gene[0])
                    test_size = adj_genes.loc[test_genes, test_genes].notna().sum().sum() / 2
                seed_id += 1

            print('double')
            tr_indices = ((genes[0].isin(train_genes)) & (genes[1].isin(train_genes))).values.nonzero()[0]
            te_indices = ((genes[0].isin(test_genes)) & (genes[1].isin(test_genes))).values.nonzero()[0]

            if 'undersample' in self.balance_strategy and 'train' in self.balance_strategy:
                tr_indices = balance_by_index(samples, tr_indices, rand_seed=124+i)
            if 'undersample' in self.balance_strategy and 'test' in self.balance_strategy:
                te_indices = balance_by_index(samples, te_indices, rand_seed=124+i)
            fold_ind[i] = {'train': tr_indices, 'test': te_indices}
            fold_samples[i] = {'train': samples.iloc[tr_indices], 'test': samples.iloc[te_indices]}

        results = coll.OrderedDict()
        for fold, sample_dict in fold_ind.items():
            fold_start_time = time.time()
            if self.pretrained_model is not None:
                results[fold] = self.fit_predict_all_datasets(sample_dict, samples, init_model=self.pretrained_model[fold])
            else:
                results[fold] = self.fit_predict_all_datasets(sample_dict, samples)
            lasted = str(time.time() - fold_start_time)
            results[fold]['time']=lasted

        return results


    def fit_predict_2set(self, samples_train, samples_test, n_split=5):
        fold_ind = coll.OrderedDict()
        fold_samples = coll.OrderedDict()
        tr_all_indices = np.array(list(range(len(samples_train))))
        te_all_indices = np.array(list(range(len(samples_test))))
        for i in range(n_split):
            tr_indices = tr_all_indices.copy()
            te_indices = te_all_indices.copy()
            if 'undersample' in self.balance_strategy and 'train' in self.balance_strategy:
                tr_indices = balance_by_index(samples_train, tr_all_indices, rand_seed=124+i)
            if 'undersample' in self.balance_strategy and 'test' in self.balance_strategy:
                te_indices = balance_by_index(samples_test, te_all_indices, rand_seed=124+i)
            fold_ind[i] = {'train': tr_indices, 'test': te_indices}
            fold_samples[i] = {'train': samples_train.iloc[tr_indices], 'test': samples_test.iloc[te_indices]}

        results = coll.OrderedDict()
        for fold, sample_dict in fold_ind.items():
            fold_start_time = time.time()
            results[fold] = self.fit_predict_all_datasets_2sets(sample_dict, samples_train, samples_test)
            lasted = str(time.time() - fold_start_time)
            results[fold]['time']=lasted

        return results


    def fit_set(self, samples_train, n_split=5):
        fold_ind = coll.OrderedDict()
        fold_samples = coll.OrderedDict()
        tr_all_indices = np.array(list(range(len(samples_train))))
        for i in range(n_split):
            tr_indices = tr_all_indices.copy()
            if 'undersample' in self.balance_strategy and 'train' in self.balance_strategy:
                tr_indices = balance_by_index(samples_train, tr_all_indices, rand_seed=124+i)
            fold_ind[i] = {'train': tr_indices}
            fold_samples[i] = {'train': samples_train.iloc[tr_indices]}

        results = coll.OrderedDict()
        for fold, sample_dict in fold_ind.items():
            fold_start_time = time.time()
            results[fold] = self.fit_all_datasets(sample_dict, samples_train)
            lasted = str(time.time() - fold_start_time)
            results[fold]['time']=lasted

        return results



# ---- src/experiment.py (lines 24-61, 455-508) ----


def fit_predict_cross_validation(model, samples, fold_type, n_split, train_idx=None):
    n_split= min(n_split, len(samples[samples['class']==1]), len(samples[samples['class']==0]))
    if 'stratified' in fold_type and 'shuffle' in fold_type:
        skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=model.random_state)
    elif 'stratified' in fold_type:
        skf = StratifiedKFold(n_splits=n_split, shuffle=False, random_state=model.random_state)
    elif 'shuffle' in fold_type:
        skf = KFold(n_splits=n_split, shuffle=True, random_state=model.random_state)
    else:
        skf = KFold(n_splits=n_split, shuffle=False, random_state=model.random_state)
    indices = skf.split(samples, samples['class'].values)

    fold_ind = coll.OrderedDict()
    fold_samples = coll.OrderedDict()

    for i, (tr_indices, te_indices) in enumerate(indices):
        if 'undersample' in model.balance_strategy and 'train' in model.balance_strategy:
            tr_indices = balance_by_index(samples, tr_indices, rand_seed=124+i)
        if 'undersample' in model.balance_strategy and 'test' in model.balance_strategy:
            te_indices = balance_by_index(samples, te_indices, rand_seed=124+i)
        fold_ind[i] = {'train': tr_indices, 'test': te_indices}
        fold_samples[i] = {'train': samples.iloc[tr_indices], 'test': samples.iloc[te_indices]}

    results = coll.OrderedDict()
    for fold, sample_dict in fold_ind.items():
        fold_start_time = time.time()
        if model.pretrained_model is not None:
            results[fold] = model.fit_predict_all_datasets(sample_dict, samples, init_model=model.pretrained_model[fold])
        elif train_idx is not None:
            results[fold] = model.fit_predict_cv(sample_dict, samples, train_idx)
        else:
            results[fold] = model.fit_predict_all_datasets(sample_dict, samples)
        lasted = str(time.time() - fold_start_time)
        results[fold]['time'] = lasted

    return results



def balance_cancer_by_index(samples, indices, rand_seed=-1):
    indices_from_cancers = []
    for cancer in samples['cancer'].unique():
        tr_samples = samples.iloc[indices]
        pos_indices = indices[(tr_samples['cancer'].values==cancer)&(tr_samples['class'].values == 1)]
        neg_indices = indices[(tr_samples['cancer'].values==cancer)&(tr_samples['class'].values == 0)]
        n = min(len(pos_indices), len(neg_indices))
        np.random.seed(rand_seed)
        if len(neg_indices)==len(pos_indices):
            pass
        elif n == len(neg_indices):
            np.random.seed(rand_seed)
            pos_indices = np.random.choice(pos_indices, n, replace=False)
        elif n == len(pos_indices):
            np.random.seed(rand_seed)
            neg_indices = np.random.choice(neg_indices, n, replace=False)
        else:
            np.random.seed(rand_seed)
            pos_indices = np.random.choice(pos_indices, n, replace=False)
            np.random.seed(rand_seed)
            neg_indices = np.random.choice(neg_indices, n, replace=False)

        combined_indices = np.union1d(pos_indices, neg_indices)
        combined_indices.sort()
        indices_from_cancers.append(combined_indices)
    combined_indices = np.concatenate(indices_from_cancers)
    return combined_indices

def balance_by_index(samples, indices, rand_seed=-1):
    if 'cancer' in samples.columns:
        return balance_cancer_by_index(samples, indices, rand_seed=rand_seed)
    tr_samples = samples.iloc[indices]
    pos_indices = indices[tr_samples['class'].values == 1]
    neg_indices = indices[tr_samples['class'].values == 0]
    n = min(len(pos_indices), len(neg_indices))
    np.random.seed(rand_seed)
    if len(neg_indices)==len(pos_indices):
        pass
    elif n == len(neg_indices):
        np.random.seed(rand_seed)
        pos_indices = np.random.choice(pos_indices, n, replace=False)
    elif n == len(pos_indices):
        np.random.seed(rand_seed)
        neg_indices = np.random.choice(neg_indices, n, replace=False)
    else:
        np.random.seed(rand_seed)
        pos_indices = np.random.choice(pos_indices, n, replace=False)
        np.random.seed(rand_seed)
        neg_indices = np.random.choice(neg_indices, n, replace=False)

    combined_indices = np.union1d(pos_indices, neg_indices)
    combined_indices.sort()

    return combined_indices


# ---- src/models/ELRRF.py (lines 1014-1021) ----
def combine_preds(x):
    d = {}
    weights = x['tr_auc'] / x['tr_auc'].sum()
    #weights = np.repeat([1/x.shape[0]], x.shape[0])
    weighted_probs = x['probability'] * weights
    d['probability'] = weighted_probs.sum()
    d['avg'] = x['probability'].mean()
    return pd.Series(d, index=['avg', 'probability'])
