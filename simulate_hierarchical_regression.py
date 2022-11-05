import os
import pickle
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

parent_dir = '/Users/nakanomasahiro/PycharmProjects/research/random/Songs2022/workspace/'
import sys
sys.path.append(parent_dir)
from utils_models import *
from funcs_model_simulation import *
from funcs_learning_curves import *
from utils_sampling import *
from funcs_behavior_logistic_analysis import *
from utils_dataset import *

datasetName = 'takahashi2016roesch2009burton2018Valid'
modelName = 'fourState_full'

## 1. simulate data
NSessions = 100
NTrials = 57

params = get_params(datasetName, modelName, parent_dir)
# run simulation
dataValid = model_simulate(modelName, params, NSessions, NTrials)

## 2. run hierarchical logistic regression
# mark if a trial is correct or not
dataValid['correctChoice'] = 1*(dataValid['odor']=='left') + 2*(dataValid['odor']=='right') + (dataValid['odor']=='free')*(
    1*((dataValid['blockType']=='short_long')|(dataValid['blockType']=='big_small')) +
    2*((dataValid['blockType']=='long_short')|(dataValid['blockType']=='small_big')) )
dataValid['correct'] = (dataValid['correctChoice'] == dataValid['choice'])

stanCodeName = 'logisticRegressionMultiVar'
analysisName = 'lastWrong_NCorrectForced_improvement'
fitName = datasetName + '_' + analysisName
ratList, startSubject, NCorrectForcedBetweenList, accuracyList = prep_data(dataValid, fitName)
dd = dict(P=1, Ns=dataValid['rat'].unique().size, Nt=len(ratList), startSubject=startSubject, X=np.array(NCorrectForcedBetweenList)[:,np.newaxis], y=np.array(accuracyList))
fit = fitModel(modelName=stanCodeName, datasetName=fitName, dd=dd, samplingInfo=samplingInfo, moreControl={'max_treedepth':10, 'adapt_delta':0.99}, parent_dir=parent_dir, postfix='_simu')

## 3. compare the results
sort=False
reorder = False
iRegressor = 1
allSamples = pd.read_csv(parent_dir + 'model_fits/' + fitName + '_' + stanCodeName + '_allSamples.csv')
coeff = allSamples.loc[allSamples['warmup']==0, [col for col in allSamples if col.startswith('beta['+str(iRegressor+1))]].values
if sort:
    coeff = coeff[:, np.argsort(np.mean(coeff, axis=0))] # sort
if reorder:
    coeff = coeff[:, ratOrder]
x = np.arange(coeff.shape[1])+1 # index of rat
y = np.mean(coeff, axis=0)
err_n = np.quantile(coeff, 0.025, axis=0)
err_p = np.quantile(coeff, 0.975, axis=0)
plt.axhline(0, color='gray', linestyle='--', linewidth=1.5)
plt.errorbar(x=x, y=y, yerr=[y-err_n, err_p-y], ecolor='k', fmt='None', capsize=5, linewidth=2)

simuSamples = pd.read_csv(parent_dir + 'model_fits/' + fitName + '_' + stanCodeName + '_allSamples_simu.csv')
coeff_simu = simuSamples.loc[simuSamples['warmup']==0, [col for col in simuSamples if col.startswith('beta['+str(iRegressor+1))]].values
x = np.arange(coeff_simu.shape[1])+1+len(ratOrder)
y = np.mean(coeff_simu, axis=0)
err_n = np.quantile(coeff_simu, 0.025, axis=0)
err_p = np.quantile(coeff_simu, 0.975, axis=0)
plt.errorbar(x=x, y=y, yerr=[y-err_n, err_p-y], ecolor='r', fmt='None', capsize=5, linewidth=2)
plt.savefig('fourState_full_simulation_hierarchical_regression.png')
