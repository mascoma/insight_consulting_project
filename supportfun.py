from IPython.core.pylabtools import figsize
figsize(12.5, 3.5)
import scipy.stats as stats
from scipy.stats import nbinom
import pymc3 as pm
import numpy as np
import pandas as pd
import scipy.stats as st
from datetime import datetime, date, time
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import scale
from sklearn import cross_validation, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn import cross_validation, linear_model

def feature_engineering (df):  ## what is the good strategy for all the data?
    # calculate the time length in second of each streak
    start_time = df['local_start_at'][0:].astype('str').tolist()
    end_time = df['local_stop_at'][0:].astype('str').tolist()
    time_len = {}
    for i in range(len(start_time)):
        if end_time[i] != 'nan' and start_time[i] != 'nan':
            time_len[i] = ((np.datetime64(end_time[i]) - np.datetime64(start_time[i])).astype(float))
        else:
            time_len[i] = -99999
    df['time_len'] = time_len.values()
    
    # split the start time stamps to year, month, day and time
    year_start = {}
    month_start = {}
    day_start = {}
    time_start = {}
    weekday = {}
    hour_start = {}
    minute_start = {}
    second_start = {}
    for i in range(len(start_time)):
        dt = datetime.strptime(start_time[i], "%Y-%m-%d %H:%M:%S")
        year_start[i] = dt.year
        month_start[i] = dt.month
        day_start[i] = dt.day
        weekday[i] = dt.weekday()
        hour_start[i] = dt.hour
        minute_start[i] = dt.minute
        second_start[i] = dt.second
    # convert the start time to munites
    time_minutes = np.array(hour_start.values())*60 + np.array(minute_start.values()) + np.array(second_start.values())/60
    df['start_year'] = year_start.values()
    df['start_month'] = month_start.values()
    df['start_day'] = day_start.values()
    df['weekday'] = weekday.values() # monday is 0, sunday is 6
    df['time_minutes'] = time_minutes
    df['start_ymd'], df['start_time'] = df['start_at'].str.split(' ', 1).str
    weekday_bi = {}
    for i in range(len(df['weekday'])):
        if (df['weekday'].iloc[i] == 0)|(df['weekday'].iloc[i] == 1)|(df['weekday'].iloc[i] == 2)|(df['weekday'].iloc[i] == 3)|(df['weekday'].iloc[i] == 4):
            weekday_bi[i] = 0
            continue
        if (df['weekday'].iloc[i] == 5)|(df['weekday'].iloc[i] == 6):
            weekday_bi[i] = 1
            continue
    df['weekday_bi'] = weekday_bi.values()
    
    # column to indicate how many rows need to be added to stratch the time interval between start and stop by minutes
    rep = {}
    for i in range(len(df['streak_type'])):
        rep[i] = len(range(int(math.ceil((df.time_len.iloc[i]/60))))) + 1
    df['replicates'] = rep.values()
    
    # remove bad records and 'active', 'badsignal', 'type_20', 'charging'
    df_clean = df[(df['streak_type'] != 'badsignal') & (df['streak_type'] != 'charging') & 
              (df['streak_type'] != 'active') & (df['streak_type'] != 'inactive') & 
              (df['streak_type'] != 'type_20') &
              (df['streak_type'] != 'neutral') & (df['streak_type'] != 'activity') & 
              (df['start_year'] > 2010) &  (df['streak_type'] != 'sedentary') & 
              (df['time_len'] > 0) & (df['time_len'] < 10000)]
    
    # convert streak type to number
    df_clean.streak_type = df_clean.streak_type.replace('calm', 0)
    #df_clean.streak_type = df_clean.streak_type.replace('activity', 3)
    df_clean.streak_type = df_clean.streak_type.replace('focus', 1)
    df_clean.streak_type = df_clean.streak_type.replace('tense', 2)
    #df_clean.streak_type = df_clean.streak_type.replace('sedentary', 4)
    
    # stratch the table
    df_clean_ext = df_clean.loc[np.repeat(df_clean.index.values, df_clean.replicates)]
    time_min_ext = []
    for i in range(len(df_clean.time_minutes)):
        for j in range(df_clean.replicates.iloc[i]):
            if math.ceil((df_clean.time_len.iloc[i]/60)) == math.floor((df_clean.time_len.iloc[i]/60)):
                time_min_ext.append(df_clean.time_minutes.iloc[i] + j)
            else:
                if j < (df_clean.replicates.iloc[i]):
                    time_min_ext.append(df_clean.time_minutes.iloc[i] + j)
                if j == (df_clean.replicates.iloc[i]):
                    time_min_ext.append(df_clean.time_minutes.iloc[i] + (j-1) + (df_clean.time_len.iloc[i] - (j-1)*60)/60)
    df_clean_ext['record_min'] = time_min_ext
    
    return df_clean_ext


def hours(df_short):
    hour = [] 
    for i in range(len(df_short.record_min)):
        if 0 <= df_short.record_min[i] <= 60:
            hour.append(0)
        if 60 < df_short.record_min[i] <= 120:
            hour.append(1)
        if 120 < df_short.record_min[i] <= 180:
            hour.append(2) 
        if 180 < df_short.record_min[i] <= 240:
            hour.append(3)
        if 240 < df_short.record_min[i] <= 300:
            hour.append(4)
        if 300 < df_short.record_min[i] <= 360:
            hour.append(5)
        if 360 < df_short.record_min[i] <= 420:
            hour.append(6)
        if 420 < df_short.record_min[i] <= 480:
            hour.append(7)
        if 480 < df_short.record_min[i] <= 540:
            hour.append(8)
        if 540 < df_short.record_min[i] <= 600:
            hour.append(9)
        if 600 < df_short.record_min[i] <= 660:
            hour.append(10) 
        if 660 < df_short.record_min[i] <= 720:
            hour.append(11)
        if 720 < df_short.record_min[i] <= 780:
            hour.append(12)
        if 780 < df_short.record_min[i] <= 840:
            hour.append(13)
        if 840 < df_short.record_min[i] <= 900:
            hour.append(14)
        if 900 < df_short.record_min[i] <= 960:
            hour.append(15)  
        if 960 < df_short.record_min[i] <= 1020:
            hour.append(16)
        if 1020 < df_short.record_min[i] <= 1080:
            hour.append(17)
        if 1080 < df_short.record_min[i] <= 1140:
            hour.append(18)
        if 1140 < df_short.record_min[i] <= 1200:
            hour.append(19)
        if 1200 < df_short.record_min[i] <= 1260:
            hour.append(20) 
        if 1260 < df_short.record_min[i] <= 1320:
            hour.append(21)
        if 1320 < df_short.record_min[i] <= 1380:
            hour.append(22)
        if 1380 < df_short.record_min[i] <= 1440:
            hour.append(23)
        if  1440 < df_short.record_min[i]:
            hour.append(0) 
    return hour


def classfication(clf, nfold, feature_importance, X, y):
    kf = KFold(n_splits=nfold)
    kf.get_n_splits(X)
    scores = []
    Con = np.zeros([3, 3])
    for train_index, test_index in kf.split(X):
        
        clf.fit(X[train_index], y[train_index])
        scores.append(clf.score(X[test_index], y[test_index]))
        y_pred = clf.predict(X[test_index])
        Con += confusion_matrix(y[test_index], y_pred)
    Con = Con/Con.sum(axis=1)[:, np.newaxis]
    if feature_importance:
        print clf.feature_importances_
    
    print scores
    print 'accurancy: ' , np.round(np.array(scores).mean(), 3), np.round(np.array(scores).std(), 3), '\n'
    print 'confusion_matrix: ', np.round(Con, 3), '\n'

    
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def data_explore(data, color, ylable, title) : 
    count_data = np.array(data)
    n_count_data = len(count_data)
    plt.bar(np.arange(n_count_data), count_data, color=color)
    plt.xlabel("Time (days)")
    plt.ylabel(ylable)
    plt.title(title)
    plt.xlim(0, n_count_data);
    return count_data


def runmcmc(count_data, n, burnin):
    n_count_data = len(count_data)
    alpha = 1.0 / count_data.mean()  # Recall count_data is the
                               # variable that holds our txt counts
    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)

    tau = pm.DiscreteUniform("tau", lower=0, upper=n_count_data)

    print("Random output:", tau.random(), tau.random(), tau.random())

    @pm.deterministic
    def lambda_(tau=tau, lambda_1=lambda_1, lambda_2=lambda_2):
        out = np.zeros(n_count_data)
        out[:tau] = lambda_1  # lambda before tau is lambda1
        out[tau:] = lambda_2  # lambda after (and including) tau is lambda2
        return out

    observation = pm.Poisson("obs", lambda_, value=count_data, observed=True)

    model = pm.Model([observation, lambda_1, lambda_2, tau])

    # Mysterious code to be explained in Chapter 3.
    mcmc = pm.MCMC(model)
    mcmc.sample(n, burnin, 1)
    return mcmc

def runmcmc2(count_data, n, burnin):
    n_count_data = len(count_data)
    alpha = 1.0 / count_data.mean()  # Recall count_data is the
                               # variable that holds our txt counts
    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)
    lambda_3 = pm.Exponential("lambda_3", alpha)

    tau_1 = pm.DiscreteUniform("tau_1", lower=0, upper=(n_count_data - 1))
    tau_2 = pm.DiscreteUniform("tau_2", lower=tau_1, upper= n_count_data )

    @pm.deterministic
    def lambda_(tau_1=tau_1,tau_2=tau_2, lambda_1=lambda_1, lambda_2=lambda_2, lambda_3=lambda_3):
        out = np.zeros(n_count_data)
        out[:tau_1] = lambda_1  # lambda before tau is lambda1
        out[tau_1:tau_2] = lambda_2  # lambda after (and including) tau is lambda2
        out[tau_2:] = lambda_3
        return out

    observation = pm.Poisson("obs", lambda_, value=count_data, observed=True)

    model = pm.Model([observation, lambda_1, lambda_2, lambda_3, tau_1, tau_2])

    # Mysterious code to be explained in Chapter 3.
    mcmc = pm.MCMC(model)
    mcmc.sample(n, burnin, 1)
    return mcmc

def hourly_count(g, color, streaktype):
    figsize(28, 22)
    f, axarr = plt.subplots(4, 6)
    k = 0
    for i in range(6):
        for j in range(4):
            axarr[j, i].hist(g[g.hours == k].time_len/float(60), color = color)
            axarr[j, i].set_title('time ' + str(k) + ' ' + streaktype)
            k = k + 1
            axarr[j, i].set(xlabel= 'minute', ylabel= 'count')
    plt.show()
    

def get_npdatetime(df):
    np_dt = [] # np_dt for plot
    start_ymd = df.start_ymd.value_counts().index
    for i in range(len(start_ymd)):
        np_dt.append(np.datetime64(start_ymd[i]))
    np_dt.sort()
    return np_dt

def mcmc_model(df, i, niter, njobs): # n: number of trails; h: number of type observed
# mu: mean of prior distribution, tau: std of prior distribution, 
# niters: number of generations for each MCMC chain, njobs: number of MCMC chains 
    nparray = supportfun.get_npdatetime(df)
    with pm.Model() as model:
        alpha = 1/(df[df.hours == i].time_len/60).mean()  # pripor
        lambda_1 = pm.Exponential("lambda_1", alpha)
        # define likelihood
        with model:
            observation = pm.Poisson("obs", lambda_1, observed=(df[df.start_ymd==str(nparray[-1])].time_len/60)) # likelihood
        # inference
        with model:
            start = pm.find_MAP()
            step = pm.Metropolis()
            trace = pm.sample(niter, tune=5000, step=step, start=start, njobs=njobs)        
    return trace   
    
def significant_level(basemean, estimatemean, p_value):
    if ((p_value < 0.01) & (basemean > estimatemean)):
        return -2
    elif ((p_value < 0.05) & (basemean > estimatemean)):
        return -1
    elif ((p_value < 0.01) & (basemean < estimatemean)):
        return 2
    elif ((p_value < 0.05) & (basemean < estimatemean)):
        return 1
    elif (p_value > 0.05):
        return 0 

# plot a prediction
def plot_prediction(streak_level, title):
    figsize(12, 9)
    xobjects = ('tense', 'focused', 'calm')
    yobjects = ('very low', 'low', 'baseline', 'high', 'very high')
    colors = ('#f08080', '#4169e1', '#8fbc8f')
    df_level = pd.DataFrame(streak_level)
    x_pos = np.arange(len(xobjects))
    y_pos = np.array([-2, -1, 0, 1, 2])
    df_level.plot(kind = 'bar', color = colors, legend=None)
    plt.axhline(y=0,color='#cd853f',ls='dashed')
    plt.yticks(y_pos, yobjects, fontsize = 15) 
    plt.xticks(x_pos, xobjects, fontsize = 18, rotation='horizontal')
    plt.title(title)
    plt.show()
