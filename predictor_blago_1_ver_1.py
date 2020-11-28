#!/usr/bin/env python
# coding: utf-8

# In[22]:


from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
import operator
import pymongo
from sklearn.preprocessing import StandardScaler

from pymongo import MongoClient
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import imageio
import json
import threading


# In[2]:



from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split

import pickle
from joblib import dump, load

from sklearn import svm


# In[3]:



import time
from dateutil import *
from dateutil.tz import *


# In[4]:


from datetime import datetime
timestamp = 1580943600 #1545730073
dt_object = datetime.fromtimestamp(timestamp)
#print("dt_object =", dt_object)
#print("type(dt_object) =", type(dt_object))


# In[5]:


client = MongoClient('mongodb://localhost:27017/')
db3 = client.CurrentForecastData
print(db3.list_collection_names())


# In[6]:


def ds_for_solarRad_hourly_forecast_model_from_weathebit3_e_prod_1_n(move,j,k,d,v,n): #rconvent 
    #ds1,blu01,en time Z to UTC, blu1, sinhron by last, correct,ghi_backwards 
    client = MongoClient('mongodb://localhost:27017/')
    db3 = client.CurrentForecastData
    #print(db3.collection_names()) 
    coll = db3["raw-meter-askoe-blago-1"]
    cursor = coll.find({'metering_channel': 'Active-','humantime':{ '$gt': '2020-05-17T23:59:23Z' }})  
    targets2 = []
    for doc in cursor:
        payload = doc['payload']
        for x, y in payload.items():
            if (x > '2020-08-31T23:59:23Z' and x < '2020-10-05T23:59:23Z') or x > '2020-10-07T23:59:23Z':
                datetime_object2 = datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ')
                timestamp2 = datetime.timestamp(datetime_object2)
                if not y['value'] == None:                    
                    targets2.append([x,timestamp2,y['value']])
                    targets2_last = timestamp2 
    targets2 = sorted(targets2,key=lambda d_t: d_t[0])       
                
    targets1_1 = targets2#[:-(k-j)]           
    
    #utc_zone = tz.gettz('UTC')
    #local_zone = tz.gettz('Europe/Kiev')
    utc_zone = tz.tzutc()
    local_zone = tz.tzlocal()
    print(local_zone)
    print(utc_zone)    
    targets1 = targets1_1
    for ti in range(len(targets1_1)):
        # Convert time string to datetime
        local_time = datetime.strptime(targets1_1[ti][0], '%Y-%m-%dT%H:%M:%SZ')
        # Tell the datetime object that it's in local time zone since 
        # datetime objects are 'naive' by default
        local_time = local_time.replace(tzinfo=local_zone)
        #print(local_time)
        # Convert time to UTC
        utc_time = local_time.astimezone(utc_zone)
        timestamp = datetime.timestamp(utc_time)
        #targets1[ti][0] = targets1_1[ti][0]
        targets1[ti][1] = timestamp
        
    print('targets1_last:   ', targets2_last)    
    print('targets1:___',len(targets1), targets1[:3],targets1[-3:]) 

    def ML_adding_vectors_to_input_matrix_and_target_from_similar_periodic_documents21(cursor):
        d_s = {}
        d_s_inf = {}
        d = 0
        for doc in cursor:
            d = d+1
            payload = doc['payload']
            data_1h = payload['data_1h']
            time = data_1h['time']
            ghi_instant =data_1h['ghi_instant']
            #print('d______',d) 
            for s in range(len(time)):
                it3 = time[s]
                t_s = int(datetime.timestamp(datetime.strptime(it3,'%Y-%m-%d %H:%M')))             
                for item in targets1:
                    #print('aaaaaaaa',x,y)                        
                    item1 = item[1] - 7200
                    #print("it,item1___",it[0],item1)
                    if (t_s >= item1 and t_s <= item1 + 180) or                     (t_s < item1 and t_s > item1 - 180): 
                        f_out = [[ghi_instant[s]],t_s,it3]
                        #print('f_out',f_out)
                        d_s[it3] = [[t_s,it3],item,[data_1h['ghi_backwards'][s]]]
                        break
                        """
                                    ,data_1h['ghi_instant'][s],\
                                        data_1h['gni_instant'][s],\
                                         data_1h['gni_backwards'][s],\
                                         data_1h['dni_instant'][s],\
                                        data_1h['dni_backwards'][s],data_1h['dif_instant'][s],\
                                         data_1h['dif_backwards'][s],\
                                        data_1h['extraterrestrialradiation_instant'][s],\
                                         data_1h['extraterrestrialradiation_backwards'][s]]]
                        """
                if len(targets1) > 0 and t_s > targets1[-1][1]-7200:
                    f_out = [[ghi_instant[s]],t_s,it3]
                    #print(f_out)
                    d_s_inf[it3] = [[t_s,it3],[data_1h['ghi_backwards'][s]]]
                    """
                                    ,data_1h['ghi_instant'][s],\
                                    data_1h['gni_instant'][s],\
                                     data_1h['gni_backwards'][s],\
                                     data_1h['dni_instant'][s],\
                                    data_1h['dni_backwards'][s],data_1h['dif_instant'][s],\
                                     data_1h['dif_backwards'][s],\
                                    data_1h['extraterrestrialradiation_instant'][s],\
                                     data_1h['extraterrestrialradiation_backwards'][s]]]
                    """
        return d_s, d_s_inf
      
    
    coll1 = db3["meteoblue-forecast"]

    cursor1 = coll1.find({ 'vendor': 'meteoblue.com'})#.limit(200)#,'humantime'> '2020-05-26T04:33:51Z'})

    d_s, d_s_inf = ML_adding_vectors_to_input_matrix_and_target_from_similar_periodic_documents21(
                    cursor1
    )
    features=[]
    features_time_utc=[]
    y_targets=[]
    y_targets_time=[]

    for x, y in d_s.items():
        if not (None in y[1] or None in y[2]):
            features.append(y[2])
            features_time_utc.append([[y[0][1],y[0][0]],y[0][0]])
            y_targets.append(y[1][2])
            y_targets_time.append(y[1][:2])
    X_features_inference = []
    X_features_inference_time_utc = []
    for x, y in d_s_inf.items():
        if not None in y[1]:
            X_features_inference.append(y[1])
            X_features_inference_time_utc.append([[y[0][1],y[0][0]],y[0][0]])

    X_features_inference2 = X_features_inference
    X_features_inference_time_utc2 = X_features_inference_time_utc    

    print('len(features), len(features_time_utc),len(y_targets),len(y_targets_time)',         len(features), len(features_time_utc),len(y_targets),len(y_targets_time))
    
    #print([item[0][0] for item in features_time_utc[:10]])
    #print([it[0] for it in y_targets_time[:10]])
    #print([item[1] for item in features_time_utc[-20:]])
    #print([it[1] for it in y_targets_time[-20:]])
    #print(len(features_time_utc), features_time_utc)

    targets_as_list_of_lists = y_targets #[item[1] for item in targets1]
    
    #print(targets_as_list_of_lists)
    print('infer',len(X_features_inference2),len(X_features_inference_time_utc2))
        
    
    return targets1,features,features_time_utc, y_targets, y_targets_time, X_features_inference2,            X_features_inference_time_utc2,targets1,'filtr2'


# In[7]:


def inference_stat_3_3_1_move_t(cod,ii,move,v,b,k,d,x,targets1,fc_1,X_features_inference,                         X_features_inference_time_utc1,adr):
    #stat1,-10800,fc with time 2 09-26 with h_rad
    targets1p = targets1
    #print(targets1p[-1])
    value = fc_1   
    f = open(adr, "r")
    predicted_before_2 = json.loads(f.read())
    f.close()
    
    X_features_inference_time_utc = predicted_before_2[9]
    h_inf = predicted_before_2[10]
    h_true = predicted_before_2[11]
    r_hour_err_for_values_more_100 =  predicted_before_2[12]
    mean_r_hour_err_for_values_more_100 =  predicted_before_2[13]
    h_begin = predicted_before_2[14]    
    loss_all = predicted_before_2[16]    
    whole_loss_ = predicted_before_2[-3]
    features_old = predicted_before_2[-2]
    h_rad = predicted_before_2[-1]
    r_hour_err_for_values_more_0_ = []
    r_hour_err_for_5_values_more_0_ = []
    h_inf_ = [it_i[1] for it_i in h_inf]
    h_true_ =[it_t[1] for it_t in h_true]
    fc_w_t_old =  predicted_before_2[0]    
    length_bound = int(len(predicted_before_2[0])/15)    
    print('len(predicted_before_2[0])',len(predicted_before_2[0]))
    if length_bound > 0:     
        print('***predicted_before_2[0]___:  ',len(X_features_inference_time_utc),len(predicted_before_2[0]),
              predicted_before_2[0][:length_bound],predicted_before_2[0][-length_bound:])
    print('X_features_inference stat',len(X_features_inference_time_utc1),len(X_features_inference),          X_features_inference[-3:],X_features_inference_time_utc1[-3:])
    #print('inference value:',[int(it) for it in predicted_before_2[0]],len(predicted_before_2[0]))
    Y_label_before_n = []
    time_after_call_n = []
    predicted_before = []
    jj=0 
    for it in fc_w_t_old:
        for item in targets1:
            item1 = item[1]-7200
            
            if ((it[1][1] >= item1 and it[1][1] <= item1 + 180) or (it[1][1] < item1 and it[1][1] >                                                                     item1 - 80))            and not item[1] in time_after_call_n:
                print('www', it[1][1])
                print("wwwww", it)

                time_after_call_n.append(item[1])
                Y_label_before_n.append(item)
                predicted_before.append([it[1][0],it[0]])
                h_inf.append([it[1][0],it[0]])
                h_true.append([item[0],item[2]])
                h_rad.append(features_old[jj])

                print('h_inf',h_inf,h_rad)
                h_inf_ = [it_i[1] for it_i in h_inf]
                h_true_ =[it_t[1] for it_t in h_true]
                break
        jj = jj + 1
        targets1_last = item        
  
    diff_all = abs(np.asarray(h_inf_, dtype=np.float32) -                               np.asarray(h_true_,dtype=np.float32))
    whole_energy = sum(np.asarray(h_true_,dtype=np.float32))
    print('whole_energy, np.sum(diff_all)___',ii,whole_energy,np.sum(diff_all))
    print('mean diff_all:___',np.mean(diff_all),len(diff_all))

    loss = []
    loss_ = []
    r_hour_err_for_5_values_all_ = []
    for n in range(len(h_inf)):
        if n % 1 == 0:
            r_hour_err_for_5_values_all_.append([h_inf[n],h_true[n],diff_all[n]]) 
        if h_true_[n] > 0:
     
            r_hour_err_for_values_more_0_.append(np.around(diff_all[n]/h_true_[n],2))
            r_hour_err_for_values_more_0_mean = np.mean(r_hour_err_for_values_more_0_)
            if n % 1 == 0:
                r_hour_err_for_5_values_more_0_.append([np.around(diff_all[n]/                                                h_true_[n],2),h_true[n],diff_all[n]])
                r_hour_err_for_5_values_more_0_mean = np.mean([it[0] for it                                                               in r_hour_err_for_5_values_more_0_])
                if diff_all[n]/h_true_[n] > 0.05:
                    loss.append(np.around(diff_all[n],2))
                    loss_.append([np.around(diff_all[n]/h_true_[n],2),h_true[n],diff_all[n]])
        else:
            r_hour_err_for_values_more_0_mean = None
            r_hour_err_for_5_values_more_0_mean =None

    whole_loss =  np.sum(loss)
    print('whole_loss_',whole_loss_)
    if whole_loss_:
        whole_loss_ = whole_loss_.append(r_hour_err_for_5_values_more_0_)
    else:
        whole_loss_ = []
    
    print('loss_all:___',loss_all,len(loss_all))
    print('loss r_hour_err_for_values_more_0_:---',r_hour_err_for_values_more_0_,          len(r_hour_err_for_values_more_0_))                
    print('r_hour_err_for_5_values_more_0_', r_hour_err_for_5_values_more_0_,len(r_hour_err_for_5_values_more_0_))
    print('loss:-',loss,len(loss))
    print('whole loss,whole_loss/whole_energy:',ii, whole_loss,np.around(whole_loss/(whole_energy+0.01),2),)
    print('loss_',loss_,len(loss_))
    print('r_hour_err_for_5_values_all_:',r_hour_err_for_5_values_all_,len(r_hour_err_for_5_values_all_))
    print('r_hour_err_for_values_more_0_mean:___', r_hour_err_for_values_more_0_mean)
    print('r_hour_err_for_5_values_more_0_mean:___',r_hour_err_for_5_values_more_0_mean)
    stat = [cod, str([ii,move,v,b,k,d,len(diff_all),whole_loss,whole_energy,np.sum(diff_all),                      np.around(whole_loss/(whole_energy+0.01),2),whole_loss_,x]),            str(r_hour_err_for_5_values_more_0_),            str(len(r_hour_err_for_5_values_more_0_)),str(np.array(loss_).tolist()),str(len(loss_))]    
    #print("loss_5_whole,loss_5_mean:  ",loss_5_whole,loss_5_mean)
    h_inf_99 = []    
    h_true_99 = []
    
    for i in range(len(h_true)):
        if h_true_[i] > 99:
            h_inf_99.append(np.around(h_inf_[i],2))
            h_true_99.append(np.around(h_true_[i]))
    print("***len predicted_before,Y_label_before_n _", len(predicted_before), len(Y_label_before_n))    
    Y_label_before_1 = []
    time_after_call = []

    for item in targets1:
    
        if item[1] >= predicted_before_2[2] and not item[1] in time_after_call:
            time_after_call.append(item[1])
            if len(Y_label_before_1) < len(predicted_before_2[0]):
                Y_label_before_1.append([item[2],item[0],item[1]])
    mi = min(len(predicted_before),len(Y_label_before_n))            
    de_before_1_c = abs(np.asarray([it_p[1] for it_p in predicted_before[:mi]], dtype=np.float32) -                         np.asarray([item_n[2] for item_n in Y_label_before_n[:mi]],dtype=np.float32))
    for n in range(mi):
        #Y_label_before_n[n][2] = 100
        if Y_label_before_n[n][2] > 99:
            print('de_before_1_c[n],Y_label_before_n[n][2]___',de_before_1_c[n],Y_label_before_n[n][2])
            r_hour_err_for_values_more_100.append(np.around(de_before_1_c[n]/Y_label_before_n[n][2],2))
    mean_r_hour_err_for_values_more_100 = [np.around(np.mean(r_hour_err_for_values_more_100),2),                                len(r_hour_err_for_values_more_100)]        
    print('***r_hour_err_for_values_more_100,mean_r_hour_err_for_values_more_100 ___',          r_hour_err_for_values_more_100, mean_r_hour_err_for_values_more_100)
    print('***h_inf_99_',h_inf_99)
    print('***h_true_99_', h_true_99)
    diff = abs(np.asarray(h_inf_99, dtype=np.float32) - np.asarray(h_true_99,dtype=np.float32))
    r_hour_err_for_values_more_100_ = []
    for n in range(len(h_inf_99)):  
            r_hour_err_for_values_more_100_.append(np.around(diff[n]/h_true_99[n],2))
    mean_r_hour_err_for_values_more_100_ = [np.around(np.mean(r_hour_err_for_values_more_100_),2),                                len(r_hour_err_for_values_more_100_)]        
    print('***r_hour_err_for_values_more_100_ ,mean_r_hour_err_for_values_more_100_  ___',          r_hour_err_for_values_more_100_, mean_r_hour_err_for_values_more_100_)    
    de_before_1 = sum(de_before_1_c)
    m =  len(Y_label_before_n)

    print('***Y_label_before_n[2] _',[it[2] for it in Y_label_before_n],len(Y_label_before_n))
    print('***Y_label_before_n _ ',[it for it in Y_label_before_n])
   
    print('***predicted_before: ', predicted_before)

    de_before_sum_1 = predicted_before_2[3] + de_before_1
    de_before_hours_sum_1 = predicted_before_2[4] + m    
 
    if m > 0 and de_before_hours_sum_1 > 0:
        #print('*** start 2020-02-27T19:00 fc before_1: ')
        er = round(de_before_1/m,2)
        print('*** Mean inference error: ', er)
        Mean_error = round(de_before_sum_1/de_before_hours_sum_1,2)
        li = predicted_before_2[6]
        li.append(er)
        test_time = predicted_before[-1][0]#[it[0][0][0] for it in predicted_before]

    else:
        li = predicted_before_2[6]
        test_time = predicted_before_2[15]
       
    if len(li) > 1:
        #li = li[1:]
        mean_inf_err = round(sum(np.asarray(li[1:],dtype=np.float32))/len(li[1:]),2)
    else:
        mean_inf_err = None
    #print('***inf_err - li,mean_inf_err: ', li, mean_inf_err)
    print('***first hour of forecast(utc): ', X_features_inference_time_utc1[0])
        
    predicted_before_2 = [value,targets1[-1][0], targets1[-1][1],de_before_sum_1,                          de_before_hours_sum_1,Y_label_before_n,li,mean_inf_err,len(li),
                          X_features_inference_time_utc1,h_inf,h_true,r_hour_err_for_values_more_100,\
                         mean_r_hour_err_for_values_more_100,h_begin,test_time,loss_all,whole_loss_,\
                         X_features_inference,h_rad]
    #if m > 0:
    f = open(adr, 'w')   
    y = json.dumps(predicted_before_2)
    f.write(y) 
    f.close()    
       
    print('***h_rad_____',h_rad)    
    print('***h_inf_____',np.around(h_inf_,2))
    print('***h_true_____',h_true_)
    y = np.array(h_inf_)
    y1 = np.array(h_true_)
    dif_h_trueh_h_inf = np.around(y1 - y,2)
    print('***dif_h_trueh_h_inf___',dif_h_trueh_h_inf)
    
    x_ax = range(len(h_inf))
    #plt.scatter(x_ax, y_predicted, s=5, color="blue", label="original")
    plt.plot(x_ax, y1, lw=0.8, color="red", label="true")
    plt.plot(x_ax, y, lw=0.8, color="green", label="predict")    
    plt.legend()
    plt.show()
    print(h_begin,' ... ', test_time)#X_features_inference_time_utc[0][0],', ... ,'
    if len(h_rad) > 1:
        h_rad1 = [it[0] for it in h_rad]
        print('***h_rad1_____',np.around(h_rad1,2))
        x_ax = np.array(h_rad1)
        print('xax',x_ax,y,y1)
        #plt.scatter(x_ax, y_predicted, s=5, color="blue", label="original")
        plt.plot(x_ax, y1, lw=0.8, color="red", label="true")
        plt.plot(x_ax, y, lw=0.8, color="green", label="predict")    
        plt.legend()
        plt.show()
        print(h_begin,' ... ', test_time)
        #if len(h_rad1) > 0:
        x_ax = np.array(h_rad1)
        print('xax',x_ax,y,y1)
        plt.scatter(x_ax, y1, s=15, color="red", label="true")
        plt.scatter(x_ax, y, s=15, color="green", label="predict")        
        #plt.plot(x_ax, y1, lw=0.8, color="red", label="true")
        #plt.plot(x_ax, y, lw=0.8, color="green", label="predict")    
        plt.legend()
        plt.show()
        print(h_begin,' ... ', test_time)          
    x = range(len(h_inf))
    #y = np.linspace(0, 10, 3)#np.sin(x)
    fig = plt.figure()
    ax = fig.add_subplot(111) # We'll explain the "111" later. Basically, 1 row and 1 column.    
    plt.plot(x, y, 'o', color='green',label= 'predict');
    plt.plot(x, y1, 'o', color='red',label= 'true');
    plt.legend()
    fig.set_figwidth(17)
    fig.set_figheight(7)    
    plt.show()    
    print(h_begin,' ... ', test_time)#X_features_inference_time_utc[0][0],', ... ,',X_features_inference_time_utc[-1][0])
    x = range(len(h_inf))
    #y = np.linspace(0, 10, 3)#np.sin(x)  
    plt.plot(x, dif_h_trueh_h_inf, 'o', color='blue',label= 'dif: h_trueh - h_inf');
    plt.legend()
    plt.show()

    return m, de_before_1_c,li,mean_inf_err,len(li),stat,whole_loss,whole_loss_,            r_hour_err_for_5_values_more_0_


# In[8]:



def opt_ML_solarRad_hourly_forecast_model_dataset_weatherbit3(model_opt,fc, X_features_inference,                                                          X_features_inference_time_utc,y_targets):
if len(X_features_inference) > 0:

    M = 0
    j =0
    for item in y_targets:
        if not item == None:
            if float(item) > M:
                M = float(item)
        else:
            y_targets[j] = -1
        j = j+1

    scaler = StandardScaler()  # doctest: +SKIP
    Y_label_np = np.array(y_targets, dtype=np.float32)
    fit_model = model_opt
    #if len(X_features_inference) > 0:
    predicted = fit_model.predict(X_features_inference[:48])#np.array().reshape(1, -1))
    fc_infer_1 = [int(item)*int(item>15)*int((M-item)>0) + M*int(int(item>M)) for item in predicted]
    fc_infer_1_cor = []
    for it in fc_infer_1:
        if it >1000:
            fc_infer_1_cor.append(it)
        else:
            fc_infer_1_cor.append(it)
    fc_infer_with_time = []
    fc_infer_1 = fc_infer_1_cor
    fc_old = fc
    for i in range(len(fc_infer_1)):
        if i < 4:
            for it in fc_old:
                if it[1][0] == X_features_inference_time_utc[i][0][0]:
                    fc_infer_with_time.append(it)
        else:
            fc_infer_with_time.append([fc_infer_1[i],X_features_inference_time_utc[i][0]])
    length_bound_old = int(len(fc_old)/15)
    length_bound_new = int(len(fc_infer_with_time)/15)
    print('length_old,length_bound_new',len(fc_old),len(fc_infer_with_time))
    if length_bound_old > 0:
        print('fc_old', fc_old[:2*length_bound_old],fc_old[-2*length_bound_old:])
    if length_bound_new:    
        print('fc_new', fc_infer_with_time[:2*length_bound_new],fc_infer_with_time[-2*length_bound_new:])
    print()
    prediction1 = {

        'product': 'solrad_48hours_v23_forecast',
        'forecast': 'solarRad',
        'timeUnit': '1 hour',
        'period': '1 hour',
        'duration': '48 hours',
        'startTime_utc': X_features_inference_time_utc[0][0],
        'startTimestamp': X_features_inference_time_utc[0][1],
        'value': fc_infer_with_time,
        'version': '23',
        'source': {'product': 'Hourly forecast weather data for 48 hour(s)', 'vendor': 'weathebit.io'}
        }
    db3['_48_hours_forecasts_5a'].insert(prediction1)

    x_inf = range(len(fc_infer_1))   
    y = np.array(fc_infer_1)
    #y1 = np.array(h_true)
    plt.plot(x_inf, y, 'o', color='green',label=str(X_features_inference_time_utc[0])+',...,'+str(X_features_inference_time_utc[-1]))
    plt.legend()
    plt.show()
    print(X_features_inference_time_utc[0][0],', ... ,',X_features_inference_time_utc[-1][0])
    x_inf = range(len(fc_infer_1))   
    y = np.array(fc_infer_1)
    #y1 = np.array(h_true)
    plt.plot(x_inf, y, 'o', color='green',label=str(X_features_inference_time_utc[0][0])+',...,'+             str(X_features_inference_time_utc[-1][0]))
    plt.legend()
    plt.show()
    print(X_features_inference_time_utc[0][0],', ... ,',X_features_inference_time_utc[-1][0])    
    x_ax = range(len(fc_infer_1))
    #plt.scatter(x_ax, y_predicted, s=5, color="blue", label="original")
    plt.plot(x_ax, fc_infer_1, lw=0.8, color="green", label="predicted")
    plt.legend()
    plt.show()

    print(X_features_inference_time_utc[0][0],', ... ,',X_features_inference_time_utc[-1][0])
    print('fc_infer_1: ', M,len(fc_infer_with_time),prediction1['startTimestamp'],          datetime.fromtimestamp(prediction1['startTimestamp']))
else:
    prediction1 = [] 
return prediction1,M


# In[9]:


def predict_c_n_comb(cod,ii,move,j,v,b1,k,d,n,x,adr,model_adressis,fc,model_opt):
    u = 10000
    opt_list = []
    X_features_inference_opt,model_opt,n_0,a_0,f_0,j_0,r_0,l_0,mean_nRMSE_0 = None,None,None,None,None,None,        None,None,None
    fc_np_m= []
    list_fc = []
    for k1 in range(b1):
        clf = load(model_adressis[k1]) 
        s = pickle.dumps(clf)
        model_opt1 = pickle.loads(s)         
        targets1,features,features_time_utc, y_targets, y_targets_time, X_features_inference,        X_features_inference_time_utc,keys_weatherbit,filtr2 =             ds_for_solarRad_hourly_forecast_model_from_weathebit3_e_prod_1_n(move,j,k,v,d,n[k1])
        if len(X_features_inference) > 0:

            prediction1,M = opt_ML_solarRad_hourly_forecast_model_dataset_weatherbit3(model_opt1,fc,                            X_features_inference, X_features_inference_time_utc,y_targets)
            #if len(prediction1):
            fc_new = prediction1        
            list_fc.append(fc_new['value'])
            fc_np_m = list_fc[0]

    m, de_before_1_c,li,mean_inf_err,len_li,stat,whole_loss,whole_loss_,r_hour_err_for_5_values_more_0_ =        inference_stat_3_3_1_move_t(cod,j,move,v,b1,k,d,x,targets1,fc_np_m,X_features_inference,
                             X_features_inference_time_utc,adr)
    print('m, de_before_1_c,li,mean_inf_err,len_li:___',m, np.around(de_before_1_c,2),np.around(li,2),          mean_inf_err,len_li)
    #return fc_np_m,stat,whole_loss,model_opt1,n_0,r_0,l_0,whole_loss_,r_hour_err_for_5_values_more_0_


# In[10]:


def fitting_n_opt_comb(j,k,v,b,d,move,n,adr,model_adressis):
    u = 10000   
    opt_list = []
    X_features_inference_opt,model_opt,n_0,a_0,f_0,j_0,r_0,l_0,mean_nRMSE_0 = None,None,None,None,None,None,        None,None,None
    fc_np_m= []
    list_fc = []
    for k1 in range(b):
        clf = load(model_adressis[k1]) 
        s = pickle.dumps(clf)
        model_opt1 = pickle.loads(s)
        print('model_opt1',model_opt1)
        targets1,features,features_time_utc, y_targets, y_targets_time, X_features_inference,            X_features_inference_time_utc,keys_weatherbit,filtr2 =                 ds_for_solarRad_hourly_forecast_model_from_weathebit3_e_prod_1_n(move,j,k,v,d,n[0])
        prediction1,M = opt_ML_solarRad_hourly_forecast_model_dataset_weatherbit3_start(model_opt1,                        X_features_inference, X_features_inference_time_utc,y_targets)
        fc = prediction1            
        list_fc.append(fc['value'])
        fc_np_m = list_fc[0]

    start_forecasting_timestamp = X_features_inference_time_utc[0][1] #fc['startTimestamp']
    start_forecasting_Z = datetime.fromtimestamp(start_forecasting_timestamp)
    start_forecasting_utc = X_features_inference_time_utc[0][0][0] #datetime.utcfromtimestamp(start_forecasting_timestamp)

    print('M,len(fc["value"]),start_forecasting_timestamp,start_forecasting_Z:___',M,len(fc_np_m),len(fc['value']),          start_forecasting_timestamp,start_forecasting_Z)
    print('X_features_inference_time_utc: ', len(X_features_inference_time_utc),
          X_features_inference_time_utc[0],'...',X_features_inference_time_utc[-1])    
    start_forecasting_Z = str(start_forecasting_Z)
    start_forecasting_utc = str(start_forecasting_utc) 
    #print('filtr2',len(filtr2),filtr2[:5],filtr2[-5:])
    predicted_before_2 = [fc_np_m, '', X_features_inference_time_utc[0][1], 0.01, 1, [], [],
                          0, 0, X_features_inference_time_utc,[['',0]],[['',0]],[],[],\
                          [start_forecasting_utc, start_forecasting_Z],[[[]]],[0],[],\
                          X_features_inference,[[0]]]
    f = open(adr, "w")

    y = json.dumps(predicted_before_2)
    f.write(y) 
    f.close()
    return fc_np_m,model_opt1, X_features_inference_time_utc


# In[11]:



def opt_ML_solarRad_hourly_forecast_model_dataset_weatherbit3_start(model_opt, X_features_inference,                                                          X_features_inference_time_utc,y_targets):    
M = 0
j =0
for item in y_targets:
    if not item == None:
        if float(item) > M:
            M = float(item)
    else:
        y_targets[j] = -1
    j = j+1

scaler = StandardScaler()  # doctest: +SKIP    
Y_label_np = np.array(y_targets, dtype=np.float32)

fit_model = model_opt
predicted = fit_model.predict(X_features_inference[:48])#np.array().reshape(1, -1))

#print('mean_nRMSE:  %.2f'%  mean_nRMSE_0)

fc_infer_1 = [int(item)*int(item>0)*int((M-item)>0) + M*int(int(item>M)) for item in predicted]
fc_infer_with_time = []
for i in range(len(fc_infer_1)):
    fc_infer_with_time.append([fc_infer_1[i],X_features_inference_time_utc[i][0]])
length_bound = int(len(fc_infer_with_time)/15)    
print('len(fc)',len(fc_infer_with_time))
if length_bound > 0:
    print('fc', fc_infer_with_time[:length_bound],fc_infer_with_time[-length_bound:])
print()
prediction1 = {

    'product': 'solrad_48hours_v23_forecast',
    'forecast': 'solarRad',
    'timeUnit': '1 hour',
    'period': '1 hour',
    'duration': '48 hours',
    'startTime_utc': X_features_inference_time_utc[0][0],
    'startTimestamp': X_features_inference_time_utc[0][1],
    'value': fc_infer_with_time,
    'version': '23',
    'source': {'product': 'Hourly forecast weather data for 48 hour(s)', 'vendor': 'weathebit.io'}
    }
x_inf = range(len(fc_infer_1))   
y = np.array(fc_infer_1)
#y1 = np.array(h_true)
plt.plot(x_inf, y, 'o', color='green',label=str(X_features_inference_time_utc[0])+',...,'+str(X_features_inference_time_utc[-1]))
plt.legend()
plt.show()
print(X_features_inference_time_utc[0][0],', ... ,',X_features_inference_time_utc[-1][0])
x_inf = range(len(fc_infer_1))   
y = np.array(fc_infer_1)
#y1 = np.array(h_true)
plt.plot(x_inf, y, 'o', color='green',label=str(X_features_inference_time_utc[0][0])+',...,'+         str(X_features_inference_time_utc[-1][0]))
plt.legend()
plt.show()
print(X_features_inference_time_utc[0][0],', ... ,',X_features_inference_time_utc[-1][0])        

x_ax = range(len(fc_infer_1))
#plt.scatter(x_ax, y_predicted, s=5, color="blue", label="original")
plt.plot(x_ax, fc_infer_1, lw=0.8, color="green", label="predicted")
plt.legend()    
plt.show() 
print(X_features_inference_time_utc[0][0],', ... ,',X_features_inference_time_utc[-1][0])    
print('fc_infer_1: ', M,len(fc_infer_with_time),prediction1['startTimestamp'],      datetime.fromtimestamp(prediction1['startTimestamp']))
return prediction1,M


# In[23]:


# start block123, bit10,ds2;s1at1#-10800;###ghi_backward, 10-01T08z - 10-05T19Z,
u1 = 1000000
t0 = time.time()
cod = "***_138_5a_0012_11_16***"
adr_ = "_138_5a_0012__11_16_SVM_"
adr = "f" + adr_ + "1.json"
j = 0
k = 1
v = 5
x = '***&&&^^^'
b = 1
d = 5
loss_a = 0
loss_c = []
model_adressis = ["m_138_1_0012__11_16_SVM_.joblib"]
n = [1]
b1 = len(n)
for ii in range(1):  
    move = ii
    targets1,features,features_time_utc, y_targets, y_targets_time, X_features_inference,            X_features_inference_time_utc,keys_weatherbit,filtr2 =                 ds_for_solarRad_hourly_forecast_model_from_weathebit3_e_prod_1_n(move,j,k,v,d,n[0])
    fc_np_m,model_opt1, X_features_inference_time_utc = fitting_n_opt_comb(j,k,v,b,d,move,n,                                                                           adr,model_adressis)
t_fit = time.time() - t0
print('t_fit****',t_fit)     
    
    


# In[25]:


def predictor():
    u1 = 1000000
    t0 = time.time()
    cod = "***_138_5a_0012_11_16***"
    adr_ = "_138_5a_0012__11_16_SVM_"
    adr = "f" + adr_ + "1.json"
    j = 0
    k = 0
    v = 5
    x = '***&&&^^^'
    b = 1
    d = 5
    loss_a = 0
    loss_c = []
    model_adressis = ["m_138_1_0012__11_16_SVM_.joblib"]

    n = [1]
    b1 = len(n)
    ii =1
    move = 1
    i = 0
    n1=1
    f = open(adr, "r")
    #predicted_before_2 = [value, targets1[-1][1],0,1,[]]
    predicted_before_2 = json.loads(f.read())
    f.close()
    if len( predicted_before_2)>0:
        fc = predicted_before_2[0]
        print('fc',fc)
        #forecast,stat,whole_loss,model_opt,n_0,r_0,l_0,whole_loss_,r_hour_err_for_5_values_more_0_ =\
        predict_c_n_comb(cod,ii,move,j,v,b1,k,d,n,x,adr,model_adressis,fc,'model_opt1')
        """
        f = open("comb" + adr_+".json", "a")   
        y = json.dumps(str(stat))
        f.write(y) 
        f.close()          
        print('##########################stat - ii,j,k,v,b,d,move:---',ii,i,j,k,v,b,d,move,\
              "_____coord:_____",stat)
        f = open("comb_stat" + adr_+".json", "a")
        y = json.dumps(stat)
        f.write(y) 
        f.close()
        """
    t_fit = time.time() - t0
    print('t_fit****',t_fit)


def setInterval(func,time):
    e = threading.Event()
    while not e.wait(time):
        func()


# In[ ]:


t2 = setInterval(predictor, 3600)


# In[ ]:




