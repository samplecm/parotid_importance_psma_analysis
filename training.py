#this code was used for model training as in the article 

"PSMA PET as a predictive tool for sub-regional importance estimates in the parotid gland"

#By Caleb Sample, Arman Rahmim, Francois Benard, Jonn Wu, Haley Clark.



import six 
import sys
sys.modules['sklearn.externals.six'] = six
import Visuals_Subsegs
import numpy as np
import copy
import os
import feature_selection
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn import metrics, svm
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from feature_selection import lincom, pairwise_correlation_filter, pca, use_n_most_correlated, sort_by_correlation
import matplotlib.pyplot as plt
import extract_radiomics
import csv
import itertools
import time
import pickle
def warn(*args, **kwargs):
    pass
import warnings 
warnings.warn = warn     #suppress sklearn warnings
        

def main(deblurred=False):
    #define models and hyperparameters
    models = {
        'lin_reg': {
            'model': LinearRegression(),
            'params':
            {}
        },
    

        'svm': {
            'model': svm.SVR(),
            'params': {
                'epsilon': [0.01, 0.05, 0.1],
                'kernel': ['linear', 'rbf', 'sigmoid', 'poly'], 
                'max_iter': [1000],
                'degree': [2,3],
                'gamma': ['scale','auto'],
                'C': [0.1],
                'coef0': [-0.3,-0.2,-0.1, 0]

    
            }  
                    
        },
        
        'rf': {
            'model': RandomForestRegressor(),
            'params': {
                'n_estimators': [5,10,15,20],
                'max_depth': [3, 5,8 ,None],
                'criterion': ["absolute_error", "squared_error"]
            }
        },
        
        'cit': {
            'model': DecisionTreeRegressor(),
            'params': {
                'max_depth': [10,15, 20, 25, None],
                'criterion': ["absolute_error", "squared_error"]
            }
        },
        'kr': {
           'model': KernelRidge(),
           'params': {
               'alpha': [0.001, 0.01, 0.1, 0.5, 1, 5, 10],
               'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],     
               'degree': [2,3,4],
               'coef0': [-1,0,1]     
           },    
        }
    }

    #do cross validation separately for the different pet scalings (compare results after)

    #make results folder if dne
    results_folder = os.path.join(os.getcwd(), "test_results")
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    for suv_type in ["bw"]:
        print(f"Training model for suv type: {suv_type}")
        print(f"Loading design matrix... deblurred = {deblurred}")

        #this function loads the design matrix using radiomics features which have been extracted for each sub-region in each parotid of the gland.
        #if repeating this study, format your design matrix such that each row is a specific sub-region for a specific parotid gland, and each column is a different radiomic feature value. 
        data = extract_radiomics.load_design_matrix(try_load=False, normalize_by_whole=True, suv_type=suv_type, deblurred=deblurred)
        
        x_all = data["all_spatial"][:,:-1]
        y_all = data["all_spatial"][:,-1]
        feature_names = data["feature_names_spatial_all"]

        # x_all = data["all_w_attrs"][:,:-1]
        # y_all = data["all_w_attrs"][:,-1]
        # feature_names = data["feature_names_all_w_attrs"]

        # #make sure not more columns than rows
        # if x_all.shape[0] < x_all.shape[1]:
        #     x_all = x_all[:,x_all.shape[0]-1]
        #     feature_names = feature_names[:,x_all.shape[0]-1]
        #split data into training, test
        #x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.1, random_state=1, shuffle=True)


    #Loop through to add tuples for hidden layers sizes param
    #cross_val(x_train, y_train, models, feature_names)

        
        #results = load_cross_val_results(x_all, y_all, models, feature_names, suv=suv_type, deblurred=deblurred)
        results = cross_val_population(x_all, y_all, models, feature_names, suv=suv_type, deblurred=deblurred)

        
        #make a csv out of test results
        csv_file_name = os.path.join(os.getcwd(),"importance_comparison", "test_results_" + suv_type + f"deblurred_{deblurred}.csv")

        #collect stats of mae
        with open(csv_file_name, mode='w', newline='') as fp:
            writer = csv.writer(fp)
            headers = ["Fold"]    
            for key in results[0]:   #get header names
                headers.append(key)
            writer.writerow(headers)
            for fold in results:
                fold_results = [fold]
                for result_key in results[fold]:
                    fold_results.append(str(results[fold][result_key]))
                writer.writerow(fold_results)

    return


def get_subseg_avgs(x, y, names, combine_l_r=True):
    #every 60 points is for a certain importance region, with alternating between left and right
    #returns all features avgs and another matrix horizontally stacked which is the std of the features
    if combine_l_r == False:
        avgs = np.zeros((36, x.shape[1]))
        stds = np.zeros((36, x.shape[1]))
        y_new = np.zeros(36)
        for i in range(18):
            l_vals = []
            r_vals = []
            y_vals = []
            for j in range(60):
                val = x[j+i*60,:]
                y_vals.append(y[j+i*60])
                if j % 2 == 0:
                    l_vals.append(val)
                else:
                    r_vals.append(val)
            r_vals = np.array(r_vals)
            l_vals = np.array(l_vals)
            std = (np.std(y_vals))
            if np.std(y_vals) > 1e-10:
                raise Exception("Expected to be getting all points for one importance region but got multiple importances.")   
            y_val = y_vals[0]
            r_avg = np.mean(r_vals, axis=0)       
            r_std = np.std(r_vals, axis=0)

            l_avg = np.mean(l_vals, axis=0)       
            l_std = np.std(l_vals, axis=0)

            avgs[2*i,:] = l_avg
            avgs[2*i+1,:] = r_avg

            stds[2*i,:] = l_std
            stds[2*i+1,:] = r_std

            y_new[2*i:2*i+2] = y_val
        all_data = avgs
        all_names = names
        #all_data = np.hstack((avgs, stds))    
        all_names = copy.deepcopy(names)
        # for name in names:
        #     all_names.append(str(name + "_std"))
        return all_data, y_new, all_names
    else:
        avgs = np.zeros((18, x.shape[1]))
        stds = np.zeros((18, x.shape[1]))
        y_new = np.zeros(18)
        for i in range(18):
            vals = []
            y_vals = []
            for j in range(60):
                val = x[j+i*60,:]
                y_vals.append(y[j+i*60])
                vals.append(val)
               
            vals = np.array(vals)
            
            std = (np.std(y_vals))
            if np.std(y_vals) > 1e-10:
                raise Exception("Expected to be getting all points for one importance region but got multiple importances.")   
            y_val = y_vals[0]
            avg = np.mean(vals, axis=0)       
            std = np.std(vals, axis=0)

            avgs[i,:] = avg
            stds[i,:] = std
            y_new[i] = y_val
        all_data = avgs
        all_names = names
        # all_data = np.hstack((avgs, stds))    
        # all_names = copy.deepcopy(names)
        # for name in names:
        #     all_names.append(str(name + "_std"))
        return all_data, y_new, all_names

def cross_val_population(x_all, y_all, models, feature_names, suv="", deblurred=False):

    x, y, feature_names = get_subseg_avgs(x_all, y_all, feature_names, combine_l_r=False)   #get population averages for each subseg 
    x, feature_names, [x_all] = sort_by_correlation(x, y, feature_names, corr_type="spearman", x_others=[x_all])
    test_results = {}
    kf_outer = KFold(n_splits=9, shuffle=True, random_state=2)
    
    fold_outer = 0
    
    for outer_idx, test_idx in kf_outer.split(x):
        results = {}   #key for each inner fold, holds top 3 best performers for each in nested list (see inner loop end for formatting) 
        x_outer, x_test = x[outer_idx], x[test_idx]
        y_outer, y_test = y[outer_idx], y[test_idx]
        #perform inside loop for tuning hyperparameters 
        kf_inner = KFold(n_splits=8, shuffle=True, random_state=2) #will do an outside cross-validation test set to evaluate model performance
        
        fold = 0    #iterate after each fold
        x_all_outer = copy.deepcopy(x_all)
        for ti in reversed(list(test_idx)):
            x_all_outer = np.delete(x_all_outer, range(ti*30 , ti*30+30), axis=0)
        
        for train_idx, val_idxs in kf_inner.split(x_outer):    
            inner_fold_start_time = time.time()
            #get the design matrix of non averaged features to use with certain feature selection methods
            
            

            fold_results = [] #holds results of every combo tested in fold

            x_train, x_val = x_outer[train_idx], x_outer[val_idxs]
            y_train, y_val = y_outer[train_idx], y_outer[val_idxs]

            x_all_inner = copy.deepcopy(x_all_outer)
            for val_idx in reversed(list(val_idxs)):
                x_all_inner = np.delete(x_all_inner, range(val_idx*30 , val_idx*30+30), axis=0)
            #sort columns by population correlation with importance
            x_train, feature_names, [x_val, x_full, x_all_inner, x_outer_model_cv] = sort_by_correlation(x_train, y_train, feature_names, corr_type="spearman", x_others=[x_val, x, x_all_inner, copy.deepcopy(x_outer)])
            
            for normalize in [True]:    #normalize data by statistical z value or not
                # if normalize == True:
                scaler = StandardScaler()
                scaler.fit(x_full)
                x_train = scaler.transform(x_train)
                x_val = scaler.transform(x_val)
                x_full = scaler.transform(x_full)
                x_all_inner = scaler.transform(x_all_inner)
                x_outer_model_cv = scaler.transform(x_outer_model_cv)
                #     x_scaled = scaler.transform(x_outer) #normalize by statistical z value
                #     x_train, x_val = x_scaled[train_idx], x_scaled[val_idxs]
                #     y_train, y_val = y_outer[train_idx], y_outer[val_idxs]
                # else:
                #     x_train, x_val = x_outer[train_idx], x_outer[val_idxs]
                #     y_train, y_val = y_outer[train_idx], y_outer[val_idxs]
                
                for fs_method in [pca, lincom, use_n_most_correlated]:    #loop through different fs methods, #taking out lincom for pop data (m < n)  #took out pairwise correlation filter
                    fs_method_name =fs_method.__name__

                    # if fs_method_name == "lincom":
                    #     fs_cutoffs = [0.01, 0.03,0.05,0.1, 0.2,0.3,0.4]   #different cutoffs used in methods
                    if fs_method_name == "use_n_most_correlated":   
                        fs_cutoffs = [[6,8], [0.9, 0.92]]   #first list is n (number of features to return) and second is cutoff for pairwise corr filter applied first
                        fs_cutoffs = list(itertools.product(fs_cutoffs[0], fs_cutoffs[1]))
                    elif fs_method_name == "pca":
                        fs_cutoffs = [5,6,8,10,12,15,20,25,30]#[0.1, 0.15, 0.2,0.4,0.5, 0.6,0.7, 0.8, 0.9,0.95]    
                    elif fs_method_name == "lincom":
                        fs_cutoffs = [0.05, 0.1, 0.2, 0.3]    
             
                    for cutoff in fs_cutoffs:    

                        #print progress
                        print(f"Outer Fold: {fold_outer+1} | Inner Fold: {fold+1} | Normalize: {normalize} | FS Method: {fs_method_name} | Cutoff: {cutoff}")
                        maes = []
                        # x_train, feature_names = feature_selection.sort_by_correlation(x_train, y_train, feature_names, corr_type="spearman")
                        # for i in range(3):
                        #     Visuals_Subsegs.importance_vs_feature(x_train[:,i],y_train, feature_names[i])
                        if fs_method_name == "use_n_most_correlated":
                            x_fs, feature_names_fs, deleted_features_fs, [x_val_fs, x_full_fs, x_outer_model_cv_fs] = fs_method(x_train, y_train, feature_names, n=cutoff[0], cutoff=cutoff[1], x_others=[x_val, x_full, x_outer_model_cv])
                        elif fs_method_name == "pairwise_correlation_filter":
                            x_fs, feature_names_fs, deleted_features_fs, [x_val_fs, x_full_fs, x_outer_model_cv_fs] = fs_method(x_train, feature_names, x_others=[x_val,x_full, x_outer_model_cv], cutoff=cutoff)     
                        elif fs_method_name == "pca":
                            _, feature_names_fs, deleted_features_fs, [x_fs, x_val_fs, x_full_fs, x_outer_model_cv_fs] = fs_method(x_all_inner, feature_names, x_others=[x_train, x_val, x_full, x_outer_model_cv], cutoff=cutoff)
                            scaler = StandardScaler()   #pca doesnt have normalized final features, so normalize again
                            scaler.fit(x_full_fs)
                            #x_full_fs = scaler.transform(x_full_fs)
                            x_fs = scaler.transform(x_fs) #normalize by statistical z value
                            x_val_fs = scaler.transform(x_val_fs) #normalize by statistical z value
                            x_outer_model_cv_fs = scaler.transform(x_outer_model_cv_fs)
                        else:
                            _, feature_names_fs, deleted_features_fs, [x_fs, x_val_fs, x_full_fs] = fs_method(x_all_inner, feature_names, x_others=[x_train, x_val, x_full], cutoff=cutoff, scale_data=False)    
                        #plot first few features
                        #x_fs, feature_names_fs = feature_selection.sort_by_correlation(x_fs, y_train, feature_names)

                            



                        for model_name, model_dict in models.items():
                            model = model_dict['model']
                            params = model_dict['params']
                            best_params, best_score = nested_cross_val(model, params, x_outer_model_cv_fs, y_outer)
                            
                            model.set_params(**best_params)    #define the model using best params from nested model cv
                            #now need to train model with these parameters on x_train and evaluate on x_val
                            model = model.fit(x_fs, y_train)
                            y_pred = model.predict(x_val_fs)
                            y_pred = np.clip(y_pred, 0,1)
                            mae = metrics.mean_absolute_error(y_val, y_pred)
                            maes.append(mae)
                            #print(mae)
                            results_list_part1 = [mae, model_name, fs_method.__name__, cutoff, normalize]           
 

                            results_list = []
                            results_list.append(results_list_part1)
                            results_list.append(best_params)
                            fold_results.append(results_list)
                            
                            
                            #test
                            # fold_results.sort(key=lambda x: x[0][0])  #first sort the retults by mae        
                            # results[str(fold)] = fold_results      
                            # fold+=1 

                            # #now need to get the best performing parameters from the cross validation (mode) and use to build our final model.     
                            # fold_outer += 1    

                            # get_final_best_params(copy.deepcopy(results))

                        print(f"avg mae: {np.min(maes)}")

            #now want to return the top 3 best performing (lowest mae) hyperparameters for each fold
            fold_results.sort(key=lambda x: x[0][0])  #first sort the retults by mae        
            results[str(fold)] = fold_results 
            fold_time = (time.time() - inner_fold_start_time) / 60     
            fold+=1 
            print(f"Finished inner fold: {fold} of outer fold {fold_outer} in {round(fold_time,1)} minutes")
            # if fold == 2:
            #     break
 
        #save results dict
        fold_results_dir = os.path.join(os.getcwd(),"importance_comparison", f"fold_results_{suv}_deblurred_{deblurred}")
        if not os.path.exists(fold_results_dir):
            os.mkdir(fold_results_dir)
        fold_results_path = os.path.join(fold_results_dir, str(fold_outer))
        with open(fold_results_path, "wb") as fp:
            pickle.dump(results, fp)    

        best_model, best_fs_name, best_cutoff, best_norm  = get_final_best_params(copy.deepcopy(results))

        x_outer, feature_names, [x_test, x_full, x_all_outer] = sort_by_correlation(x_outer, y_outer, feature_names, corr_type="spearman", x_others=[x_test, x, x_all_outer])

        # # if best_norm == True:
        # scaler = StandardScaler()
        # scaler.fit(x_full)
        # x_outer = scaler.transform(x_outer)
        # x_test = scaler.transform(x_test)
        # x_full = scaler.transform(x_full)
        # x_all_outer = scaler.transform(x_all_outer)

        for fs_method in [pca, lincom, use_n_most_correlated]:    #loop through different fs methods #taking out lincom for pop data (m < n)
            if best_fs_name == fs_method.__name__:
                best_fs = fs_method

        if best_fs_name == "use_n_most_correlated":
            x_fs, feature_names_fs, deleted_features_fs, [x_test_fs, x_full_fs] = best_fs(x_outer, y_outer, feature_names, n=best_cutoff[0], cutoff=best_cutoff[1], x_others=[x_test, x_full])
        elif best_fs_name == "pairwise_correlation_filter":
            x_fs, feature_names_fs, deleted_features_fs, [x_test_fs, x_full_fs] = best_fs(x_outer, feature_names, x_others=[x_test, x_full], cutoff=best_cutoff) 
        elif best_fs_name == "pca":
            _, feature_names_fs, deleted_features_fs, [x_fs, x_test_fs, x_full_fs] = best_fs(x_all_outer, feature_names, x_others=[x_outer, x_test, x_full], cutoff=best_cutoff)
            # scaler = StandardScaler()   #pca doesnt have normalized final features, so normalize again
            # scaler.fit(x_full_fs)
            # #x_full_fs = scaler.transform(x_full_fs)
            # x_fs = scaler.transform(x_fs) #normalize by statistical z value
            # x_test_fs = scaler.transform(x_test_fs) #normalize by statistical z value
        
        else:
            _, feature_names_fs, deleted_features_fs, [x_fs, x_test_fs, x_full_fs] = best_fs(x_all_outer, feature_names, x_others=[x_outer, x_test, x_full], cutoff=best_cutoff, scale_data=best_norm)        

        #now need to create the model with these params, normalize if needed and do feature selection. 
        # if best_norm == True:
        scaler = StandardScaler()
        scaler.fit(x_full_fs)
        x_fs = scaler.transform(x_fs) #normalize by statistical z value
        x_test_fs = scaler.transform(x_test_fs)


        model = models[best_model]['model']  

        params = models[best_model]['params']
        
        best_params, _ = nested_cross_val(model, params, x_full_fs, y, folds=9)

        model.set_params(**best_params)    #define the model using best params from nested model cv
        #now need to train model with these parameters on x_train and evaluate on x_val
        model = model.fit(x_fs, y_outer)
        y_pred = model.predict(x_test_fs)
        y_pred = np.clip(y_pred, 0,1)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        print(f"mae: {mae}")
        #mse = metrics.mean_squared_error(y_val, y_pred)
        #r2 = metrics.r2_score(y_val, y_pred)
        test_results[fold_outer] = {}
        test_results[fold_outer]['mae'] = mae
        test_results[fold_outer]['imp real'] = y_test
        test_results[fold_outer]['imp pred'] = y_pred
        test_results[fold_outer]["best_model"] = best_model
        test_results[fold_outer]["best_fs"] = best_fs_name
        test_results[fold_outer]["best_norm"] = best_norm
        test_results[fold_outer]["best_cutoff"] = best_cutoff
        test_results[fold_outer]["feature_names"] = feature_names_fs
        
        fold_outer += 1 

    with open(os.path.join(fold_results_dir, "test_results"), "wb") as fp:
        pickle.dump(test_results, fp)
    return test_results

def load_cross_val_results(x_all, y_all, models, feature_names, suv="", deblurred=False):

    x, y, feature_names = get_subseg_avgs(x_all, y_all, feature_names, combine_l_r=False)   #get population averages for each subseg 
    x, feature_names, [x_all] = sort_by_correlation(x, y, feature_names, corr_type="spearman", x_others=[x_all])
    test_results = {}
    kf_outer = KFold(n_splits=9, shuffle=True, random_state=2)
    
    fold_outer = 0
    
    for outer_idx, test_idx in kf_outer.split(x):
        results = {}   #key for each inner fold, holds top 3 best performers for each in nested list (see inner loop end for formatting) 
        x_outer, x_test = x[outer_idx], x[test_idx]
        y_outer, y_test = y[outer_idx], y[test_idx]
        #perform inside loop for tuning hyperparameters 
        
        fold = 0    #iterate after each fold
        x_all_outer = copy.deepcopy(x_all)
        for ti in reversed(list(test_idx)):
            x_all_outer = np.delete(x_all_outer, range(ti*30 , ti*30+30), axis=0)
        
        

        fold_results_dir = os.path.join(os.getcwd(),"importance_comparison", f"fold_results_{suv}_deblurred_{deblurred}")
        if not os.path.exists(fold_results_dir):
            os.mkdir(fold_results_dir)
        fold_results_path = os.path.join(fold_results_dir, str(fold))
        with open(fold_results_path, "rb") as fp:
            results = pickle.load(fp)    

        best_model, best_fs_name, best_cutoff, best_norm  = get_final_best_params(copy.deepcopy(results))

        x_outer, feature_names, [x_test, x_full, x_all_outer] = sort_by_correlation(x_outer, y_outer, feature_names, corr_type="spearman", x_others=[x_test, x, x_all_outer])

        # scaler = StandardScaler()
        # scaler.fit(x_full)
        # x_outer = scaler.transform(x_outer)
        # x_test = scaler.transform(x_test)
        # x_full = scaler.transform(x_full)
        # x_all_outer = scaler.transform(x_all_outer)
        for fs_method in [pca, lincom, use_n_most_correlated]:    #loop through different fs methods #taking out lincom for pop data (m < n)
            if best_fs_name == fs_method.__name__:
                best_fs = fs_method

        if best_fs_name == "use_n_most_correlated":
            x_fs, feature_names_fs, deleted_features_fs, [x_test_fs, x_full_fs] = best_fs(x_outer, y_outer, feature_names, n=best_cutoff[0], cutoff=best_cutoff[1], x_others=[x_test, x_full])
        elif best_fs_name == "pairwise_correlation_filter":
            x_fs, feature_names_fs, deleted_features_fs, [x_test_fs, x_full_fs] = best_fs(x_outer, feature_names, x_others=[x_test, x_full], cutoff=best_cutoff) 
        elif best_fs_name == "pca":
            _, feature_names_fs, deleted_features_fs, [x_fs, x_test_fs, x_full_fs] = best_fs(x_all_outer, feature_names, x_others=[x_outer, x_test, x_full], cutoff=best_cutoff)
            # scaler = StandardScaler()   #pca doesnt have normalized final features, so normalize again
            # scaler.fit(x_full_fs)
            # #x_full_fs = scaler.transform(x_full_fs)
            # x_fs = scaler.transform(x_fs) #normalize by statistical z value
            # x_test_fs = scaler.transform(x_test_fs) #normalize by statistical z value
        
        else:
            _, feature_names_fs, deleted_features_fs, [x_fs, x_test_fs, x_full_fs] = best_fs(x_all_outer, feature_names, x_others=[x_outer, x_test, x_full], cutoff=best_cutoff, scale_data=best_norm)        

        #now need to create the model with these params, normalize if needed and do feature selection. 
        # if best_norm == True:
        scaler = StandardScaler()
        scaler.fit(x_full_fs)
        x_fs = scaler.transform(x_fs) #normalize by statistical z value
        x_test_fs = scaler.transform(x_test_fs)


        model = models[best_model]['model']  

        params = models[best_model]['params']
        
        best_params, _ = nested_cross_val(model, params, x_full_fs, y)

        model.set_params(**best_params)    #define the model using best params from nested model cv
        #now need to train model with these parameters on x_train and evaluate on x_val
        model = model.fit(x_fs, y_outer)
        y_pred = model.predict(x_test_fs)
        y_pred = np.clip(y_pred, 0,1)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        print(f"mae: {mae}")
        #mse = metrics.mean_squared_error(y_val, y_pred)
        #r2 = metrics.r2_score(y_val, y_pred)
        test_results[fold_outer] = {}
        test_results[fold_outer]['mae'] = mae
        test_results[fold_outer]['imp real'] = y_test
        test_results[fold_outer]['imp pred'] = y_pred
        test_results[fold_outer]["best_model"] = best_model
        test_results[fold_outer]["best_fs"] = best_fs_name
        test_results[fold_outer]["best_norm"] = best_norm
        test_results[fold_outer]["best_cutoff"] = best_cutoff
        test_results[fold_outer]["feature_names"] = feature_names_fs
        
        fold_outer += 1 

    with open(os.path.join(fold_results_dir, "test_results"), "wb") as fp:
        pickle.dump(test_results, fp)
    return test_results

def get_final_best_params_old(results):
    results = copy.deepcopy(results)
    #first get model type
    fs_type_counts = {}
    for fold in results:
        for i in range(3):
            fs_name = results[fold][i][0][2]
            if fs_name in fs_type_counts:
                fs_type_counts[fs_name] +=1
            else:
                fs_type_counts[fs_name] = 1
    max_count = max(fs_type_counts.values())       
    most_occurring_fs = [fs_name for fs_name, count in fs_type_counts.items() if count == max_count]     

    best_fs = most_occurring_fs[0]

    #get rid of all the items that don't include the best FS (nested analysis)
    for fold in results:
        for i, item in reversed(list(enumerate(results[fold]))):
            if item[0][2] != best_fs:
                del results[fold][i]  

    #now do the same for the cutoff:
    cutoff_type_counts = {}
    for fold in results:
        cutoff_name = results[fold][0][0][3]
        if cutoff_name in cutoff_type_counts:
            cutoff_type_counts[cutoff_name] +=1
        else:
            cutoff_type_counts[cutoff_name] = 1
    max_count = max(cutoff_type_counts.values())       
    most_occurring_cutoff = [cutoff_name for cutoff_name, count in cutoff_type_counts.items() if count == max_count]     

    best_cutoff = most_occurring_cutoff[0]

    
    #get rid of all the items that don't include the best cutoff (nested analysis)
    for fold in results:
        for i, item in reversed(list(enumerate(results[fold]))):
            if item[0][3] != best_cutoff:
                del results[fold][i]   

    #now do the same for the normalization:
    norm_type_counts = {}
    for fold in results:
        norm_name = results[fold][0][0][4]
        if norm_name in norm_type_counts:
            norm_type_counts[norm_name] +=1
        else:
            norm_type_counts[norm_name] = 1
    max_count = max(norm_type_counts.values())       
    most_occurring_norm = [norm_name for norm_name, count in norm_type_counts.items() if count == max_count]     


    best_norm = most_occurring_norm[0]

    
    for fold in results:
        for i, item in reversed(list(enumerate(results[fold]))):
            if item[0][4] != best_norm:
                del results[fold][i]

    model_type_counts = {}
    for fold in results:
        for i in range(3): #consider top 3 performers in each fold
            model_name = results[fold][i][0][1] #list order [mae, model_name, fs_method.__name__, cutoff, normalize]
            if model_name in model_type_counts:
                model_type_counts[model_name] +=1
            else:
                model_type_counts[model_name] = 1
    max_count = max(model_type_counts.values())       
    most_occurring_models = [model_name for model_name, count in model_type_counts.items() if count == max_count]  #this will either add only winner or multiple if its a tie   
    best_model = most_occurring_models[0]
    # #            
    #now get the best feature selection method
    #
    # #first get rid of all the items that don't include the best model (nested analysis)
    # for fold in results:
    #     for i, item in reversed(list(enumerate(results[fold]))):
    #         if item[0][1] != best_model:
    #             del results[fold][i]    

    

    return [best_model, best_fs, best_cutoff, best_norm]     
            # results_list_part1 = [mae, model_name, fs_method.__name__, cutoff, normalize]
                            
            #                 results_list_part2 = []
            #                 # for param in best_params:
            #                 #     results_list_part2.append([param,best_params[param]])

            #                 results_list_part3 = feature_names_fs   

            #                 results_list = []
            #                 results_list.append(results_list_part1)
            #                 results_list.append(results_list_part2)   #best params 
            #                 results_list.append(results_list_part3)   #feature names

            #                 fold_results.append(results_list)
def get_fs_and_model_performances(old_results):
    best_data = {'rf': [], 'kr': [], 'svm': [], 'lin_reg': [], "cit": [], 'use_n_most_correlated': [], 'pca': [], 'lincom': []}
    old_results = copy.deepcopy(old_results)
    results = {}
    for fold in old_results:
        results[fold] = {}
        for val in old_results[fold]:
            results[fold][str(val[0][1:])] = val
    mae_avgs = {}
    for result in results['0']: #for all combos of model, feature, cutoff..
        mae_avgs[result] = [results['0'][result][0][0]] #add mae
        for fold in list(results.keys())[1:]:
            mae_avgs[result].append(results[fold][result][0][0]) #add mae for all other folds
    for combo in mae_avgs:
        mae_avgs[combo] = np.mean(mae_avgs[combo])
    #sort by mae 
    mae_avgs = dict(sorted(mae_avgs.items(), key=lambda item: item[1]))
    for val in mae_avgs:
        for key in best_data:
            if key in val and best_data[key] == []:
                best_data[key] = mae_avgs[val]
        all_full = True
        for key in best_data:
            if best_data[key] == []:
                all_full = False
        if all_full == True:
            return best_data


    


def get_final_best_params(old_results):
    old_results = copy.deepcopy(old_results)
    results = {}
    for fold in old_results:
        results[fold] = {}
        for val in old_results[fold]:
            results[fold][str(val[0][1:])] = val
    mae_avgs = {}
    for result in results['0']: #for all combos of model, feature, cutoff..
        mae_avgs[result] = [results['0'][result][0][0]] #add mae
        for fold in list(results.keys())[1:]:
            mae_avgs[result].append(results[fold][result][0][0]) #add mae for all other folds
    for combo in mae_avgs:
        mae_avgs[combo] = np.mean(mae_avgs[combo])
    #sort by mae 
    mae_avgs = dict(sorted(mae_avgs.items(), key=lambda item: item[1]))
    best_model, best_fs_name, best_cutoff, best_norm = eval(next(iter(mae_avgs))) #first element, highest mae
    return best_model, best_fs_name, best_cutoff, best_norm





def nested_cross_val(model, params, x, y, folds=8):
    kf = KFold(n_splits=folds, shuffle=True, random_state=1)
    gs = GridSearchCV(model, param_grid=params, cv=kf, scoring="neg_mean_absolute_error", error_score=np.nan)
    gs.fit(x,y)
    return gs.best_params_, gs.best_score_

def make_perturbation_plot():

    models = {
        'lin_reg': {
            'model': LinearRegression(),
            'params':
            {}
        },
    

        'svm': {
            'model': svm.SVR(),
            'params': {
                'epsilon': [0.01, 0.05, 0.1,],
                'kernel': ['linear', 'rbf', 'sigmoid', 'poly'], 
                'max_iter': [1000],
                'degree': [2,3],
                'gamma': ['scale','auto'],
                'C': [0.1],
                'coef0': [-0.3,-0.2,-0.1, 0]

    
            }  
                    
        },
        
        'rf': {
            'model': RandomForestRegressor(),
            'params': {
                'n_estimators': [3, 5, 7, 10],
                'max_depth': [3, 5,8 ,None],
                'criterion': ["absolute_error", "squared_error"]
            }
        },
        
        'cit': {
            'model': DecisionTreeRegressor(),
            'params': {
                'max_depth': [5,10,15, 20, 25, None],
                'criterion': ["absolute_error", "squared_error"]
            }
        },
        'kr': {
           'model': KernelRidge(),
           'params': {
               'alpha': [0.1, 0.5,1,2],
               'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],     
               'degree': [2,3,4],
               'coef0': [-1,0,1]  
           },    
        }

    }
    #def load_cross_val_results(x_all, y_all, models, feature_names, suv="", deblurred=False):
    print(f"Loading design matrix...")
    data = extract_radiomics.load_design_matrix(try_load=False, normalize_by_whole=True, suv_type="bw", deblurred=True)
    
    x_all = data["all_spatial"][:,:-1]
    y_all = data["all_spatial"][:,-1]
    feature_names = data["feature_names_spatial_all"]

    try:
        2/0
        with open(os.path.join(os.getcwd(), "importance_comparison", "mae_preds.txt"), "rb") as fp:
            [model_mae, preds, mae_preds,y] = pickle.load(fp)
    except:
            
        #first estimate the error by making a LOO  error prediction model
        x, y, feature_names = get_subseg_avgs(x_all, y_all, feature_names, combine_l_r=False)   #get population averages for each subseg 
        x, feature_names, [x_all] = sort_by_correlation(x, y, feature_names, corr_type="spearman", x_others=[x_all])    
        kf_outer = KFold(n_splits=36, shuffle=True, random_state=2)
        
        #make mae list to be the "y" for new error prediction model
        maes = np.zeros((36))
        preds = np.zeros((36))
        for outer_idx, test_idx in kf_outer.split(x):
            x_outer, x_test = x[outer_idx], x[test_idx]
            y_outer, y_test = y[outer_idx], y[test_idx]
            x_all_outer = copy.deepcopy(x_all)
            for ti in reversed(list(test_idx)):
                x_all_outer = np.delete(x_all_outer, range(ti*30 , ti*30+30), axis=0)


            x_outer, feature_names, [x_all_outer, x_full, x_test] = sort_by_correlation(x_outer, y_outer, feature_names, corr_type="spearman", x_others=[x_all_outer, x, x_test])

            _, _,_, [x_fs,x_test_fs, x_full_fs] = pca(x_all_outer, feature_names, x_others=[x_outer, x_test, x_full], cutoff=20, scale_data=True)

            scaler = StandardScaler()
            scaler.fit(x_full_fs)
            x_full_fs = scaler.transform(x_full_fs)
            x_fs = scaler.transform(x_fs)
            x_test_fs = scaler.transform(x_test_fs)


            model = models['kr']['model']  
            params = models['kr']['params']

            best_params, _ = nested_cross_val(model, params, x_full_fs, y)

            model.set_params(**best_params)    #define the model using best params from nested model cv
            #now need to train model with these parameters on x_train and evaluate on x_val
            model = model.fit(x_fs, y_outer)

            y_pred = model.predict(x_test_fs)
            y_pred = np.clip(y_pred, 0,1)
            mae = metrics.mean_absolute_error(y_test, y_pred)
            preds[test_idx] = y_pred[0]
            maes[test_idx] = mae

        #now retrain a model predicting the absolute error of each prediction.
        mae_preds = np.zeros((36))
        #mae preds will contain the predicted absolute error of the prediction corresponding to the y value in.
        for outer_idx, test_idx in kf_outer.split(x):
            x_outer, x_test = x[outer_idx], x[test_idx]
            mae_outer, mae_test = maes[outer_idx], maes[test_idx]

            x_outer, feature_names, [x_all, x_test, x_full] = sort_by_correlation(x_outer, mae_outer, feature_names, corr_type="spearman", x_others=[x_all, x_test, x])

            _, _,_, [x_fs,x_test_fs, x_full_fs] = pca(x_all, feature_names, x_others=[x_outer, x_test, x_full], cutoff=20, scale_data=True)

            scaler = StandardScaler()
            scaler.fit(x_full_fs)

            x_fs = scaler.transform(x_fs)
            x_test_fs = scaler.transform(x_test_fs)


            model_mae = models['kr']['model']  
            params = models['kr']['params']

            best_params, _ = nested_cross_val(model_mae, params, x_fs, y_outer)

            model_mae.set_params(**best_params)    #define the model using best params from nested model cv
            #now need to train model with these parameters on x_train and evaluate on x_val
            model_mae = model_mae.fit(x_fs, mae_outer)

            y_pred = model_mae.predict(x_test_fs)
            #mae = metrics.mean_absolute_error(y_pred, mae_test)
            mae_preds[test_idx] = np.clip(y_pred, 0,1)[0]
        with open(os.path.join(os.getcwd(), "importance_comparison", "mae_preds.txt"), "wb") as fp:
            pickle.dump([model_mae, preds, mae_preds, y], fp)




    preds = list(preds)
    mae_preds = list(mae_preds)
    fig, ax = plt.subplots(figsize=(15,15))
    ax.errorbar(list(y), preds, yerr=mae_preds, fmt='o', linestyle='none', color='orange')
    ax.plot(np.linspace(0,1,1000), np.linspace(0,1,1000), color='r')
    ax.set_xlabel("Importance", fontsize=18)
    ax.set_ylabel("Importance Prediction", fontsize=18)
    ax.tick_params(axis='both', which='both', labelsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show(block=True)



    x, y, feature_names = get_subseg_avgs(x_all, y_all, feature_names, combine_l_r=False)   #get population averages for each subseg 

    x, feature_names, [x_all] = sort_by_correlation(x, y, feature_names, corr_type="spearman", x_others=[x_all])    
    #now train a model on
    x_outer = x
    y = y
    x_outer, feature_names, [x_all] = sort_by_correlation(x_outer, y, feature_names, corr_type="spearman", x_others=[x_all])


    x_all, _,_, [x_fs] = pca(x_all, feature_names, x_others=[x_outer], cutoff=20, scale_data=True)

    scaler = StandardScaler()
    scaler.fit(x_all)
    x_all = scaler.transform(x_all)
    x_fs = scaler.transform(x_fs)

    model = models['kr']['model']  
    params = models['kr']['params']

    best_params, _ = nested_cross_val(model, params, x_fs, y)

    model.set_params(**best_params)    #define the model using best params from nested model cv
    #now need to train model with these parameters on x_train and evaluate on x_val
    model = model.fit(x_fs, y)

    #now predict the importance for region 2 for all patients individually... 
    preds = model.predict(x_all[-60:,:])
    preds = np.clip(preds, 0,2)
    avg = np.mean(preds)



    # #make plot of patient importance scattered with real importance
    for j in range(1):
        preds = []
        mae_preds = []
        importance_vals = []
        for i in range(18):
            pred = model.predict(x_all[i*60+j,:].reshape(1,-1))[0]
            pred_mae = model_mae.predict(x_all[i*60+j,:].reshape(1,-1))[0]
            preds.append(pred)
            mae_preds.append(pred_mae)
            importance_vals.append(y[i*2])
        preds = np.clip(np.array(preds), 0,10)
        mae_preds = np.clip(np.array(mae_preds),0,1)
        zipped = list(zip(preds, importance_vals))
        sorted_vals = sorted(zipped, key=lambda x: x[1])
        preds, importance_vals = zip(*sorted_vals)
        #preds = preds / np.amax(preds)
        fig, ax = plt.subplots(figsize=(12,8))
        ax.errorbar(list(range(18)), preds, yerr=mae_preds, c='orange', markersize=7,fmt='o', linestyle='none')

        ax.scatter(list(range(18)), importance_vals, c='mediumorchid', marker='s', s=70)

        ax.set_xlabel("Sub-Region", fontsize=18)
        ax.set_ylabel("Importance", fontsize=18)
        from  matplotlib.ticker import FuncFormatter
        plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x+1)))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='both', labelsize=16)
        plt.show(block=True)



    #make plot of parotid in 3d with population imp colors, patient impo colors, then combined

    preds = []
    mae_preds = []
    importance_vals = []
    for i in range(18):
        pred = model.predict(x_all[i*60+0,:].reshape(1,-1))[0]
        pred_mae = model_mae.predict(x_all[i*60+0,:].reshape(1,-1))[0]
        preds.append(pred)
        mae_preds.append(pred_mae)
        importance_vals.append(y[i*2])
    preds = np.clip(np.array(preds), 0,10)
    mae_preds = np.clip(np.array(mae_preds),0,1)
    # zipped = list(zip(preds, importance_vals))
    # sorted_vals = sorted(zipped, key=lambda x: x[1])
    # preds, importance_vals = zip(*sorted_vals)
    #preds = preds / np.amax(preds)
    #load masks
    with open("/media/caleb/WDBlue/PET_PSMA/pet_analysis_phd/data/01/mask_dict", "rb") as fp:
        contours = pickle.load(fp)["PET"]["PAROTID_L_JSW"].segmented_contours_reg


    #population importance is 
    pop_imp = np.array([0.751310670731707,  0.526618902439024,   0.386310975609756,
        1,   0.937500000000000,   0.169969512195122,   0.538871951219512 ,  0.318064024390244,   0.167751524390244,
        0.348320884146341,   0.00611608231707317, 0.0636128048780488,  0.764222560975610,   0.0481192835365854,  0.166463414634146,
        0.272984146341463,   0.0484897103658537,  0.035493902439024])
    ind_imp = preds
    ind_imp_difs = np.array(ind_imp) - np.array(pop_imp)
    ind_imp_difs[ind_imp_difs < 0] = 0
    imp_comb = np.zeros((18))
    imp_comb = pop_imp*2 / (np.exp(-3*ind_imp_difs)+1)
    #if importance diff is more than 50% hgiher prediction in region, then boost constraints so highest constraint 
    

    from Visuals_Subsegs import plotSubsegments
    plotSubsegments(contours, values=pop_imp, min_val = np.amin(imp_comb), max_val=np.amax(imp_comb))
    plotSubsegments(contours, values=ind_imp)
    plotSubsegments(contours, values=imp_comb)
        
    return 

def feature_importance_plot():
    data = extract_radiomics.load_design_matrix(try_load=False, normalize_by_whole=True, suv_type="bw", deblurred=True)
    
    x_all = data["all_spatial"][:,:-1]
    y_all = data["all_spatial"][:,-1]
    feature_names = data["feature_names_spatial_all"]


    #first estimate the error by making a LOO  error prediction model
    x, y, feature_names = get_subseg_avgs(x_all, y_all, feature_names, combine_l_r=False)   #get population averages for each subseg 
   
    x, feature_names, [x_all] = sort_by_correlation(x, y, feature_names, corr_type="spearman", x_others=[x_all])

    _, _,_, [x] = pca(x_all, feature_names, x_others=[x], cutoff=20, scale_data=True, print_top_features=True)

    return
