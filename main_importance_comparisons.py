#this code was used for model training as in the article 

"PSMA PET as a predictive tool for sub-regional importance estimates in the parotid gland"

#By Caleb Sample, Arman Rahmim, Francois Benard, Jonn Wu, Haley Clark.



import uptake_analysis
import os 
import pickle
import numpy as np
from scipy.stats import spearmanr, ttest_rel
from scipy.spatial import ConvexHull
from copy import deepcopy
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, binary_dilation
import matplotlib.pyplot as plt
import math
from skimage.segmentation import find_boundaries
from matplotlib.widgets import Slider
import matplotlib.ticker as ticker
from skimage.measure import marching_cubes
from utils import plot_3d_image, plot_3d_voxel_image
import get_contour_masks
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import extract_radiomics
import nrrd
import radiomics
import six
import copy
import glob
from training import make_perturbation_plot, feature_importance_plot
#import feature_selection    #cant import on windows

dicom_folder = os.path.join(os.getcwd(), "SG_PETRT")



def main():
    data_folder = "/media/caleb/WDBlue/PET_PSMA/pet_ivim_analysis_phd/data"
    #uptake_analysis.process_all_img_arrays(dicom_folder)
    #uptake_analysis.process_all_mask_arrays(dicom_folder, subsegmentation=[2,1,2], roi_names=[["sm_","j","w"], ["par", "j", "w"], ["mandible"]])
    #get_han_subregion_masks(data_folder, deblurred=False)

    #importance_vs_uptake_clark(data_folder, deblurred=True, plot=True)
    #feature_importance_plot()
    #make_perturbation_plot()
    #make_best_model_and_fs_plots()
    #importance_vs_uptake_vanluijk(data_folder, deblurred=True)
    
    #lateral_uptake_clark(data_folder, deblurred=False)

    #importance_vs_uptake_han(data_folder, deblurred=True)

    #importance_vs_buettner(data_folder, deblurred=True)

    # uptake_analysis.get_best_threshold(data_folder, deblurred=True)
    # uptake_analysis.get_best_threshold(data_folder, deblurred=False)
    
    # uptake_analysis.get_total_uptake(data_folder, deblurred=True)
    # uptake_analysis.get_total_uptake(data_folder, deblurred=False)
    #uptake_analysis.plot_best_thresholds(roi="par", deblurred=True)
    #uptake_analysis.plot_best_thresholds(roi="par", deblurred=False)
    #cross_val_plot()
    #plot_pet_with_texture()
    analyze_pet_ct_correlation_subsegs()
    


def analyze_pet_ct_correlation_subsegs():
    lr_vals = []
    sr_vals = []
    uptakes = []

    ss_uptakes = []
    ss_uptakes_ratios = []
    ss_srs = []
    ss_lrs = []
    ss_srs_ratios = []
    ss_lrs_ratios = []
    for i in range(18):
        ss_uptakes.append([])
        ss_uptakes_ratios.append([])
        ss_srs.append([])
        ss_lrs.append([])
        ss_srs_ratios.append([])
        ss_lrs_ratios.append([])

    params = "params_feat_maps.yaml"
    data_folder = os.path.join(os.getcwd(), "SG_PETRT")
    patient_nums = os.listdir(data_folder)
    patient_nums.sort()
    for patient_num in patient_nums:
        
        for modality in ["CT"]:      
            save_path = os.path.join(os.getcwd(), "feature_maps", patient_num)
            
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            save_path = os.path.join(save_path, modality)    
            #pet and ct nrrd files will be in separate subfolders
            path_nrrd = os.path.join(os.getcwd(), "nrrd_files", patient_num)

            path_nrrd = os.path.join(path_nrrd, modality)       
            img_nrrd_path = os.path.join(path_nrrd, "images")
            structures_nrrd_path = os.path.join(path_nrrd, "masks")

            img_series_path = os.path.join(os.path.join(os.getcwd(), "data"), patient_num, "img_dict")
            mask_path = os.path.join(os.path.join(os.getcwd(), "data"), patient_num, "mask_dict")
            with open(img_series_path, "rb") as fp:
                img_dict = pickle.load(fp) 
            with open(mask_path, "rb") as fp:
                mask_dict = pickle.load(fp) 
            img_series_pet = img_dict["PET"]
            suv_factors = img_series_pet.suv_factors
            img_pet = img_series_pet.image_array * suv_factors[1]   #lbm suv  


        
            for img_path in glob.glob(os.path.join(img_nrrd_path,"*")):
                if "bw" in img_path or "bsa" in img_path:
                    continue #only calculate for lbm image (pet)
                for struct_file in os.listdir(structures_nrrd_path):
                    structure_name = struct_file.split("__")[1]
                    segmentation_name = struct_file.split("__")[2]
                    if "whole" not in segmentation_name:
                        continue #dont calculate for subsegment masks
                    mask_pet_subsegs = mask_dict["PET"][structure_name].subseg_masks_reg
                    mask_ct_subsegs = mask_dict["CT"][structure_name].subseg_masks_reg

                    mask_pet = mask_dict["PET"][structure_name].whole_roi_masks
                    mask_ct = mask_dict["CT"][structure_name].whole_roi_masks

                    #mask, header_mask = nrrd.read(mask_path)
                    lr_img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(save_path, str("original_glrlm_LongRunEmphasis" + "__" + structure_name + "__" + modality + '__.nrrd'))))
                    sr_img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(save_path, str("original_glrlm_ShortRunEmphasis" + "__" + structure_name + "__" + modality + '__.nrrd'))))

                    uptake = img_pet[mask_pet]
                    lr = lr_img[mask_ct]
                    sr = sr_img[mask_ct]
                    uptakes.append(np.mean(uptake))
                    lr_vals.append(np.mean(lr))
                    sr_vals.append(np.mean(sr))

                    for i in range(18):
                        ss_uptakes[i].append(np.mean(img_pet[mask_pet_subsegs[i]]))
                        ss_uptakes_ratios[i].append((ss_uptakes[i][-1]-uptakes[-1])/uptakes[-1])
                        ss_srs[i].append(np.mean(sr_img[mask_ct_subsegs[i]]))
                        ss_lrs[i].append(np.mean(lr_img[mask_ct_subsegs[i]]))
                        ss_srs_ratios[i].append((ss_srs[i][-1] - sr_vals[-1])/sr_vals[-1])
                        ss_lrs_ratios[i].append((ss_lrs[i][-1] - lr_vals[-1])/lr_vals[-1])

                    print(f'{structure_name} of patient {patient_num}')

    #now get correlations.
    import scipy.stats
    whole_PSMA_lr_corr = scipy.stats.spearmanr(uptakes, lr_vals)  
    whole_PSMA_sr_corr = scipy.stats.spearmanr(uptakes, sr_vals)  
    ss_uptakes_stds = copy.deepcopy(ss_uptakes)
    
    ss_lrs_stds = copy.deepcopy(ss_lrs)
    ss_srs_stds = copy.deepcopy(ss_srs)
    ss_lrs_ratios_stds = copy.deepcopy(ss_lrs_ratios)
    ss_srs_ratios_stds = copy.deepcopy(ss_srs_ratios)
    #average over subsegs
    for i in range(18):
        ss_uptakes_stds = np.std(ss_uptakes[i])
        ss_lrs_stds[i] = np.std(ss_lrs[i])
        ss_srs_stds[i] = np.std(ss_srs[i])
        ss_lrs_ratios_stds[i] = np.std(ss_lrs_ratios[i])
        ss_srs_ratios_stds[i] = np.std(ss_srs_ratios[i])

        ss_uptakes[i] = np.mean(ss_uptakes[i])
        ss_uptakes_ratios[i] = np.mean(ss_uptakes_ratios[i])
        ss_lrs[i] = np.mean(ss_lrs[i])
        ss_srs[i] = np.mean(ss_srs[i])
        ss_lrs_ratios[i] = np.mean(ss_lrs_ratios[i])
        ss_srs_ratios[i] = np.mean(ss_srs_ratios[i])
    ss_psma_lr_corr = scipy.stats.spearmanr(ss_uptakes,ss_lrs)
    ss_psma_sr_corr = scipy.stats.spearmanr(ss_uptakes, ss_srs)

    ss_psma_lr_ratio_corr = scipy.stats.spearmanr(ss_uptakes_ratios,ss_lrs_ratios)
    ss_psma_sr_ratio_corr = scipy.stats.spearmanr(ss_uptakes_ratios, ss_srs_ratios)

    fig, ax = plt.subplots(3,2, figsize=(30,30))
    ax[0,0].errorbar(ss_uptakes, ss_lrs,  yerr=ss_lrs_stds,fmt='o', color='turquoise', markerfacecolor='turquoise', markeredgecolor='black')#,label="Sub-segment Absolute PSMA PET vs CT GLRLML ($r_s = 0.90, p < 0.001$)")
    ax[0,1].errorbar(ss_uptakes, ss_srs, yerr=ss_srs_stds, fmt='o', color='darkviolet', markerfacecolor='darkviolet', markeredgecolor='black')#, label="Sub-segment Absolute PSMA PET vs CT GLRLMS ($r_s = -0.74, p < 0.001$)")
    ax[1,0].errorbar(ss_uptakes_ratios, ss_lrs_ratios,  yerr=ss_lrs_ratios_stds,fmt='o', color='turquoise', markerfacecolor='turquoise', markeredgecolor='black')#, label="Sub-segment Relative PSMA PET vs CT GLRLML ($r_s = 0.89, p < 0.001$)")
    ax[1,1].errorbar(ss_uptakes_ratios, ss_srs_ratios, yerr=ss_srs_ratios_stds, fmt='o', color='darkviolet', markerfacecolor='darkviolet', markeredgecolor='black')#, label="Sub-segment RelativePSMA PET vs CT GLRLMS ($r_s = -0.77, p < 0.001$)")
    ax[2,0].errorbar(uptakes, lr_vals,fmt='o', color='turquoise', markerfacecolor='turquoise', markeredgecolor='black')#, label="Whole Gland PSMA PET vs CT GLRLML ($r_s = 0.15, p = 0.23$)")
    ax[2,1].errorbar(uptakes, sr_vals, fmt='o', color='darkviolet', markerfacecolor='darkviolet', markeredgecolor='black')#, label="Whole Gland PSMA PET vs CT GLRLMS ($r_s = -0.22, p = 0.10$)")
    
    ax[1,0].set_xlabel(r"$\frac{\overline{SUV^{segment}_{mean}} - \overline{SUV^{whole}_{mean}}}{\overline{SUV^{whole}_{mean}}}$", fontsize=16)
    ax[1,1].set_xlabel(r"$\frac{\overline{SUV^{segment}_{mean}} - \overline{SUV^{whole}_{mean}}}{\overline{SUV^{whole}_{mean}}}$", fontsize=16)
    ax[0,0].set_xlabel(r"$\overline{SUV^{segment}_{mean}}$", fontsize=16)
    ax[0,1].set_xlabel(r"$\overline{SUV^{segment}_{mean}}$", fontsize=16)
    ax[2,0].set_xlabel(r"$SUV^{whole}_{mean}$", fontsize=16)
    ax[2,1].set_xlabel(r"$SUV^{whole}_{mean}$", fontsize=16)

    # ax[0,0].legend(loc="upper left", frameon=False)
    # ax[0,1].legend(loc="upper left", frameon=False)
    # ax[1,0].legend(loc="upper left", frameon=False)
    # ax[1,1].legend(loc="upper left", frameon=False)
    # ax[2,0].legend(loc="upper left", frameon=False)
    # ax[2,1].legend(loc="upper left", frameon=False)

    ax[1,0].set_ylabel(r"CT $\frac{\overline{GLRLML^{segment}_{mean}} - \overline{GLRLML^{whole}_{mean}}}{\overline{GLRLML^{whole}_{mean}}}$", fontsize=16)
    ax[1,1].set_ylabel(r"CT $\frac{\overline{GLRLMS^{segment}_{mean}} - \overline{GLRLMS^{whole}_{mean}}}{\overline{GLRLMS^{whole}_{mean}}}$", fontsize=16)
    ax[0,0].set_ylabel(r"CT $\overline{GLRLML_{mean}}$", fontsize=16)
    ax[0,1].set_ylabel(r"CT $\overline{GLRLMS_{mean}}$", fontsize=16)
    ax[2,0].set_ylabel(r"CT $\overline{GLRLML^{whole}_{mean}}$", fontsize=16)
    ax[2,1].set_ylabel(r"CT $\overline{GLRLMS^{whole}_{mean}}$", fontsize=16)

    ax[0,0].set_title("Sub-segment Absolute PSMA PET vs CT GLRLML ($r_s = 0.90, p < 0.001$)", fontsize=8,loc="left")
    ax[0,1].set_title("Sub-segment Absolute PSMA PET vs CT GLRLMS ($r_s = -0.74, p < 0.001$)", fontsize=8,loc="left")
    ax[1,0].set_title("Sub-segment Relative PSMA PET vs CT GLRLML ($r_s = 0.89, p < 0.001$)", fontsize=8,loc="left")
    ax[1,1].set_title("Sub-segment RelativePSMA PET vs CT GLRLMS ($r_s = -0.77, p < 0.001$)", fontsize=8, loc="left")
    ax[2,0].set_title("Whole Gland PSMA PET vs CT GLRLML ($r_s = 0.15, p = 0.23$)", fontsize=8, loc="left")
    ax[2,1].set_title("Whole Gland PSMA PET vs CT GLRLMS ($r_s = -0.22, p = 0.10$)", fontsize=8, loc="left")

    ax[0,0].spines['top'].set_visible(False)
    ax[0,0].spines['right'].set_visible(False)
    ax[0,1].spines['top'].set_visible(False)
    ax[0,1].spines['right'].set_visible(False)
    ax[1,0].spines['top'].set_visible(False)
    ax[1,0].spines['right'].set_visible(False)
    ax[1,1].spines['top'].set_visible(False)
    ax[1,1].spines['right'].set_visible(False)
    ax[2,0].spines['top'].set_visible(False)
    ax[2,0].spines['right'].set_visible(False)
    ax[2,1].spines['top'].set_visible(False)
    ax[2,1].spines['right'].set_visible(False)
    
    plt.show()
    print("")

def plot_pet_with_texture():
    data_folder = os.path.join(os.getcwd(), "SG_PETRT")
    patient_nums = os.listdir(data_folder)
    patient_nums.sort()



    for patient_num in patient_nums[1:]:    

        img_series_path = os.path.join(os.path.join(os.getcwd(), "data_deblur"), patient_num, "img_dict")

        with open(img_series_path, "rb") as fp:
            img_dict = pickle.load(fp) 
        img_series_pet = img_dict["PET"]
        img_ct = img_dict["CT"].image_array
        suv_factors = img_series_pet.suv_factors
        img_pet = img_series_pet.image_array * suv_factors[1]   #lbm suv  

        save_path = os.path.join(os.getcwd(), "feature_maps", patient_num)
        # lr_img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(save_path, "CT", str("original_glrlm_LongRunEmphasis"  + "__" + "CT" + '__.nrrd'))))
        # sr_img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(save_path,"CT", str("original_glrlm_ShortRunEmphasis"  + "__" + "CT" + '__.nrrd'))))
        fig, ax = plt.subplots(1,2)
        cmap = "plasma"
        ax[0].imshow(img_pet[58,50:160,80:175], cmap=cmap)
        # ax[1].imshow(lr_img[2,...],cmap=cmap)
        # ax[2].imshow(sr_img[2,...],cmap=cmap)
        ax[1].imshow(img_ct[54,50:-150, 100:-100], cmap=cmap, vmin=-1100, vmax=1000)

        plt.show()

    
def cross_val_plot():
    vals = np.ones((81, 9)) * 0.6
    vals_other = np.zeros((81,9))
    vals_other_2 = np.zeros((81,9))
    test_idx = -1 
    val_idx = 0
    for val in range(81):
        if val % 9 == 0:
            test_idx += 1
            vals[val,test_idx] = 1
            vals_other[val, :test_idx] = 0.5
            vals_other[val, test_idx+1:] = 0.5
            val_idx = 0
            continue
            
        if val_idx == test_idx:
            val_idx += 1
        vals_other_2[val,test_idx] = 1
        vals[val, val_idx] = 0.1
        val_idx += 1

    masked_vals = np.ma.masked_equal(vals_other, 0)
    masked_vals2 = np.ma.masked_equal(vals_other_2, 0)
    fig, ax = plt.subplots(figsize=(20,20))
    skip_edge_values = [1]
    ax.pcolor(vals, cmap="plasma", edgecolor="k")
    ax.pcolor(masked_vals, cmap="cool", edgecolor="k")
    ax.pcolor(masked_vals2, cmap="hot", edgecolor="k")
    plt.xticks([])
    plt.yticks([])

    # Turn off both x and y axes
    #plt.axis('off')

        

    # ax.pcolor(vals, edgecolors='k', linewidths=2, cmap="plasma")
    plt.gca().invert_yaxis()
    plt.show(block=True)
def uptake_sa_v_correction():
    
    try:
        with open(os.path.join(os.getcwd(), "stats", "uptake_sa_v_corrections"), "rb") as fp:
            all_patient_uptakes =pickle.load(fp)    
    except:
        all_patient_uptakes = {}
        all_patient_uptakes["regular"] = []    
        all_patient_uptakes["corrected"] = []   
        all_patient_uptakes["ratios"] = [] #sa/v ratios
        all_patient_uptakes["volumes"] = []
        all_patient_uptakes["sas"] = []
        all_patient_uptakes["regular_sm"] = []    
        all_patient_uptakes["corrected_sm"] = []   
        all_patient_uptakes["ratios_sm"] = [] #sa/v ratios
        all_patient_uptakes["volumes_sm"] = []
        all_patient_uptakes["sas_sm"] = []


        patient_nums = os.listdir(data_folder)
        patient_nums.sort()
        for patient_num in patient_nums:
            print(f"Calculating uptakes for patient {patient_num}")
            patient_folder = os.path.join(data_folder, patient_num)
                
            img_series_path = os.path.join(os.getcwd(), "data", patient_num, "img_dict")

            with open(img_series_path, "rb") as fp:
                img_dict = pickle.load(fp)
            #load masks 
            with open(os.path.join(os.getcwd(), "data", patient_num, "mask_dict"), "rb") as fp:
                mask_dict = pickle.load(fp)
            structure_masks_pet = mask_dict["PET"]
            img_series_pet = img_dict["PET"]
            suv_factors = img_series_pet.suv_factors
            img_pet = img_series_pet.image_array * suv_factors[1]   #lbm suv     
            voxel_volume_pet = img_dict["PET"].pixel_spacing[0] * img_dict["PET"].pixel_spacing[1] * img_dict["PET"].slice_thickness 
        
            for structure in structure_masks_pet:
                if "par" in structure.lower() and "sup" not in structure.lower():
                    roi_mask = structure_masks_pet[structure].whole_roi_masks
                    roi_contours = structure_masks_pet[structure].whole_roi_img
                    all_patient_uptakes["regular"].append(np.nanmean(img_pet[roi_mask]))
                    
                    #now need surface area and volume of gland
                    volume = np.count_nonzero(roi_mask) * voxel_volume_pet
                    sa = get_contour_roi_sa(roi_contours)
                    
                    sa_v_ratio = sa/volume
                    all_patient_uptakes["corrected"].append(np.nanmean(img_pet[roi_mask]/ sa_v_ratio))
                    all_patient_uptakes["ratios"].append(sa_v_ratio)
                    all_patient_uptakes["volumes"].append(volume)
                    all_patient_uptakes["sas"].append(sa)
                elif "sm" in structure.lower() and "sup" not in structure.lower():
                    roi_mask = structure_masks_pet[structure].whole_roi_masks
                    roi_contours = structure_masks_pet[structure].whole_roi_img
                    all_patient_uptakes["regular_sm"].append(np.nanmean(img_pet[roi_mask]))
                    
                    #now need surface area and volume of gland
                    volume = np.count_nonzero(roi_mask) * voxel_volume_pet
                    sa = get_contour_roi_sa(roi_contours)
                    
                    sa_v_ratio = sa/volume
                    all_patient_uptakes["corrected_sm"].append(np.nanmean(img_pet[roi_mask]/ sa_v_ratio))
                    all_patient_uptakes["ratios_sm"].append(sa_v_ratio)
                    all_patient_uptakes["volumes_sm"].append(volume)
                    all_patient_uptakes["sas_sm"].append(sa)
        with open(os.path.join(os.getcwd(), "stats", "uptake_sa_v_corrections"), "wb") as fp:
            pickle.dump(all_patient_uptakes, fp)
    for i, item in enumerate(all_patient_uptakes["corrected_sm"]):
        all_patient_uptakes["corrected_sm"][i] = np.nanmean(item)
    for i, item in enumerate(all_patient_uptakes["regular_sm"]):
        all_patient_uptakes["regular_sm"][i] = np.nanmean(item)

    for i, item in enumerate(all_patient_uptakes["corrected"]):
        all_patient_uptakes["corrected"][i] = np.nanmean(item)
    for i, item in enumerate(all_patient_uptakes["regular"]):
        all_patient_uptakes["regular"][i] = np.nanmean(item)
    all_patient_uptakes["sas"][33] = 11000
    all_patient_uptakes["ratios"][33] = 0.13

    with open(os.path.join(os.getcwd(), "stats", "uptake_sa_v_corrections"), "wb") as fp:
        pickle.dump(all_patient_uptakes, fp)
    #get correlation of corrected and uncorrected volume with the different metrics
   
    #do for submandibulars
    corr_cr, p_cr = spearmanr(all_patient_uptakes["corrected"], all_patient_uptakes["ratios"])
    corr_cv, p_cv = spearmanr(all_patient_uptakes["corrected"], all_patient_uptakes["volumes"])
    corr_csa, p_csa = spearmanr(all_patient_uptakes["corrected"], all_patient_uptakes["sas"])

    corr_rr, p_rr = spearmanr(all_patient_uptakes["regular"], all_patient_uptakes["ratios"])
    corr_rv, p_rv = spearmanr(all_patient_uptakes["regular"], all_patient_uptakes["volumes"])
    corr_rsa, p_rsa = spearmanr(all_patient_uptakes["regular"], all_patient_uptakes["sas"])
    
    fig, ax = plt.subplots(nrows=3, ncols=2)
    m1, b1 = np.polyfit(all_patient_uptakes["ratios"], all_patient_uptakes["regular"], 1)
    ax[0,0].plot(all_patient_uptakes["ratios"], m1*np.array(all_patient_uptakes["ratios"]) + b1, color="g")
    ax[0,0].scatter(all_patient_uptakes["ratios"], all_patient_uptakes["regular"])

    m2, b2 = np.polyfit(all_patient_uptakes["ratios"], all_patient_uptakes["corrected"], 1)
    ax[0,1].plot(all_patient_uptakes["ratios"], m2*np.array(all_patient_uptakes["ratios"]) + b2, color="b")
    ax[0,1].scatter(all_patient_uptakes["ratios"], all_patient_uptakes["corrected"])

    m3, b3 = np.polyfit(all_patient_uptakes["volumes"], all_patient_uptakes["regular"], 1)
    ax[1,0].plot(all_patient_uptakes["volumes"], m3*np.array(all_patient_uptakes["volumes"]) + b3, color="g")
    ax[1,0].scatter(all_patient_uptakes["volumes"], all_patient_uptakes["regular"])

    m4, b4 = np.polyfit(all_patient_uptakes["volumes"], all_patient_uptakes["corrected"], 1)
    ax[1,1].plot(all_patient_uptakes["volumes"], m4*np.array(all_patient_uptakes["volumes"]) + b4, color="b")
    ax[1,1].scatter(all_patient_uptakes["volumes"], all_patient_uptakes["corrected"])

    m5, b5 = np.polyfit(all_patient_uptakes["sas"], all_patient_uptakes["regular"], 1)
    ax[2,0].plot(all_patient_uptakes["sas"], m5*np.array(all_patient_uptakes["sas"]) + b5, color="g")
    ax[2,0].scatter(all_patient_uptakes["sas"], all_patient_uptakes["regular"])

    m6, b6 = np.polyfit(all_patient_uptakes["sas"], all_patient_uptakes["corrected"], 1)
    ax[2,1].plot(all_patient_uptakes["sas"], m6*np.array(all_patient_uptakes["sas"]) + b6, color="b")
    ax[2,1].scatter(all_patient_uptakes["sas"], all_patient_uptakes["corrected"])
    plt.show(block=True)


    #do for submandibulars
    corr_cr, p_cr = spearmanr(all_patient_uptakes["corrected_sm"], all_patient_uptakes["ratios_sm"])
    corr_cv, p_cv = spearmanr(all_patient_uptakes["corrected_sm"], all_patient_uptakes["volumes_sm"])
    corr_csa, p_csa = spearmanr(all_patient_uptakes["corrected_sm"], all_patient_uptakes["sas_sm"])

    corr_rr, p_rr = spearmanr(all_patient_uptakes["regular_sm"], all_patient_uptakes["ratios_sm"])
    corr_rv, p_rv = spearmanr(all_patient_uptakes["regular_sm"], all_patient_uptakes["volumes_sm"])
    corr_rsa, p_rsa = spearmanr(all_patient_uptakes["regular_sm"], all_patient_uptakes["sas_sm"])
    
    fig, ax = plt.subplots(nrows=3, ncols=2)
    m1, b1 = np.polyfit(all_patient_uptakes["ratios_sm"], all_patient_uptakes["regular_sm"], 1)
    ax[0,0].plot(all_patient_uptakes["ratios_sm"], m1*np.array(all_patient_uptakes["ratios_sm"]) + b1, color="g")
    ax[0,0].scatter(all_patient_uptakes["ratios_sm"], all_patient_uptakes["regular_sm"])

    m2, b2 = np.polyfit(all_patient_uptakes["ratios_sm"], all_patient_uptakes["corrected_sm"], 1)
    ax[0,1].plot(all_patient_uptakes["ratios_sm"], m2*np.array(all_patient_uptakes["ratios_sm"]) + b2, color="b")
    ax[0,1].scatter(all_patient_uptakes["ratios_sm"], all_patient_uptakes["corrected_sm"])

    m3, b3 = np.polyfit(all_patient_uptakes["volumes_sm"], all_patient_uptakes["regular_sm"], 1)
    ax[1,0].plot(all_patient_uptakes["volumes_sm"], m3*np.array(all_patient_uptakes["volumes_sm"]) + b3, color="g")
    ax[1,0].scatter(all_patient_uptakes["volumes_sm"], all_patient_uptakes["regular_sm"])

    m4, b4 = np.polyfit(all_patient_uptakes["volumes_sm"], all_patient_uptakes["corrected_sm"], 1)
    ax[1,1].plot(all_patient_uptakes["volumes_sm"], m4*np.array(all_patient_uptakes["volumes_sm"]) + b4, color="b")
    ax[1,1].scatter(all_patient_uptakes["volumes_sm"], all_patient_uptakes["corrected_sm"])

    m5, b5 = np.polyfit(all_patient_uptakes["sas_sm"], all_patient_uptakes["regular_sm"], 1)
    ax[2,0].plot(all_patient_uptakes["sas_sm"], m5*np.array(all_patient_uptakes["sas_sm"]) + b5, color="g")
    ax[2,0].scatter(all_patient_uptakes["sas_sm"], all_patient_uptakes["regular_sm"])

    m6, b6 = np.polyfit(all_patient_uptakes["sas_sm"], all_patient_uptakes["corrected_sm"], 1)
    ax[2,1].plot(all_patient_uptakes["sas_sm"], m6*np.array(all_patient_uptakes["sas_sm"]) + b6, color="b")
    ax[2,1].scatter(all_patient_uptakes["sas_sm"], all_patient_uptakes["corrected_sm"])
    plt.show(block=True)
    return

def get_contour_roi_sa(contours):
    #first prepare contour points for sa calc
    for c, contour in enumerate(contours):
        contours[c] = contour[0] #get rid of nested contour list
    points = np.concatenate(contours, axis=0)
    hull = ConvexHull(points)
    sa =hull.area


    return sa


def importance_vs_uptake_clark(data_folder, deblurred=False, plot=False):
    #first get the mean uptake for each subsegment over all patients
    all_patient_uptakes = {}
    importance_vals = [0.751310670731707,  0.526618902439024,   0.386310975609756,
        1,   0.937500000000000,   0.169969512195122,   0.538871951219512 ,  0.318064024390244,   0.167751524390244,
        0.348320884146341,   0.00611608231707317, 0.0636128048780488,  0.764222560975610,   0.0481192835365854,  0.166463414634146,
        0.272984146341463,   0.0484897103658537,  0.035493902439024]
    try:
        2/0
        with open(os.path.join(os.getcwd(), "stats", f"avg_uptake_clark_deblurred_{deblurred}"), "rb") as fp:
            all_patient_uptakes =pickle.load(fp)    
        #now get average over patients
        avg_uptakes = []
        std_uptakes = []
        for i in range(18):
            avg_uptakes.append(np.mean(all_patient_uptakes[i]))
            std_uptakes.append(np.std(all_patient_uptakes[i]))    
    except:


        if deblurred == False:
            for i in range(18):
                all_patient_uptakes[i] = []

            patient_nums = os.listdir(data_folder)
            patient_nums.sort()
            for patient_num in patient_nums:
                print(f"Calculating uptakes for patient {patient_num}")
                patient_folder = os.path.join(data_folder, patient_num)
                    
                img_series_path = os.path.join(os.getcwd(), "data", patient_num, "img_dict")

                with open(img_series_path, "rb") as fp:
                    img_dict = pickle.load(fp)
                #load masks 
                with open(os.path.join(os.getcwd(), "data", patient_num, "mask_dict"), "rb") as fp:
                    mask_dict = pickle.load(fp)
                structure_masks_pet = mask_dict["PET"]
                img_series_pet = img_dict["PET"]
                suv_factors = img_series_pet.suv_factors
                img_pet = img_series_pet.image_array * suv_factors[1]   #lbm suv     

                #calculate subseg suv avgs
                for structure in structure_masks_pet:
                    if "par" in structure.lower() and "sup" not in structure.lower():
                        subsegment_list = structure_masks_pet[structure].subseg_masks_reg
                        whole_roi = structure_masks_pet[structure].whole_roi_masks
                        for s, subseg in enumerate(subsegment_list):

                            uptake = img_pet[subseg]
                            mean = np.nanmax(uptake)
                            all_patient_uptakes[s].append(mean)

        elif deblurred == True:
            for i in range(18):
                all_patient_uptakes[i] = []
            img_folder = data_folder + "_deblur"
            patient_nums = os.listdir(data_folder)
            patient_nums.sort()
            for patient_num in patient_nums:
                print(f"Calculating uptakes for patient {patient_num}")
                patient_folder = os.path.join(data_folder, patient_num)
                    
                img_series_path = os.path.join(img_folder, patient_num, "img_dict")
                with open(img_series_path, "rb") as fp:
                    img_dict = pickle.load(fp)
                with open(os.path.join(data_folder, patient_num, "mask_dict"), "rb") as fp:
                    mask_dict = pickle.load(fp)

                structure_masks_pet = mask_dict["PET"]
 
                with open(os.path.join(os.getcwd(),"predictions",  f"{patient_num}_registered_predictions.txt"), "rb") as fp:
                    img_pet, _ = pickle.load(fp)
                #calculate subseg suv avgs
                # with open(os.path.join(os.getcwd(), "stats", f"avg_uptake_clark_deblurred_{deblurred}"), "rb") as fp:
                #     all_patient_uptakes =pickle.load(fp)    
                # #now get average over patients
                # avg_uptakes = []
                # std_uptakes = []
                # for i in range(18):
                #     avg_uptakes.append(np.mean(all_patient_uptakes[i]))
                #     std_uptakes.append(np.std(all_patient_uptakes[i]))
                for structure in structure_masks_pet:
                    if "par" in structure.lower() and "sup" not in structure.lower():
                        subsegment_list = structure_masks_pet[structure].subseg_masks_reg_deblur
                        # plot_subsegs_importance_voxels(subsegment_list,avg_uptakes)
                        plot_subsegs_importance_voxels(subsegment_list,importance_vals)
                        whole_roi = structure_masks_pet[structure].whole_roi_masks_deblur
                        for s, subseg in enumerate(subsegment_list):

                            # plot_3d_image(whole_roi)
                            # plot_3d_image(subseg)
                            # img_pet[whole_roi] = 0
                            # plot_3d_image(img_pet)
                            uptake = img_pet[subseg]
                            mean = np.nanmean(uptake)
                            all_patient_uptakes[s].append(mean)



        #now get average over patients
        avg_uptakes = []
        std_uptakes = []
        for i in range(18):
            avg_uptakes.append(np.mean(all_patient_uptakes[i]))
            std_uptakes.append(np.std(all_patient_uptakes[i]))
        with open(os.path.join(os.getcwd(), "stats", f"avg_uptake_clark_deblurred_{deblurred}"), "wb") as fp:
            pickle.dump(all_patient_uptakes, fp)


    zipped = list(zip(avg_uptakes, importance_vals))
    sorted_vals = sorted(zipped, key=lambda x: x[0])
    avg_uptakes, importance_vals = zip(*sorted_vals)

    #get correlation with importance
    corr, p = spearmanr(avg_uptakes, importance_vals)
    print(f"Correlation = {corr}, p = {p}")
    if plot == True:
        fig, ax = plt.subplots(figsize=(15,15))
        slope, intercept = np.polyfit(avg_uptakes, importance_vals, deg=1)
        line = slope * np.array(avg_uptakes) + intercept
        ax.scatter(avg_uptakes, importance_vals, c="chocolate", s=40)
        ax.plot(avg_uptakes, line, color='firebrick', label=f"Spearman's rank: {round(corr,2)}", linewidth=1.3)
        ax.set_xlabel("$\overline{SUV_{lbm}}$", fontsize=20)
        ax.set_ylabel("Relative Importance", fontsize=20)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tick_params(axis='both', which='both', labelsize=12)
        plt.legend(fontsize=20)
        plt.show(block=True)
            
    return

def importance_vs_uptake_vanluijk(data_folder, deblurred=False):
    #for the van luijk importance, the region is near the dorsal edge of mandible. for sub-region, get the mandible mask, and then dilate it by 4 voxels and then boolean intersect with parotids, and test uptake
    all_patient_uptakes = {}
    try:
        a = 2/0
        with open(os.path.join(os.getcwd(), "stats", f"avg_uptake_van_deblurred_{deblurred}"), "rb") as fp:
            all_patient_uptakes =pickle.load(fp)    
    except:
        if deblurred == False:
            all_patient_uptakes["critical"] = []    #uptakes in "critical region"
            all_patient_uptakes["non_critical"] = []   #uptakes in non-"critical" region

            patient_nums = os.listdir(data_folder)
            patient_nums.sort()
            for patient_num in patient_nums:
                print(f"Calculating uptakes for patient {patient_num}")
                patient_folder = os.path.join(data_folder, patient_num)
                    
                img_series_path = os.path.join(os.getcwd(), "data", patient_num, "img_dict")

                with open(img_series_path, "rb") as fp:
                    img_dict = pickle.load(fp)
                #load masks 
                with open(os.path.join(os.getcwd(), "data", patient_num, "mask_dict"), "rb") as fp:
                    mask_dict = pickle.load(fp)
                structure_masks_pet = mask_dict["PET"]
                img_series_pet = img_dict["PET"]
                suv_factors = img_series_pet.suv_factors
                img_pet = img_series_pet.image_array * suv_factors[1]   #lbm suv     

                #get mandible mask
                for structure in structure_masks_pet:
                    if "mandible" in structure.lower():
                        mask_mandible = structure_masks_pet[structure].whole_roi_masks
                #want to dilate the mandible for getting region around it after boolean with parotid
                mask_mandible = binary_dilation(mask_mandible, structure=np.ones((3,3,1)))
                #calculate subseg suv avgs
                for structure in structure_masks_pet:
                    if "par" in structure.lower() and "sup" not in structure.lower():
                        roi_mask = structure_masks_pet[structure].whole_roi_masks
                        #now for critical region, take boolean intersection with dilated mandible mask. for non critical, take boolean subtraction
                        mask_crit = np.logical_and(mask_mandible, roi_mask)
                        mask_ncrit = np.logical_and(roi_mask, np.logical_not(mask_mandible))
                        # combined = mask_crit * 2 + mask_ncrit
                        # plot_3d_image(combined)
    
                        #now calculate uptake avg in each region
                        if np.amax(mask_crit.astype(np.uint8)) != 0:
                            uptake_crit = np.nanmax(img_pet[mask_crit])
                            all_patient_uptakes["critical"].append(uptake_crit)
                            uptake_ncrit = np.nanmax(img_pet[mask_ncrit])
                            all_patient_uptakes["non_critical"].append(uptake_ncrit)

        elif deblurred == True:
            all_patient_uptakes["critical"] = []    #uptakes in "critical region"
            all_patient_uptakes["non_critical"] = []   #uptakes in non-"critical" region

            img_folder = data_folder + "_deblur"
            patient_nums = os.listdir(data_folder)
            patient_nums.sort()
            for patient_num in patient_nums:
                print(f"Calculating uptakes for patient {patient_num}")
                patient_folder = os.path.join(data_folder, patient_num)
                    
                img_series_path = os.path.join(img_folder, patient_num, "img_dict")

                with open(img_series_path, "rb") as fp:
                    img_dict = pickle.load(fp)
                #load masks 
                with open(os.path.join(os.getcwd(), "data", patient_num, "mask_dict"), "rb") as fp:
                    mask_dict = pickle.load(fp)
                structure_masks_pet = mask_dict["PET"]
        
                with open(os.path.join(os.getcwd(),"predictions",  f"{patient_num}_registered_predictions.txt"), "rb") as fp:
                    img_pet, _ = pickle.load(fp)


                #get mandible mask
                for structure in structure_masks_pet:
                    if "mandible" in structure.lower():
                        mask_mandible = structure_masks_pet[structure].whole_roi_masks_deblur
                #want to dilate the mandible for getting region around it after boolean with parotid
                mask_mandible = binary_dilation(mask_mandible, structure=np.ones((6,6,1)))
                #calculate subseg suv avgs
                for structure in structure_masks_pet:
                    if "par" in structure.lower() and "sup" not in structure.lower():
                        roi_mask = structure_masks_pet[structure].whole_roi_masks_deblur
                        #now for critical region, take boolean intersection with dilated mandible mask. for non critical, take boolean subtraction
                        mask_crit = np.logical_and(mask_mandible, roi_mask)
                        z_inds = np.where(roi_mask == 1)[0]
                        min_z = np.amin(z_inds)
                        max_z = np.amax(z_inds)
                        mask_crit[0:min_z + int((max_z-min_z)/2),:,:] = 0
                        mask_ncrit = np.logical_and(roi_mask, np.logical_not(mask_crit))
                        
                        # combined = mask_crit * 2 + mask_ncrit
                        # plot_3d_image(combined)
                        #now calculate uptake avg in each region
                        if np.amax(mask_crit.astype(np.uint8)) != 0:
                           # plot_3d_voxel_image_vanluijk(mask_crit, mask_ncrit)
                            uptake_crit = np.nanmean(img_pet[mask_crit])
                            uptake_ncrit = np.nanmean(img_pet[mask_ncrit])
                                
                            all_patient_uptakes["critical"].append(uptake_crit)
                            all_patient_uptakes["non_critical"].append(uptake_ncrit)
        with open(os.path.join(os.getcwd(), "stats", f"avg_uptake_van_deblurred_{deblurred}"), "wb") as fp:
            pickle.dump(all_patient_uptakes, fp)

    #now get average over patients
    avg_crit = np.mean(all_patient_uptakes["critical"])
    std_crit = np.std(all_patient_uptakes["critical"])
    avg_ncrit = np.mean(all_patient_uptakes["non_critical"])
    std_ncrit = np.std(all_patient_uptakes["non_critical"])

    #for this region.. not enough subsegs to test a correlation. so instead 
    t, p = ttest_rel(all_patient_uptakes["critical"], all_patient_uptakes["non_critical"])
    print(f"crit: {avg_crit} +- {std_crit}")
    print(f"non-crit: {avg_ncrit} +- {std_ncrit}")
    print(f"\nt: {t}, p: {p}")
    return

def plot_3d_voxel_image_vanluijk(mask_crit, mask_ncrit):
    mask_crit = mask_crit.astype(int)
    mask_ncrit = mask_ncrit.astype(int)
    mask_crit = np.swapaxes(mask_crit, 0, 2)
    mask_ncrit = np.swapaxes(mask_ncrit, 0, 2)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # Set the background color to white
    ax.set_facecolor((1.0, 1.0, 1.0))

    # Remove axis grids
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)

    depth, height, width = mask_crit.shape
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_zlim(0, depth)

    vertices_crit, faces_crit, _, _ = marching_cubes(mask_crit, level=0.5, spacing=[2.7,2.7,2.8])
    vertices_ncrit, faces_ncrit, _, _ = marching_cubes(mask_ncrit, level=0.5, spacing=[2.7,2.7,2.8])

    min_inds = np.amin(vertices_ncrit, axis=0)
    max_inds = np.amax(vertices_ncrit, axis=0)    
    ax.set_xlim(min_inds[0]-5, max_inds[0]+5)
    ax.set_ylim(min_inds[1]-5, max_inds[1]+5)
    ax.set_zlim(min_inds[2]-5, max_inds[2]+5)

    mask_ncrit_coll = Poly3DCollection(vertices_ncrit[faces_ncrit], facecolors=['turquoise'], alpha=0.25, zorder=1)
    mask_crit_coll = Poly3DCollection(vertices_crit[faces_crit], facecolors=['red'],edgecolor=['firebrick'], alpha=1, zorder=20)
    
    ax.add_collection3d(mask_ncrit_coll)
    ax.add_collection3d(mask_crit_coll)
    
    ax.set_box_aspect([1, 1, 1])

    
    plt.show(block=True)
    return
def lateral_uptake_clark(data_folder, deblurred=False):
    #get mean uptake in the lateral and medial half of parotid
    all_patient_uptakes = {}
    all_patient_uptakes["lateral"] = []
    all_patient_uptakes["medial"] = []
    try:
        with open(os.path.join(os.getcwd(), "stats", f"lat_uptake_clark_deblurred_{deblurred}"), "rb") as fp:
            [avg_uptakes, std_uptakes] =pickle.load(fp)    
    except:
   
        if deblurred == False:
            patient_nums = os.listdir(data_folder)
            patient_nums.sort()
            for patient_num in patient_nums:
                print(f"Calculating uptakes for patient {patient_num}")
                    
                img_series_path = os.path.join(os.getcwd(), "data", patient_num, "img_dict")

                with open(img_series_path, "rb") as fp:
                    img_dict = pickle.load(fp)
                #load masks 
                with open(os.path.join(os.getcwd(), "data", patient_num, "mask_dict"), "rb") as fp:
                    mask_dict = pickle.load(fp)
                structure_masks_pet = mask_dict["PET"]
                img_series_pet = img_dict["PET"]
                suv_factors = img_series_pet.suv_factors
                img_pet = img_series_pet.image_array * suv_factors[1]   #lbm suv     

                #calculate subseg suv avgs
                for structure in structure_masks_pet:
                    if "par" in structure.lower() and "sup" not in structure.lower():
                        subsegs_ml = structure_masks_pet[structure].subseg_masks_ml
                        whole_roi = structure_masks_pet[structure].whole_roi_masks

                        if "l" in structure.lower():   #need to flip the order if its the left/right par
                            lat_idx = 1
                            med_idx = 0
                        else:
                            lat_idx = 0
                            med_idx = 1


                        uptake_lat = img_pet[subsegs_ml[lat_idx]]
                        mean_lat = np.nanmean(uptake_lat)
                        all_patient_uptakes["lateral"].append(mean_lat)

                        uptake_med = img_pet[subsegs_ml[med_idx]]
                        mean_med = np.nanmean(uptake_med)
                        all_patient_uptakes["medial"].append(mean_med)

            #now get average over patients

            avg_lat = np.mean(all_patient_uptakes["lateral"])
            std_lat = np.std(all_patient_uptakes["lateral"])
            avg_med = np.mean(all_patient_uptakes["medial"])
            std_med = np.std(all_patient_uptakes["medial"])
        elif deblurred == True:
            patient_nums = os.listdir(data_folder)
            patient_nums.sort()
            for patient_num in patient_nums:
                print(f"Calculating uptakes for patient {patient_num}")
                    
                img_series_path = os.path.join(os.getcwd(), "data_deblur", patient_num, "img_dict")

                with open(img_series_path, "rb") as fp:
                    img_dict = pickle.load(fp)
                #load masks 
                with open(os.path.join(os.getcwd(), "data", patient_num, "mask_dict"), "rb") as fp:
                    mask_dict = pickle.load(fp)
                structure_masks_pet = mask_dict["PET"]
                img_series_pet = img_dict["PET"]
                with open(os.path.join(os.getcwd(),"predictions",  f"{patient_num}_registered_predictions.txt"), "rb") as fp:
                    img_pet, _ = pickle.load(fp)   

                #calculate subseg suv avgs
                for structure in structure_masks_pet:
                    if "par" in structure.lower() and "sup" not in structure.lower():
                        subsegs_ml = structure_masks_pet[structure].subseg_masks_ml_deblur
                        whole_roi = structure_masks_pet[structure].whole_roi_masks_deblur
                        #print(structure)
                        # plot_3d_image(img_pet)
                        # plot_3d_image(whole_roi)
                        # plot_3d_image(subsegs_ml[0])
                        # plot_3d_image(subsegs_ml[1])

                        if "l" in structure.lower():   #need to flip the order if its the left/right par
                            lat_idx = 1
                            med_idx = 0
                        else:
                            lat_idx = 0
                            med_idx = 1


                        uptake_lat = img_pet[subsegs_ml[lat_idx]]
                        mean_lat = np.nanmean(uptake_lat)
                        all_patient_uptakes["lateral"].append(mean_lat)

                        uptake_med = img_pet[subsegs_ml[med_idx]]
                        mean_med = np.nanmean(uptake_med)
                        all_patient_uptakes["medial"].append(mean_med)

            #now get average over patients

            avg_lat = np.mean(all_patient_uptakes["lateral"])
            std_lat = np.std(all_patient_uptakes["lateral"])
            avg_med = np.mean(all_patient_uptakes["medial"])
            std_med = np.std(all_patient_uptakes["medial"])
            
        with open(os.path.join(os.getcwd(), "stats", f"lat_uptake_clark_deblurred_{deblurred}"), "wb") as fp:
            pickle.dump(all_patient_uptakes, fp)
    #get correlation with importance
    t, p = ttest_rel(all_patient_uptakes["lateral"], all_patient_uptakes["medial"])
    print(f"{t}, {p}")
    return    

def importance_vs_uptake_han(data_folder, deblurred=False):
    importance_list_injury = np.array([31.6, 25.8, 45.6, 63.2, 50.6, 81.6, 65.4, 62.6, 82.8]) / 100
    blah = (importance_list_injury -np.amin(importance_list_injury))/ (np.amax(importance_list_injury)-np.amin(importance_list_injury))
    importance_list_recovery = np.array([57.4, 58.1, 18.7, 0.6, 32.8, -11.4, 3.6, -14.76, 0.7]) / 100
    #don't let values be negative
    importance_list_recovery = np.clip(importance_list_recovery, 0, 1)

    try:
        2/0
        with open(os.path.join(os.getcwd(), "stats", f"avg_uptake_han_deblurred_{deblurred}"), "rb") as fp:
            all_patient_uptakes =pickle.load(fp)    
    except:
        if deblurred == False:
            all_patient_uptakes = {"top": {}, "middle": {}, "bottom": {}} #superficial - inferior region
            for section in all_patient_uptakes:
                all_patient_uptakes[section]["p"] = [] #posterior angle region
                all_patient_uptakes[section]["a"] = [] #anterior angle region
                all_patient_uptakes[section]["m"] = [] #medial angle region 
            

            patient_nums = os.listdir(data_folder)
            patient_nums.sort()
            for patient_num in patient_nums:
                print(f"Calculating uptakes for patient {patient_num}")
                patient_folder = os.path.join(data_folder, patient_num)
                    
                img_series_path = os.path.join(os.getcwd(), "data", patient_num, "img_dict")

                with open(img_series_path, "rb") as fp:
                    img_dict = pickle.load(fp)
                #load masks 
                with open(os.path.join(os.getcwd(), "data", patient_num, "mask_dict"), "rb") as fp:
                    mask_dict = pickle.load(fp)

                structure_masks_pet = mask_dict["PET"]
                img_series_pet = img_dict["PET"]
                suv_factors = img_series_pet.suv_factors
                img_pet = img_series_pet.image_array * suv_factors[1]   #lbm suv     

                #calculate subseg suv avgs
                for structure in structure_masks_pet:
                    if "par" in structure.lower() and "sup" not in structure.lower():
                        roi_masks = structure_masks_pet[structure].han_masks
                        whole_masks = structure_masks_pet[structure].whole_roi_masks
                        #now for critical region, take boolean intersection with dilated mandible mask. for non critical, take boolean subtraction

                        for section in all_patient_uptakes:
                            for angle_region in all_patient_uptakes[section]:
        
                                uptake = img_pet[roi_masks[section][angle_region]]
                                
                                all_patient_uptakes[section][angle_region].append(np.nanmax(uptake))
        elif deblurred == True:
            all_patient_uptakes = {"top": {}, "middle": {}, "bottom": {}} #superficial - inferior region
            for section in all_patient_uptakes:
                all_patient_uptakes[section]["p"] = [] #posterior angle region
                all_patient_uptakes[section]["a"] = [] #anterior angle region
                all_patient_uptakes[section]["m"] = [] #medial angle region 
            

            patient_nums = os.listdir(data_folder)
            patient_nums.sort()
            for patient_num in patient_nums:
                print(f"Calculating uptakes for patient {patient_num}")
                    
                img_series_path = os.path.join(os.getcwd(), "data_deblur", patient_num, "img_dict")

                with open(img_series_path, "rb") as fp:
                    img_dict = pickle.load(fp)
                #load masks 
                with open(os.path.join(os.getcwd(), "data", patient_num, "mask_dict"), "rb") as fp:
                    mask_dict = pickle.load(fp)

                structure_masks_pet = mask_dict["PET"]
                img_series_pet = img_dict["PET"]
                with open(os.path.join(os.getcwd(),"predictions",  f"{patient_num}_registered_predictions.txt"), "rb") as fp:
                    img_pet, _ = pickle.load(fp)   

                #calculate subseg suv avgs
                for structure in structure_masks_pet:
                    if "par" in structure.lower() and "sup" not in structure.lower():
                        roi_masks = structure_masks_pet[structure].han_masks_deblur
                        whole_masks = structure_masks_pet[structure].whole_roi_masks_deblur
                        #now for critical region, take boolean intersection with dilated mandible mask. for non critical, take boolean subtraction
                        mask_list = []
                        
                        for section in all_patient_uptakes:
                            for angle_region in all_patient_uptakes[section]:
                                mask_list.append(roi_masks[section][angle_region])
                                uptake = img_pet[roi_masks[section][angle_region]]
                                
                                all_patient_uptakes[section][angle_region].append(np.nanmax(uptake))
                        plot_subsegs_importance_voxels(mask_list, importance_list_injury)
        with open(os.path.join(os.getcwd(), "stats", f"avg_uptake_han_deblurred_{deblurred}"), "wb") as fp:
            pickle.dump(all_patient_uptakes, fp)

    avg_uptakes = []
    std_uptakes = []
    importance_list = []
    for section in ["top", "middle", "bottom"]:
        for angle_region in ["p","a","m"]:
                
            avg_uptakes.append(np.mean(all_patient_uptakes[section][angle_region]))
            std_uptakes.append(np.std(all_patient_uptakes[section][angle_region]))

#get correlation with importance
    corr_inj, p_inj = spearmanr(avg_uptakes, importance_list_injury)
    corr_rec, p_rec = spearmanr(avg_uptakes, importance_list_recovery)
    print(f"Injury corr: {corr_inj}, p: {p_inj}")
    print(f"Injury recovery: {corr_rec}, p: {p_rec}")
    return

def plot_subsegs_importance_voxels(masks, importance_list):

    cmap = plt.get_cmap('rainbow')
    norm = plt.Normalize(vmin=np.min(importance_list), vmax=np.max(importance_list))


    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # Set the background color to white
    ax.set_facecolor((1.0, 1.0, 1.0))

    # Remove axis grids
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    #ax.voxels(masked_image, facecolors=colors, edgecolor=None, shade=False)
    # Get the dimensions of the voxel array
    for m, mask in enumerate(masks):
        mask = binary_dilation(binary_erosion(mask, structure=np.ones((1,1,1))), structure= np.ones((1,1,1)))

        depth, height, width = mask.shape
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_zlim(0, depth)
        # Loop through each voxel and plot it as a solid cube
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    if not mask[z, y, x] == 0:
                        # Define the vertices of the cube
                        vertices = [
                            (x, y, z),
                            (x + 1, y, z),
                            (x + 1, y + 1, z),
                            (x, y + 1, z),
                            (x, y, z + 1),
                            (x + 1, y, z + 1),
                            (x + 1, y + 1, z + 1),
                            (x, y + 1, z + 1)
                        ]

                        # Define the faces of the cube using the vertices
                        faces = [
                            [vertices[0], vertices[1], vertices[2], vertices[3]],
                            [vertices[4], vertices[5], vertices[6], vertices[7]],
                            [vertices[0], vertices[1], vertices[5], vertices[4]],
                            [vertices[2], vertices[3], vertices[7], vertices[6]],
                            [vertices[0], vertices[3], vertices[7], vertices[4]],
                            [vertices[1], vertices[2], vertices[6], vertices[5]]
                        ]

                        # Create a Poly3DCollection object to represent the voxel
                        voxel = Poly3DCollection(faces, facecolors=[cmap(norm(importance_list[m]))], edgecolor=[cmap(norm(importance_list[m]+0.1))])

                        # Add the voxel to the plot
                        ax.add_collection3d(voxel)
    plt.show(block=True)
    return 
def get_han_subregion_masks(data_folder, deblurred=False):

    # def update(val):
    #     slice_index = int(slider.val)
    #     ax.imshow(img[slice_index, :,:])
    #     fig.canvas.draw_idle()
    #divide by 3 superior/inferior by length, and into 3 equal angle pie slices (0 at lateral)
    patient_nums = os.listdir(data_folder)
    patient_nums.sort()
    if deblurred == False:
        for patient_num in patient_nums:
            print(f"Getting Han masks for patient {patient_num}")
                
            img_series_path = os.path.join(os.getcwd(), "data", patient_num, "img_dict")

            with open(img_series_path, "rb") as fp:
                img_dict = pickle.load(fp)
            #load masks 
            with open(os.path.join(os.getcwd(), "data", patient_num, "mask_dict"), "rb") as fp:
                mask_dict = pickle.load(fp)
            structure_masks_pet = mask_dict["PET"]
            img_series_pet = img_dict["PET"]
            suv_factors = img_series_pet.suv_factors
            img_pet = img_series_pet.image_array * suv_factors[1]   #lbm suv     
            coords = img_series_pet.coords_array_img
            #calculate subseg suv avgs
            for structure in structure_masks_pet:
                if "par" in structure.lower() and "sup" not in structure.lower():

                    mask = structure_masks_pet[structure].whole_roi_masks

                    subseg_masks = {"top": {}, "middle": {}, "bottom": {}}
                    for key in subseg_masks:
                        subseg_masks[key]["a"] = np.zeros_like(mask)
                        subseg_masks[key]["m"] = np.zeros_like(mask)
                        subseg_masks[key]["p"] = np.zeros_like(mask)
                    
                    #dilate image by ~3mm
                    mask = binary_dilation(mask)

                    #now divide into 3 by length
                    slice_min, slice_max = np.min(np.where(mask)[0]),  np.max(np.where(mask)[0])
                    third_through = int(0.3333*(slice_max-slice_min)) + slice_min
                    two_thirds_through = int(0.6667*(slice_max-slice_min)) + slice_min
                    bottom_slices = deepcopy(mask)
                    middle_slices = deepcopy(mask)
                    top_slices = deepcopy(mask)
                    for s in range(mask.shape[0]):
                        if np.amax(mask[s,:,:]) == 0:
                            continue
                        elif s < third_through:
                            middle_slices[s,:,:] = 0
                            top_slices[s,:,:] = 0
                        elif s < two_thirds_through:
                            bottom_slices[s,:,:] = 0
                            top_slices[s,:,:] = 0
                        else:
                            middle_slices[s,:,:] = 0
                            bottom_slices[s,:,:] = 0

                    # divide axial slices into pie slices
                    for s in range(mask.shape[0]):
                        if np.amax(bottom_slices[s,:,:]) > 0:
                            #get the center of mass of the slice 
                            centre, centre_idx = get_centre_of_mass(bottom_slices[s,:,:], coords, z=s)
                            #now go through voxels in the mask and see what angle they are at from center of mass. x = lateral, y = posterior
                            mask_voxels = np.where(bottom_slices[s,:,:])
                            for idx in range(len(mask_voxels[0])):
                                x = coords[0,s,mask_voxels[0][idx], mask_voxels[1][idx]] - centre[0]
                                y = coords[1,s,mask_voxels[0][idx], mask_voxels[1][idx]] - centre[1]
                                if "l" not in structure.lower(): #want the zero angle to be in lateral direction, so flip for right parotid
                                    x *= -1
                                angle = np.degrees(np.arctan2(y,x))
                                if angle >= 0 and angle <= 120:
                                    subseg_masks["bottom"]["p"][s,mask_voxels[0][idx], mask_voxels[1][idx]] = 1
                                if angle <= 0 and angle >= -120:
                                    subseg_masks["bottom"]["a"][s,mask_voxels[0][idx], mask_voxels[1][idx]]= 1
                                if angle <= -120 or angle >= 120:
                                    subseg_masks["bottom"]["m"][s,mask_voxels[0][idx], mask_voxels[1][idx]] = 1

                        if np.amax(middle_slices[s,:,:]) > 0:
                            #get the center of mass of the slice 
                            centre, centre_idx = get_centre_of_mass(middle_slices[s,:,:], coords, z=s)
                            #now go through voxels in the mask and see what angle they are at from center of mass. x = lateral, y = posterior
                            mask_voxels = np.where(middle_slices[s,:,:])
                            for idx in range(len(mask_voxels[0])):
                                x = coords[0,s,mask_voxels[0][idx], mask_voxels[1][idx]] - centre[0]
                                y = coords[1,s,mask_voxels[0][idx], mask_voxels[1][idx]] - centre[1]
                                if "l" not in structure.lower(): #want the zero angle to be in lateral direction, so flip for right parotid
                                    x *= -1
                                angle = np.degrees(np.arctan2(y,x))
                                if angle >= 0 and angle <= 120:
                                    subseg_masks["middle"]["p"][s,mask_voxels[0][idx], mask_voxels[1][idx]] = 1
                                if angle <= 0 and angle >= -120:
                                    subseg_masks["middle"]["a"][s,mask_voxels[0][idx], mask_voxels[1][idx]] = 1
                                if angle <= -120 or angle >= 120:
                                    subseg_masks["middle"]["m"][s,mask_voxels[0][idx], mask_voxels[1][idx]] = 1

                        if np.amax(top_slices[s,:,:]) > 0:
                            #get the center of mass of the slice 
                            centre, centre_idx = get_centre_of_mass(top_slices[s,:,:], coords, z=s)
                            #now go through voxels in the mask and see what angle they are at from center of mass. x = lateral, y = posterior
                            mask_voxels = np.where(top_slices[s,:,:])
                            for idx in range(len(mask_voxels[0])):
                                x = coords[0,s,mask_voxels[0][idx], mask_voxels[1][idx]] - centre[0]
                                y = coords[1,s,mask_voxels[0][idx], mask_voxels[1][idx]] - centre[1]
                                if "l" not in structure.lower(): #want the zero angle to be in lateral direction, so flip for right parotid
                                    x *= -1
                                angle = np.degrees(np.arctan2(y,x))
                                if angle >= 0 and angle <= 120:
                                    subseg_masks["top"]["p"][s,mask_voxels[0][idx], mask_voxels[1][idx]] = 1
                                if angle <= 0 and angle >= -120:
                                    subseg_masks["top"]["a"][s,mask_voxels[0][idx], mask_voxels[1][idx]] = 1
                                if angle <= -120 or angle >= 120:
                                    subseg_masks["top"]["m"][s,mask_voxels[0][idx], mask_voxels[1][idx]] = 1
                    # for section in ["bottom", "top", "middle"]:
                    #     img = np.zeros_like(subseg_masks["top"]["m"]).astype(int)
                    #     factor = 10
                    #     for angle in ["p", "a", "m"]:  
                    #         subseg_masks[section][angle] = subseg_masks[section][angle].astype(int) 
                    #         img += subseg_masks[section][angle] * factor
                    #         factor += 10            
                    #     fig, ax = plt.subplots()
                    #     ax.imshow(img[0, :,:])
                    #     ax.set_title(structure)
                    #     ax_slider = plt.axes([0.2,0.01,0.65,0.03], facecolor='green')
                    #     slider = Slider(ax=ax_slider, label="Slice", valmin=0, valmax=98, valstep=1, valinit=0)
                    #     slider.on_changed(update)
                    #     plt.show(block=True)
                    #     plt.close("all")
                    # print(structure)        
                    # plot_3d_image(mask)
                    idx = 1
                    # new_img = np.zeros_like(mask)
                    # for key in subseg_masks:
                    #     for key2 in subseg_masks[key]:
                    #         new_img[subseg_masks[key][key2]] = idx 
                    #         idx += 1       
                    #         plot_3d_image(subseg_masks[key][key2])       
                    # plot_3d_image(new_img)  
                    structure_masks_pet[structure].han_masks = subseg_masks                  
            with open(os.path.join(os.getcwd(), "data", patient_num, "mask_dict"), "wb") as fp:
                pickle.dump(mask_dict, fp)
    elif deblurred == True:
        for patient_num in patient_nums:
            print(f"Getting Han masks for patient {patient_num}")
                
            img_series_path = os.path.join(os.getcwd(), "data_deblur", patient_num, "img_dict")

            with open(img_series_path, "rb") as fp:
                img_dict = pickle.load(fp)
            #load masks 
            with open(os.path.join(os.getcwd(), "data", patient_num, "mask_dict"), "rb") as fp:
                mask_dict = pickle.load(fp)
            structure_masks_pet = mask_dict["PET"]
            img_series_pet = img_dict["PET"]
            with open(os.path.join(os.getcwd(),"predictions",  f"{patient_num}_registered_predictions.txt"), "rb") as fp:
                img_pet, _ = pickle.load(fp)   
            coords = img_series_pet.deconv_coords_2x_img
            #calculate subseg suv avgs
            for structure in structure_masks_pet:
                if not ("par" in structure.lower() and "sup" not in structure.lower()):
                    continue

                mask = structure_masks_pet[structure].whole_roi_masks_deblur

                subseg_masks = {"top": {}, "middle": {}, "bottom": {}}
                for key in subseg_masks:
                    subseg_masks[key]["a"] = np.zeros_like(mask)
                    subseg_masks[key]["m"] = np.zeros_like(mask)
                    subseg_masks[key]["p"] = np.zeros_like(mask)
                
                #dilate image by ~3mm
                mask = binary_dilation(mask).astype(np.uint8)

                #now divide into 3 by length
                slice_min, slice_max = np.min(np.where(mask)[0]),  np.max(np.where(mask)[0])
                third_through = int(0.3333*(slice_max-slice_min)) + slice_min
                two_thirds_through = int(0.6667*(slice_max-slice_min)) + slice_min
                bottom_slices = deepcopy(mask)
                middle_slices = deepcopy(mask)
                top_slices = deepcopy(mask)
                for s in range(mask.shape[0]):
 
                    if np.amax(mask[s,:,:]) == 0:
                        continue
                    elif s < third_through:
                        middle_slices[s,:,:] = 0
                        top_slices[s,:,:] = 0
                    elif s < two_thirds_through:
                        bottom_slices[s,:,:] = 0
                        top_slices[s,:,:] = 0
                    else:
                        middle_slices[s,:,:] = 0
                        bottom_slices[s,:,:] = 0

                # divide axial slices into pie slices
                for s in range(mask.shape[0]):
                    if np.amax(bottom_slices[s,:,:]) > 0:
                        #get the center of mass of the slice 
                        centre, centre_idx = get_centre_of_mass(bottom_slices[s,:,:], coords, z=s)
                        #now go through voxels in the mask and see what angle they are at from center of mass. x = lateral, y = posterior
                        mask_voxels = np.where(bottom_slices[s,:,:])
                        for idx in range(len(mask_voxels[0])):
                            x = coords[0,s,mask_voxels[0][idx], mask_voxels[1][idx]] - centre[0]
                            y = coords[1,s,mask_voxels[0][idx], mask_voxels[1][idx]] - centre[1]
                            if "l" not in structure.lower(): #want the zero angle to be in lateral direction, so flip for right parotid
                                x *= -1
                            angle = np.degrees(np.arctan2(y,x))
                            if angle >= 0 and angle <= 120:
                                subseg_masks["bottom"]["p"][s,mask_voxels[0][idx], mask_voxels[1][idx]] = 1
                            if angle <= 0 and angle >= -120:
                                subseg_masks["bottom"]["a"][s,mask_voxels[0][idx], mask_voxels[1][idx]]= 1
                            if angle <= -120 or angle >= 120:
                                subseg_masks["bottom"]["m"][s,mask_voxels[0][idx], mask_voxels[1][idx]] = 1

                    if np.amax(middle_slices[s,:,:]) > 0:
                        #get the center of mass of the slice 
                        centre, centre_idx = get_centre_of_mass(middle_slices[s,:,:], coords, z=s)
                        #now go through voxels in the mask and see what angle they are at from center of mass. x = lateral, y = posterior
                        mask_voxels = np.where(middle_slices[s,:,:])
                        for idx in range(len(mask_voxels[0])):
                            x = coords[0,s,mask_voxels[0][idx], mask_voxels[1][idx]] - centre[0]
                            y = coords[1,s,mask_voxels[0][idx], mask_voxels[1][idx]] - centre[1]
                            if "l" not in structure.lower(): #want the zero angle to be in lateral direction, so flip for right parotid
                                x *= -1
                            angle = np.degrees(np.arctan2(y,x))
                            if angle >= 0 and angle <= 120:
                                subseg_masks["middle"]["p"][s,mask_voxels[0][idx], mask_voxels[1][idx]] = 1
                            if angle <= 0 and angle >= -120:
                                subseg_masks["middle"]["a"][s,mask_voxels[0][idx], mask_voxels[1][idx]] = 1
                            if angle <= -120 or angle >= 120:
                                subseg_masks["middle"]["m"][s,mask_voxels[0][idx], mask_voxels[1][idx]] = 1

                    if np.amax(top_slices[s,:,:]) > 0:
                        #get the center of mass of the slice 
                        centre, centre_idx = get_centre_of_mass(top_slices[s,:,:], coords, z=s)
                        #now go through voxels in the mask and see what angle they are at from center of mass. x = lateral, y = posterior
                        mask_voxels = np.where(top_slices[s,:,:])
                        for idx in range(len(mask_voxels[0])):
                            x = coords[0,s,mask_voxels[0][idx], mask_voxels[1][idx]] - centre[0]
                            y = coords[1,s,mask_voxels[0][idx], mask_voxels[1][idx]] - centre[1]
                            if "l" not in structure.lower(): #want the zero angle to be in lateral direction, so flip for right parotid
                                x *= -1
                            angle = np.degrees(np.arctan2(y,x))
                            if angle >= 0 and angle <= 120:
                                subseg_masks["top"]["p"][s,mask_voxels[0][idx], mask_voxels[1][idx]] = 1
                            if angle <= 0 and angle >= -120:
                                subseg_masks["top"]["a"][s,mask_voxels[0][idx], mask_voxels[1][idx]] = 1
                            if angle <= -120 or angle >= 120:
                                subseg_masks["top"]["m"][s,mask_voxels[0][idx], mask_voxels[1][idx]] = 1
                # for section in ["bottom", "top", "middle"]:
                #     img = np.zeros_like(subseg_masks["top"]["m"]).astype(int)
                #     factor = 10
                #     for angle in ["p", "a", "m"]:  
                #         subseg_masks[section][angle] = subseg_masks[section][angle].astype(int) 
                #         img += subseg_masks[section][angle] * factor
                #         factor += 10            
                #     fig, ax = plt.subplots()
                #     ax.imshow(img[0, :,:])
                #     ax.set_title(structure)
                #     ax_slider = plt.axes([0.2,0.01,0.65,0.03], facecolor='green')
                #     slider = Slider(ax=ax_slider, label="Slice", valmin=0, valmax=98, valstep=1, valinit=0)
                #     slider.on_changed(update)
                #     plt.show(block=True)
                #     plt.close("all")
                structure_masks_pet[structure].han_masks_deblur = subseg_masks  
                # print(structure)        
                # plot_3d_image(mask)
                # idx = 1
                # new_img = np.zeros_like(mask)
                # for key in subseg_masks:
                #     for key2 in subseg_masks[key]:
                #         new_img[subseg_masks[key][key2]] = idx 
                #         idx += 1         
                # plot_3d_image(new_img)           
            with open(os.path.join(os.getcwd(), "data", patient_num, "mask_dict"), "wb") as fp:
                pickle.dump(mask_dict, fp)                        
def get_centre_of_mass(mask, coords_array, z=None):
    #get the centre of mass of a mask given the length of each index dimension in voxel_sizes.
    if len(mask.shape) == 3:
        mask = mask.astype(int)
        sum_mask = np.sum(mask)
        sum_x = np.sum(mask * coords_array[2,...])
        centre_z = np.sum(mask * coords_array[2,...]) / np.sum(mask)
        centre_y = np.sum(mask * coords_array[1,...]) / np.sum(mask)
        centre_x = np.sum(mask * coords_array[0,...]) / np.sum(mask)

        #get distance from cm to each voxel
        distances = np.sqrt((coords_array[0,...] - centre_x)**2 + (coords_array[1,...] - centre_y)**2 + (coords_array[2,...] - centre_z)**2)

        #get voxel with min dist
        centre_idx = np.unravel_index(np.argmin(distances), distances.shape)

        center_x = coords_array[0,centre_idx[0], centre_idx[1], centre_idx[2]]
        center_y = coords_array[1, centre_idx[0], centre_idx[1], centre_idx[2]]
        center_z = coords_array[2,centre_idx[0], centre_idx[1], centre_idx[2]]
        center = [center_x, center_y, center_z]
    elif len(mask.shape) == 2:
        #need z argument if its just  slice of 3d array
        if z is None:
            raise Exception("Need to provide argument for z if computing center of mass for image slice")
        centre_x = np.sum(mask * coords_array[0,z,...]) / np.sum(mask)
        centre_y = np.sum(mask * coords_array[1,z,...]) / np.sum(mask)
        #get distance from cm to each voxel
        distances = np.sqrt((coords_array[0,z,...] - centre_x)**2 + (coords_array[1,z,...] - centre_y)**2)
        #get voxel with min dist
        centre_idx = np.unravel_index(np.argmin(distances), distances.shape)
        center_x = coords_array[0, z, centre_idx[0], centre_idx[1]]
        center_y = coords_array[1, z, centre_idx[0], centre_idx[1]]
        center_z = z
        center = [center_x, center_y, center_z]
    return center, centre_idx

def importance_vs_buettner(data_folder, deblurred=False):
    #Buettner regions: caudal medial deep, cranio-caudal skewness in deep, superficial lobe
    all_patient_uptakes = {}
    all_patient_uptakes["deep_caudal_medial"] = []
    all_patient_uptakes["deep_caudal_lateral"] = []
    all_patient_uptakes["deep_cranial_medial"] = []
    all_patient_uptakes["deep_cranial_lateral"] = []
    all_patient_uptakes["not_deep_caudal_medial"] = []
    all_patient_uptakes["deep_caudal"] = []
    all_patient_uptakes["deep_cranial"] = []
    all_patient_uptakes["super_caudal"] = []
    all_patient_uptakes["super_cranial"] = []
    all_patient_uptakes["super"] = []
    all_patient_uptakes["deep"] = []
    try:
        2/0
        with open(os.path.join(os.getcwd(), "stats", f"avg_uptake_buettner_deblurred_{deblurred}"), "rb") as fp:
            all_patient_uptakes =pickle.load(fp)    
    except:
        if deblurred == False:
            patient_nums = os.listdir(data_folder)
            patient_nums.sort()
            for patient_num in patient_nums:
                print(f"Calculating uptakes for patient {patient_num}")
                patient_folder = os.path.join(data_folder, patient_num)
                    
                img_series_path = os.path.join(os.getcwd(), "data", patient_num, "img_dict")

                with open(img_series_path, "rb") as fp:
                    img_dict = pickle.load(fp)
                #load masks 
                with open(os.path.join(os.getcwd(), "data", patient_num, "mask_dict"), "rb") as fp:
                    mask_dict = pickle.load(fp)
                structure_masks_pet = mask_dict["PET"]
                img_series_pet = img_dict["PET"]
                suv_factors = img_series_pet.suv_factors
                img_pet = img_series_pet.image_array * suv_factors[1]   #lbm suv     


                #calculate subseg suv avgs
                for structure in structure_masks_pet:
                    if "par" in structure.lower() and "sup" not in structure.lower():
                        super_mask = structure_masks_pet[structure].superficial_masks
                        deep_mask = structure_masks_pet[structure].deep_masks
                        whole_mask = structure_masks_pet[structure].whole_roi_masks
                        whole_mask = binary_dilation(binary_erosion(whole_mask, structure=np.ones((2,2,2))), structure= np.ones((2,2,2)))
                        super_mask = binary_dilation(binary_erosion(super_mask, structure=np.ones((2,2,2))), structure= np.ones((2,2,2)))
                        deep_mask = binary_dilation(binary_erosion(deep_mask, structure=np.ones((2,2,2))), structure= np.ones((2,2,2)))
                        #first can simply save superficial and deep doses
                        #first can simply save superficial and deep doses
                        all_patient_uptakes["deep"].append(np.nanmean(img_pet[deep_mask]))
                        all_patient_uptakes["super"].append(np.nanmean(img_pet[super_mask]))

                        #now need to divide the deep lobe into the cranial and caudal parts, and further into the medial caudal part.
                        mask_indices_deep = np.nonzero(deep_mask)
                        min_z = np.min(mask_indices_deep[0])
                        max_z = np.max(mask_indices_deep[0])
                        min_y = np.min(mask_indices_deep[1])
                        max_y = np.max(mask_indices_deep[1])
                        min_x = np.min(mask_indices_deep[2])
                        max_x = np.max(mask_indices_deep[2])
                        height = max_z - min_z + 1
                        width_y = max_y - min_y + 1
                        width_x = max_x - min_x + 1 

                        deep_caudal_mask = np.zeros_like(deep_mask)
                        deep_cranial_mask = np.zeros_like(deep_mask)
                        deep_caudal_medial_mask = np.zeros_like(deep_mask)
                        deep_caudal_lateral_mask = np.zeros_like(deep_mask)
                        deep_cranial_medial_mask = np.zeros_like(deep_mask)
                        deep_cranial_lateral_mask = np.zeros_like(deep_mask)

                        deep_caudal_mask[min_z:min_z + int(height/2), min_y:max_y, min_x:max_x] = deep_mask[min_z:min_z + int(height/2), min_y:max_y, min_x:max_x]
                        deep_cranial_mask[min_z + int(height/2):max_z, min_y:max_y, min_x:max_x] = deep_mask[min_z + int(height/2):max_z, min_y:max_y, min_x:max_x]

                        if "l" in structure.lower():
                            deep_caudal_medial_mask[min_z:max_z, min_y:max_y, min_x:min_x+int(width_x/2)] = deep_caudal_mask[min_z:max_z, min_y:max_y, min_x:min_x + int(width_x/2)]
                            deep_caudal_lateral_mask[min_z:max_z, min_y:max_y, min_x+int(width_x/2):max_x] = deep_caudal_mask[min_z:max_z, min_y:max_y, min_x+int(width_x/2):max_x]
                            deep_cranial_medial_mask[min_z:max_z, min_y:max_y, min_x:min_x+int(width_x/2)] = deep_cranial_mask[min_z:max_z, min_y:max_y, min_x:min_x+int(width_x/2)]
                            deep_cranial_lateral_mask[min_z:max_z, min_y:max_y, min_x+int(width_x/2):max_x] = deep_cranial_mask[min_z:max_z, min_y:max_y, min_x+int(width_x/2):max_x]
                        else:
                            deep_caudal_lateral_mask[min_z:max_z, min_y:max_y, min_x:min_x+int(width_x/2)] = deep_caudal_mask[min_z:max_z, min_y:max_y, min_x:min_x+int(width_x/2)]
                            deep_caudal_medial_mask[min_z:max_z, min_y:max_y, min_x+int(width_x/2):max_x] = deep_caudal_mask[min_z:max_z, min_y:max_y, min_x+int(width_x/2):max_x]
                            deep_cranial_lateral_mask[min_z:max_z, min_y:max_y, min_x:min_x+int(width_x/2)] = deep_cranial_mask[min_z:max_z, min_y:max_y, min_x:min_x+int(width_x/2)]
                            deep_cranial_medial_mask[min_z:max_z, min_y:max_y, min_x+int(width_x/2):max_x] = deep_cranial_mask[min_z:max_z, min_y:max_y, min_x+int(width_x/2):max_x]        
                        
                        #also get superficial cranial / caudal 
                        mask_indices_super = np.nonzero(super_mask)
                        min_z = np.min(mask_indices_super[0])
                        max_z = np.max(mask_indices_super[0])
                        min_x = np.min(mask_indices_super[2])
                        max_x = np.max(mask_indices_super[2])
                        height = max_z - min_z + 1

                        width_x = max_x - min_x + 1 

                        super_caudal_mask = np.zeros_like(super_mask)
                        super_cranial_mask = np.zeros_like(super_mask)     
                        super_caudal_mask[min_z:min_z + int(height/2), min_y:max_y, min_x:max_x] = super_mask[min_z:min_z + int(height/2), min_y:max_y, min_x:max_x]
                        super_cranial_mask[min_z + int(height/2):max_z, min_y:max_y, min_x:max_x] = super_mask[min_z + int(height/2):max_z, min_y:max_y, min_x:max_x]

                        all_patient_uptakes["deep"].append(np.nanmax(img_pet[deep_mask]))
                        all_patient_uptakes["super"].append(np.nanmax(img_pet[super_mask]))
                        try:
                            all_patient_uptakes["deep_caudal_medial"].append(np.nanmax(img_pet[deep_caudal_medial_mask]))
                            all_patient_uptakes["deep_caudal_lateral"].append(np.nanmax(img_pet[deep_caudal_lateral_mask]))
                        except: 
                            pass
                        all_patient_uptakes["deep_cranial_medial"].append(np.nanmax(img_pet[deep_cranial_medial_mask]))
                        all_patient_uptakes["deep_cranial_lateral"].append(np.nanmax(img_pet[deep_cranial_lateral_mask]))
                        all_patient_uptakes["deep_caudal"].append(np.nanmax(img_pet[deep_caudal_mask]))
                        all_patient_uptakes["deep_cranial"].append(np.nanmax(img_pet[deep_cranial_mask]))
                        all_patient_uptakes["super_caudal"].append(np.nanmax(img_pet[super_caudal_mask]))
                        all_patient_uptakes["super_cranial"].append(np.nanmax(img_pet[super_cranial_mask]))

            with open(os.path.join(os.getcwd(), "stats", f"avg_uptake_buettner_deblurred_{deblurred}"), "wb") as fp:
                pickle.dump(all_patient_uptakes, fp)
        elif deblurred == True:
            patient_nums = os.listdir(data_folder)
            patient_nums.sort()
            for patient_num in patient_nums:
                print(f"Calculating uptakes for patient {patient_num}")
                patient_folder = os.path.join(data_folder, patient_num)
                    
                img_series_path = os.path.join(os.getcwd(), "data_deblur", patient_num, "img_dict")

                with open(img_series_path, "rb") as fp:
                    img_dict = pickle.load(fp)
                #load masks 
                with open(os.path.join(os.getcwd(), "data", patient_num, "mask_dict"), "rb") as fp:
                    mask_dict = pickle.load(fp)
                structure_masks_pet = mask_dict["PET"]
                img_series_pet = img_dict["PET"]
                with open(os.path.join(os.getcwd(),"predictions",  f"{patient_num}_registered_predictions.txt"), "rb") as fp:
                    img_pet, _ = pickle.load(fp)     


                #calculate subseg suv avgs
                for structure in structure_masks_pet:
                    if "par" in structure.lower() and "sup" not in structure.lower():
                        super_mask = structure_masks_pet[structure].superficial_masks_deblur
                        deep_mask = structure_masks_pet[structure].deep_masks_deblur
                        whole_mask = structure_masks_pet[structure].whole_roi_masks_deblur
                        whole_mask = binary_dilation(binary_erosion(whole_mask, structure=np.ones((2,2,2))), structure= np.ones((2,2,2)))
                        super_mask = binary_dilation(binary_erosion(super_mask, structure=np.ones((2,2,2))), structure= np.ones((2,2,2)))
                        deep_mask = binary_dilation(binary_erosion(deep_mask, structure=np.ones((2,2,2))), structure= np.ones((2,2,2)))
                        #first can simply save superficial and deep doses
                        

                        #now need to divide the deep lobe into the cranial and caudal parts, and further into the medial caudal part.
                        mask_indices_deep = np.nonzero(deep_mask)
                        min_z = np.min(mask_indices_deep[0])
                        max_z = np.max(mask_indices_deep[0])
                        min_y = np.min(mask_indices_deep[1])
                        max_y = np.max(mask_indices_deep[1])
                        min_x = np.min(mask_indices_deep[2])
                        max_x = np.max(mask_indices_deep[2])
                        height = max_z - min_z + 1
     
                        width_x = max_x - min_x + 1 

                        deep_caudal_mask = np.zeros_like(deep_mask)
                        deep_cranial_mask = np.zeros_like(deep_mask)
                        deep_caudal_medial_mask = np.zeros_like(deep_mask)
                        deep_caudal_lateral_mask = np.zeros_like(deep_mask)
                        deep_cranial_medial_mask = np.zeros_like(deep_mask)
                        deep_cranial_lateral_mask = np.zeros_like(deep_mask)

                        deep_caudal_mask[min_z:min_z + int(height/2), min_y:max_y, min_x:max_x] = deep_mask[min_z:min_z + int(height/2), min_y:max_y, min_x:max_x]
                        deep_cranial_mask[min_z + int(height/2):max_z, min_y:max_y, min_x:max_x] = deep_mask[min_z + int(height/2):max_z, min_y:max_y, min_x:max_x]

                        if "l" in structure.lower():
                            deep_caudal_medial_mask[min_z:max_z, min_y:max_y, min_x:min_x+int(width_x/2)] = deep_caudal_mask[min_z:max_z, min_y:max_y, min_x:min_x + int(width_x/2)]
                            deep_caudal_lateral_mask[min_z:max_z, min_y:max_y, min_x+int(width_x/2):max_x] = deep_caudal_mask[min_z:max_z, min_y:max_y, min_x+int(width_x/2):max_x]
                            deep_cranial_medial_mask[min_z:max_z, min_y:max_y, min_x:min_x+int(width_x/2)] = deep_cranial_mask[min_z:max_z, min_y:max_y, min_x:min_x+int(width_x/2)]
                            deep_cranial_lateral_mask[min_z:max_z, min_y:max_y, min_x+int(width_x/2):max_x] = deep_cranial_mask[min_z:max_z, min_y:max_y, min_x+int(width_x/2):max_x]
                        else:
                            deep_caudal_lateral_mask[min_z:max_z, min_y:max_y, min_x:min_x+int(width_x/2)] = deep_caudal_mask[min_z:max_z, min_y:max_y, min_x:min_x+int(width_x/2)]
                            deep_caudal_medial_mask[min_z:max_z, min_y:max_y, min_x+int(width_x/2):max_x] = deep_caudal_mask[min_z:max_z, min_y:max_y, min_x+int(width_x/2):max_x]
                            deep_cranial_lateral_mask[min_z:max_z, min_y:max_y, min_x:min_x+int(width_x/2)] = deep_cranial_mask[min_z:max_z, min_y:max_y, min_x:min_x+int(width_x/2)]
                            deep_cranial_medial_mask[min_z:max_z, min_y:max_y, min_x+int(width_x/2):max_x] = deep_cranial_mask[min_z:max_z, min_y:max_y, min_x+int(width_x/2):max_x]   
                        
                        #also get superficial cranial / caudal 
                        mask_indices_super = np.nonzero(super_mask)
                        min_z = np.min(mask_indices_super[0])
                        max_z = np.max(mask_indices_super[0])
                        min_x = np.min(mask_indices_super[2])
                        max_x = np.max(mask_indices_super[2])
                        height = max_z - min_z + 1

                        width_x = max_x - min_x + 1 

                        super_caudal_mask = np.zeros_like(super_mask)
                        super_cranial_mask = np.zeros_like(super_mask)     
                        super_caudal_mask[min_z:min_z + int(height/2), min_y:max_y, min_x:max_x] = super_mask[min_z:min_z + int(height/2), min_y:max_y, min_x:max_x]
                        super_cranial_mask[min_z + int(height/2):max_z, min_y:max_y, min_x:max_x] = super_mask[min_z + int(height/2):max_z, min_y:max_y, min_x:max_x]
                        # plot_3d_image(super_mask)
                        # plot_3d_image(deep_mask)
                        # plot_3d_image(deep_caudal_mask)
                        # plot_3d_image(deep_cranial_mask)
                        # plot_3d_image(deep_cranial_lateral_mask)
                        # plot_3d_image(deep_cranial_medial_mask)
                        # plot_3d_image(deep_caudal_lateral_mask)
                        # plot_3d_image(deep_caudal_medial_mask)
                        all_patient_uptakes["deep"].append(np.nanmedian(img_pet[deep_mask]))
                        all_patient_uptakes["super"].append(np.nanmedian(img_pet[super_mask]))
                        all_patient_uptakes["deep"].append(np.nanmedian(img_pet[deep_mask]))
                        all_patient_uptakes["super"].append(np.nanmedian(img_pet[super_mask]))
                        all_patient_uptakes["deep_caudal_medial"].append(np.nanmedian(img_pet[deep_caudal_medial_mask]))
                        all_patient_uptakes["deep_caudal_lateral"].append(np.nanmedian(img_pet[deep_caudal_lateral_mask]))
                        all_patient_uptakes["deep_cranial_medial"].append(np.nanmedian(img_pet[deep_cranial_medial_mask]))
                        all_patient_uptakes["deep_cranial_lateral"].append(np.nanmedian(img_pet[deep_cranial_lateral_mask]))
                        all_patient_uptakes["not_deep_caudal_medial"].append(np.nanmedian(img_pet[np.logical_and(deep_mask, deep_caudal_lateral_mask)]))
                        all_patient_uptakes["deep_caudal"].append(np.nanmedian(img_pet[deep_caudal_mask]))
                        all_patient_uptakes["deep_cranial"].append(np.nanmedian(img_pet[deep_cranial_mask]))
                        all_patient_uptakes["super_caudal"].append(np.nanmedian(img_pet[super_caudal_mask]))
                        all_patient_uptakes["super_cranial"].append(np.nanmedian(img_pet[super_cranial_mask]))
            with open(os.path.join(os.getcwd(), "stats", f"avg_uptake_buettner_deblurred_{deblurred}"), "wb") as fp:
                pickle.dump(all_patient_uptakes, fp)
    patient_uptakes = {}
    #now get average over patients
    patient_uptakes["deep_caudal_medial"] = [np.nanmean(all_patient_uptakes["deep_caudal_medial"]), np.nanstd(all_patient_uptakes["deep_caudal_medial"])]
    patient_uptakes["deep_caudal_lateral"]= [np.nanmean(all_patient_uptakes["deep_caudal_lateral"]), np.nanstd(all_patient_uptakes["deep_caudal_lateral"])]
    patient_uptakes["deep_cranial_medial"]= [np.nanmean(all_patient_uptakes["deep_cranial_medial"]), np.nanstd(all_patient_uptakes["deep_cranial_medial"])]
    patient_uptakes["deep_cranial_lateral"]= [np.nanmean(all_patient_uptakes["deep_cranial_lateral"]), np.nanstd(all_patient_uptakes["deep_cranial_lateral"])]
    patient_uptakes["deep_caudal"] = [np.nanmean(all_patient_uptakes["deep_caudal"]), np.nanstd(all_patient_uptakes["deep_caudal"])]
    patient_uptakes["deep_cranial"] = [np.nanmean(all_patient_uptakes["deep_cranial"]), np.nanstd(all_patient_uptakes["deep_cranial"])]
    patient_uptakes["super_caudal"] = [np.nanmean(all_patient_uptakes["super_caudal"]), np.nanstd(all_patient_uptakes["super_caudal"])]
    patient_uptakes["super_cranial"] = [np.nanmean(all_patient_uptakes["super_cranial"]), np.nanstd(all_patient_uptakes["super_cranial"])]
    patient_uptakes["deep"] = [np.nanmean(all_patient_uptakes["deep"]), np.nanstd(all_patient_uptakes["deep"])]
    patient_uptakes["super"] = [np.nanmean(all_patient_uptakes["super"]), np.nanstd(all_patient_uptakes["super"])]

    #for this region.. not enough subsegs to test a correlation. so instead 
    t_sd, p_sd = ttest_rel(all_patient_uptakes["super"], all_patient_uptakes["deep"])
    t_cc, p_cc = ttest_rel(all_patient_uptakes["deep_caudal"], all_patient_uptakes["deep_cranial"])
    t_cmd, p_cmd = ttest_rel(all_patient_uptakes["deep_caudal_medial"], all_patient_uptakes["deep_caudal_lateral"])
    t_cms, p_cms = ttest_rel(all_patient_uptakes["super_caudal"], all_patient_uptakes["super_cranial"])
    return


    return center, centre_idx    

def make_best_model_and_fs_plots():

    best_data_all = {'rf': [], 'kr': [], 'svm': [], 'lin_reg': [], "cit": [], 'use_n_most_correlated': [], 'pca': [], 'lincom': []}
    fold_results_dir = os.path.join(os.getcwd(),"importance_comparison", f"fold_results_bw_deblurred_True")
    for outer_fold in range(9):
        if not os.path.exists(fold_results_dir):
            os.mkdir(fold_results_dir)
        fold_results_path = os.path.join(fold_results_dir, str(outer_fold))
        with open(fold_results_path, "rb") as fp:
            results = pickle.load(fp)  
        import training
        best_data = training.get_fs_and_model_performances(results) 

        
    #     for fold in results:
    #         for key in best_data:
    #             best_data[key].append(1000)
    #         for combo in results[fold]:
    #             mae = combo[0][0]
    #             model = combo[0][1]
    #             fs = combo[0][2]
    #             if mae < best_data[model][-1]:
    #                 best_data[model][-1] = mae
    #             if mae < best_data[fs][-1]:
    #                 best_data[fs][-1] = mae
            
        for key in best_data:
            best_data_all[key].append(best_data[key])
    for key in best_data_all:
        best_data_all[key] = [np.mean(best_data_all[key]), np.std(best_data_all[key])]

    #now make a bar chart for each 
    model_types = ['R.F', 'K.R.', 'S.V.M.', "Lin. Reg.", "C.I.T"]
    model_scores = [best_data_all['rf'][0], best_data_all['kr'][0], best_data_all['svm'][0], best_data_all['lin_reg'][0], best_data_all['cit'][0]]
    model_stds = [best_data_all['rf'][1], best_data_all['kr'][1], best_data_all['svm'][1], best_data_all['lin_reg'][1], best_data_all['cit'][1]]

    fs_types = ["P.C.F", "P.C.A", "LinCom"]
    fs_scores = [best_data_all['use_n_most_correlated'][0], best_data_all["pca"][0], best_data_all["lincom"][0]]
    fs_stds = [best_data_all['use_n_most_correlated'][1], best_data_all["pca"][1], best_data_all["lincom"][1]]

    fig, ax = plt.subplots(1,2)
    x_pos_model = list(range(5))
    x_pos_fs = list(range(3))
    colors_model = ['tomato', 'khaki', "indianred", "mediumseagreen", "lightseagreen"]
    ax[0].bar(x_pos_model, model_scores, yerr=model_stds, align='center', alpha=0.7, edgecolor='k', ecolor='black', capsize=10, color=colors_model)
    ax[0].set_xticks(ticks=x_pos_model, labels=model_types)

    ax[1].bar(x_pos_fs, fs_scores, yerr=fs_stds, align='center', alpha=0.7,edgecolor='k', ecolor='black', capsize=10, color=colors_model[0:3])
    ax[1].set_xticks(ticks=x_pos_fs, labels=fs_types)

    ax[0].set_ylabel("Mean Absolute Error", fontsize=15)
    ax[1].set_ylabel("Mean Absolute Error", fontsize=15)
    ax[0].set_title("Model Performance", fontsize=15)
    ax[1].set_title("Feature Selection Algorithm Performance", fontsize=15)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].tick_params(axis='both', which='both', labelsize=12)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].tick_params(axis='both', which='both', labelsize=12)
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, y: f'{x:.2f}'))
    plt.show(block=True)
    return



if __name__ == "__main__":
    main()