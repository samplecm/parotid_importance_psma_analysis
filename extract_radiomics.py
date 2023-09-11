#this code was used for model training as in the article 

"PSMA PET as a predictive tool for sub-regional importance estimates in the parotid gland"

#By Caleb Sample, Arman Rahmim, Francois Benard, Jonn Wu, Haley Clark.



import pydicom 
import radiomics
import os
import dicom_to_nrrd
import six
import glob
import nrrd
import pickle
import shutil
import numpy as np
import copy 
import SimpleITK as sitk
import math

def extract_feature_maps_masked():
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
            if not os.path.exists(save_path):
                os.mkdir(save_path)              
            else:
                #delte current files in dir
                for file in os.listdir(save_path):
                    os.remove(os.path.join(save_path, file))
            #pet and ct nrrd files will be in separate subfolders
            path_nrrd = os.path.join(os.getcwd(), "nrrd_files", patient_num)

            path_nrrd = os.path.join(path_nrrd, modality)       
            img_nrrd_path = os.path.join(path_nrrd, "images")
            structures_nrrd_path = os.path.join(path_nrrd, "masks")



        
            for img_path in glob.glob(os.path.join(img_nrrd_path,"*")):
                if "bw" in img_path or "bsa" in img_path:
                    continue #only calculate for lbm image (pet)
                for struct_file in os.listdir(structures_nrrd_path):
                    structure_name = struct_file.split("__")[1]
                    segmentation_name = struct_file.split("__")[2]
                    if "whole" not in segmentation_name:
                        continue #dont calculate for subsegment masks
                    mask_path = os.path.join(structures_nrrd_path, struct_file)
                    img = sitk.ReadImage(img_path)
                    #mask, header_mask = nrrd.read(mask_path)

                    params = os.path.join(os.getcwd(), params)   #no shape

                    #calculate optimal number of histogram bins to use using sturges rule
                    mask, _ = nrrd.read(mask_path)
                    n = np.count_nonzero(mask)
                    bins = math.ceil(np.log2(n)+1)
                    #extract features 
                    print(f'Calculating feature maps for {structure_name} of patient {patient_num} with modality {modality}')
                    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(params, binCount=bins, voxelBatch=1000, maskedKernel=True, kernelRadius=2)
                    result = extractor.execute(img_path, mask_path, voxelBased=True)   
                    for key, val in six.iteritems(result):
                        if isinstance(val, sitk.Image):
                            #reshape the mask to be same size as image arrays
                            val = sitk.Resample(val,img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, val.GetPixelID())
                            sitk.WriteImage(val, os.path.join(save_path, str(key + "__" + structure_name + "__" + modality + '__.nrrd')), True)
                            print(f'Stored feature {key} for {structure_name} of patient {patient_num}')

                    del result  
            print("")      
def extract_feature_maps_whole():
    print("Starting whole extraction...")
    params = "params_feat_maps.yaml"
    data_folder = os.path.join(os.getcwd(), "SG_PETRT")
    patient_nums = os.listdir(data_folder)
    patient_nums.sort()
    for patient_num in patient_nums[1:]:
        
        for modality in ["CT"]:      
            save_path = os.path.join(os.getcwd(), "feature_maps", patient_num)
            
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            save_path = os.path.join(save_path, modality)    
            if not os.path.exists(save_path):
                os.mkdir(save_path)              
            else:
                #delte current files in dir
                for file in os.listdir(save_path):
                    os.remove(os.path.join(save_path, file))
            #pet and ct nrrd files will be in separate subfolders
            path_nrrd = os.path.join(os.getcwd(), "nrrd_files", patient_num)

            path_nrrd = os.path.join(path_nrrd, modality)       
            img_nrrd_path = os.path.join(path_nrrd, "images")




        
            for img_path in glob.glob(os.path.join(img_nrrd_path,"*")):
                if "bw" in img_path or "bsa" in img_path:
                    continue
                img = sitk.ReadImage(img_path)
                img = sitk.GetArrayFromImage(img)[50:55,75:-200,140:-150]
                from utils import plot_3d_image
                # plot_3d_image(img)
                mask = copy.deepcopy(img)
                img = sitk.GetImageFromArray(img)
                mask = mask > -1000
                #mask, header_mask = nrrd.read(mask_path)
                mask = sitk.GetImageFromArray(mask.astype(int))
                temp_img_path = os.path.join(os.getcwd(), "cache", "img.nrrd")
                temp_mask_path = os.path.join(os.getcwd(), "cache", "mask.nrrd")
                sitk.WriteImage(mask, temp_mask_path)
                sitk.WriteImage(img, temp_img_path)
                params = os.path.join(os.getcwd(), params)   #no shape

                #calculate optimal number of histogram bins to use using sturges rule
                bin_count = 25
                #extract features 
                print(f'Calculating feature maps for patient {patient_num} with modality {modality}')
                extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(params, binCount=bin_count, voxelBatch=7500, maskedKernel=True, kernelRadius=2)
                result = extractor.execute(temp_img_path,temp_mask_path, voxelBased=True)   
                for key, val in six.iteritems(result):
                    if isinstance(val, sitk.Image):
                        #reshape the mask to be same size as image arrays
                        val = sitk.Resample(val,img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, val.GetPixelID())
                        sitk.WriteImage(val, os.path.join(save_path, str(key  + "__" + modality + '__.nrrd')), True)
                        print(f'Stored feature {key} for patient {patient_num}')

                del result            


def main(deblurred=False):
    subsegmentation = [2,1,2]

    data_folder = os.path.join(os.getcwd(), "SG_PETRT")

    patient_nums = os.listdir(data_folder)
    patient_nums.sort()
    for patient_num in patient_nums:
        patient_folder = os.path.join(data_folder, patient_num)
            
        if deblurred == False:
            save_path = os.path.join(os.getcwd(), "radiomics", patient_num)
        elif deblurred == True:
            save_path = os.path.join(os.getcwd(), "radiomics_deblurred", patient_num)

        if not os.path.exists(save_path):
            os.mkdir(save_path)
       


        image_folders = os.listdir(patient_folder)    #either ct or pet
        image_folders.sort(reverse=True)


        features_dict = {}
        for image_folder in image_folders:
            files_folder = os.path.join(patient_folder, image_folder)
            if not os.path.isdir(files_folder):
                continue
            if "ct" in image_folder.lower():
                modality = "CT"
            elif "pet" in image_folder.lower():
                modality = "PET" 
                #load the patient attributes to add as model features
                patient_attribute_features = get_patient_attribute_feature_dict(os.path.join(files_folder, os.listdir(files_folder)[0]))
            else:
                continue    
                
            #pet and ct nrrd files will be in separate subfolders
            if deblurred == False:
                save_path_nrrd = os.path.join(os.getcwd(), "nrrd_files", patient_num)
            else:
                save_path_nrrd = os.path.join(os.getcwd(), "nrrd_files_deblurred", patient_num)
            if not os.path.exists(save_path_nrrd):
                os.mkdir(save_path_nrrd) 
            save_path_nrrd = os.path.join(save_path_nrrd, modality)       
            if not os.path.exists(save_path_nrrd):              
                os.mkdir(save_path_nrrd)
                os.mkdir(os.path.join(save_path_nrrd, "images"))
                os.mkdir(os.path.join(save_path_nrrd, "masks"))
            img_nrrd_path = os.path.join(save_path_nrrd, "images")
            structures_nrrd_path = os.path.join(save_path_nrrd, "masks")
            save_path_nrrd = [img_nrrd_path, structures_nrrd_path]


            print(f"Creating nrrd files for patient {patient_num}...")
            #now want radiomics data using all these images and the structure file.    
            dicom_to_nrrd.convert_all_dicoms_to_nrrd(modality, save_paths=save_path_nrrd,  patient_num=patient_num, deblurred=deblurred)
            #now want to save radiomics features
            features_dict[modality] = {}

            for img_path in glob.glob(os.path.join(img_nrrd_path,"*")):
                for struct_file in os.listdir(structures_nrrd_path):
                    structure_name = struct_file.split("__")[1]
                    segmentation_name = struct_file.split("__")[2]

                    mask_path = os.path.join(structures_nrrd_path, struct_file)
                    #img, header_img = nrrd.read(img_path)
                    #mask, header_mask = nrrd.read(mask_path)

                    params_pet_orig = os.path.join(os.getcwd(), "params_pet_orig.yaml")   
                    params_ct_orig = os.path.join(os.getcwd(), "params_ct_orig.yaml")  
                    params_pet_wavelet = os.path.join(os.getcwd(), "params_pet_wavelet.yaml")   
                    params_ct_wavelet = os.path.join(os.getcwd(), "params_ct_wavelet.yaml") 
                    params_pet_square = os.path.join(os.getcwd(), "params_pet_square.yaml")   
                    params_ct_square = os.path.join(os.getcwd(), "params_ct_square.yaml") 
                    params_pet_sqrt = os.path.join(os.getcwd(), "params_pet_sqrt.yaml")   
                    params_ct_sqrt = os.path.join(os.getcwd(), "params_ct_sqrt.yaml") 
 

                    if modality == "PET":
                        params_orig = params_pet_orig
                        params_wavelet = params_pet_wavelet
                        params_square = params_pet_square
                        params_sqrt = params_pet_sqrt
                    elif modality == "CT":
                        params_orig = params_ct_orig
                        params_wavelet = params_ct_wavelet
                        params_square = params_ct_square
                        params_sqrt = params_ct_sqrt

                    features = extract_radiomics(img_path, mask_path, params=params_orig)

                    features.update(extract_radiomics(img_path, mask_path, params=params_square))
                    features.update(extract_radiomics(img_path, mask_path, params=params_sqrt))
                    wavelet_features = extract_radiomics(img_path, mask_path, params=params_wavelet)
                    for key in list(wavelet_features.keys()):
                        if "LLL" not in key and "HHH" not in key:
                            del wavelet_features[key]
                    features.update(wavelet_features)

                    

                    if modality == "PET":
                        suv_factor = img_path.split("__")[1]
                        if suv_factor not in features_dict[modality]:
                            features_dict[modality][suv_factor] = {}
                        if structure_name not in features_dict[modality][suv_factor]:
                            features_dict[modality][suv_factor][structure_name] =  {}
                        features_dict[modality][suv_factor][structure_name][segmentation_name] = features      
                        print(f"Calculated {modality} radiomics features for subsegment {segmentation_name} of {structure_name} using {suv_factor}")    

                    else:
                        if structure_name not in features_dict[modality]:
                            features_dict[modality][structure_name] =  {}
                        features_dict[modality][structure_name][segmentation_name] = features      
                        print(f"Calculated {modality} radiomics features for subsegment {segmentation_name} of {structure_name}")
        #now add the patient attribute features
        features_dict["patient_attributes"] = patient_attribute_features
        
        with open(os.path.join(save_path, "features_dict"), "wb") as fp:
            pickle.dump(features_dict, fp)

             
def get_patient_attribute_feature_dict(struct_path):
    #extract the patient features from dicom data that can be used as model features:
    data = pydicom.dcmread(struct_path)
    sex = data[0x0010,0x0040].value
    if sex == "F":
        sex = 0.
    else:
        sex = 1.    

    height = float(data[0x0010,0x1020].value)
    age = float(data[0x0010, 0x1010].value[1:3])
    weight = float(data[0x0010, 0x1030].value)


    features = {}
    features['height'] = height
    features['weight'] = weight
    features['sex'] = sex
    features['age'] = age

    return features
def load_design_matrix(try_load=False, normalize_by_whole=True, suv_type="bw", deblurred=False):
    #suv can be scaled by body weight bw, body surface area bsa, or lean body mass lbm


    data = {}
    if deblurred == False:
        spatial_folder = os.path.join(os.getcwd(), "radiomics")
    elif deblurred == True:
        spatial_folder = os.path.join(os.getcwd(), "radiomics_deblurred")  
    feature_names_spatial = []
    feature_names_spatial_whole = []
    feature_names_spatial_ct = []
    feature_names_spatial_whole_ct = []
    feature_names_patient_attributes = []


    #get the patient attribute features 
    data["patient_attributes"] = []   
    for patient in os.listdir(spatial_folder):
        data["patient_attributes"].append([])    
        with open(os.path.join(spatial_folder, patient, "features_dict") , "rb") as fp:     
            rad_dict = pickle.load(fp)     #load the patients radiomics dictionary 
        patient_attributes = rad_dict["patient_attributes"]
        for feature_name in patient_attributes:
            feature_val = patient_attributes[feature_name]
            if feature_name not in feature_names_patient_attributes:
                feature_names_patient_attributes.append(feature_name)
            data["patient_attributes"][-1].append(feature_val)      

    
    for modality in ["CT", "PET"]:
        data[modality] = {}
        data[modality]["spatial"] = []
        data[modality]["abs"] = []
        data[modality]["phase"] = []
        for i in range(19):    #make a separate list for each subseg for now
            data[modality]["spatial"].append([])
            data[modality]["abs"].append([])
            data[modality]["phase"].append([]) 


        #get the spatial data
        for patient in os.listdir(spatial_folder):
            with open(os.path.join(spatial_folder, patient, "features_dict") , "rb") as fp:     
                rad_dict = pickle.load(fp)     #load the patients radiomics dictionary 
            for par in rad_dict['CT']: #not using CT, just using to list parotids
                if modality == "CT":
                    par_dict = rad_dict[modality][par]
                elif modality == "PET":
                    par_dict = rad_dict[modality][suv_type][par]
  
                     
                for roi_type in par_dict:    #go through all subsegments

                    if roi_type == "whole":
                        data[modality]["spatial"][18].append([])
                        for feature_name in par_dict[roi_type]:
                            if "diagnostics" in feature_name:
                                continue #not a feature
                            feature_val = par_dict[roi_type][feature_name]
                            if modality == "PET" and feature_name not in feature_names_spatial_whole:
                                feature_names_spatial_whole.append(feature_name)   
                            elif modality == "CT" and feature_name not in feature_names_spatial_whole_ct:
                                feature_names_spatial_whole_ct.append(feature_name)       
                            if hasattr(feature_val, "__len__"):
                                feature_val = float(feature_val)
                            data[modality]["spatial"][18][-1].append(feature_val)  
                    else:
                        subseg_num = int(roi_type)
                        data[modality]["spatial"][subseg_num].append([]) #new subseg data row 
                        for feature_name in par_dict[roi_type]:
                            if "diagnostics" in feature_name or "shape" in feature_name:
                                continue #not a feature
                            if modality == "PET" and feature_name not in feature_names_spatial:
                                feature_names_spatial.append(feature_name)   
                            elif modality == "CT" and feature_name not in feature_names_spatial_ct:
                                feature_names_spatial_ct.append(feature_name)  
                            feature_val = par_dict[roi_type][feature_name]
                            if hasattr(feature_val, "__len__"):
                                feature_val = float(feature_val)      
                            data[modality]["spatial"][subseg_num][-1].append(feature_val)
    final_data = {}
    importance_vals = [0.751310670731707,  0.526618902439024,   0.386310975609756,
            1,   0.937500000000000,   0.169969512195122,   0.538871951219512 ,  0.318064024390244,   0.167751524390244,
            0.348320884146341,   0.00611608231707317, 0.0636128048780488,  0.764222560975610,   0.0481192835365854,  0.166463414634146,
            0.272984146341463,   0.0484897103658537,  0.035493902439024]
    #now convert these to numpy arrays 
    for modality in ["PET", "CT"]:
        final_data[modality] = {}
        for img_type in ["spatial"]:#, "abs", "phase"]:
            final_data[modality][img_type] = {} 
            for s in range(len(data[modality][img_type][0:18])):
                for r in range(len(data[modality][img_type][s])):
                    data[modality][img_type][s][r].append(importance_vals[s])    #add the importance to the final column of the matrices (y in training)
                data[modality][img_type][s] = np.array(data[modality][img_type][s])   
            type_combined_whole = np.array(data[modality][img_type][18])
            num_features = data[modality][img_type][0].shape[1]
            type_combined = np.empty([0, num_features])
            type_combined_ratios = np.empty([0, num_features])
            for i in range(18):
                type_combined = np.vstack((type_combined, data[modality][img_type][i]))    
                for j in range(data[modality][img_type][i].shape[0]):
                    #get rid of zeros in the whole array to avoid divide by zero warning
                    data[modality][img_type][18] = np.where(np.array(data[modality][img_type][18]) == 0, 1e-6, np.array(data[modality][img_type][18]))
                    type_combined_ratios = np.vstack((type_combined_ratios, data[modality][img_type][i][j,:]/np.hstack((np.array(data[modality][img_type][18])[j,:], np.array([1])))))    #1 added to divide the imporance column. divide by the whole gland feature
                    # type_combined_ratios[-1, -1] = data[modality][img_type][i][j,-1]
            #now normalize the data in each column (each feature)
            num_features = type_combined.shape[1]-1
            import warnings
            warnings.filterwarnings('error')
            
            num_features = type_combined_whole.shape[1]-1
            final_data[modality][img_type]["subsegs_dm"] = type_combined #design matrix for subsegment feature data
            final_data[modality][img_type]["subsegs_ratios"] = type_combined_ratios 
            final_data[modality][img_type]["whole_dm"] = type_combined_whole #design matrix for subsegment feature data

        all_spatial = np.concatenate((final_data[modality]["spatial"]["subsegs_dm"][:,:-1], final_data[modality]["spatial"]["subsegs_ratios"]),axis=1)
        
        final_data[modality]["all_spatial"] = copy.deepcopy(all_spatial)
        final_data[modality]["all"] = copy.deepcopy(all)
   
    all_spatial = np.concatenate((final_data["PET"]["all_spatial"][:,:-1], final_data["CT"]["all_spatial"]), axis=1)
    all_spatial_whole = np.concatenate((final_data["PET"]["spatial"]["whole_dm"][:,:-1], final_data["CT"]["spatial"]["whole_dm"]), axis=1)
  
    final_data["all_spatial"] = all_spatial
    final_data["all_spatial_whole"] = all_spatial_whole
   
    feature_names_spatial_all = copy.deepcopy(feature_names_spatial)
    for name in feature_names_spatial:
        feature_names_spatial_all.append(str(name + "_ratio"))

    for n, name in enumerate(feature_names_spatial_ct):
        feature_names_spatial_ct[n] = (str(name + "_ct"))        
    for n,name in enumerate(copy.deepcopy(feature_names_spatial_ct)):
        feature_names_spatial_ct.append((str(name + "_ct_ratio")))

    feature_names_spatial_all.extend(copy.deepcopy(feature_names_spatial_ct))
    feature_names_spatial_all_whole = copy.deepcopy(feature_names_spatial_whole)
    feature_names_spatial_all_whole.extend(copy.deepcopy(feature_names_spatial_whole_ct))

    final_data["feature_names_spatial_all_whole"] = feature_names_spatial_all_whole
    final_data["feature_names_spatial_all"] = feature_names_spatial_all

    for i in range(len(feature_names_spatial_ct)):
        feature_names_spatial_ct[i] += "_ct"
    for i in range(len(feature_names_spatial_whole_ct)):
        feature_names_spatial_whole_ct[i] += "_ct"    

  
    final_data["feature_names_spatial_whole_ct"] = feature_names_spatial_whole_ct

    temp_attr_array = np.array(data["patient_attributes"])
    attr_array = np.zeros((60*18, len(feature_names_patient_attributes)))
    for i in range(18*60):
        j = i % 60
        j = int(j / 2) #2 parotids for each patient, so switch patient every 2 in each subregion
        attr_array[i,:] = temp_attr_array[j,:]
    all_names_with_attrs = copy.deepcopy(feature_names_patient_attributes)
    all_names_with_attrs.extend(feature_names_spatial_all)

    all_data_with_attrs = np.concatenate((attr_array, all_spatial),axis=1)

    final_data["all_w_attrs"] = all_data_with_attrs
    final_data["feature_names_all_w_attrs"] = all_names_with_attrs


    with open(os.path.join(os.getcwd(), "importance_comparison", f"data_dict_{deblurred}_{suv_type}"), "wb") as fp:
        pickle.dump(final_data, fp)    
    return final_data

def rtstruct_to_nrrd():
    return

def extract_radiomics(image_path, mask_path, params=None):

    if params is not None:
        extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(params)
    else:  
        raise Exception("need a params file for extracting radiomics")
        extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()     
    result = extractor.execute(image_path, mask_path)   
    # for key, val in six.iteritems(result):
    #     print(f"{key}: :{val}")
    return result

def get_img_series_metadata(img_dir):
    #will return a list which has the metadata for each image in slice order. 
    #list is sorted from smallest to largest z. 

    file_paths = glob.glob(os.path.join(img_dir, "*"))
    img_list = []
    for path in file_paths:
        try:
            metadata = pydicom.dcmread(path)
        except:
            print(f"Could not load {path}. Continuing...")  
            continue  
        img_list.append(metadata)
    img_list.sort(key=lambda x: x.ImagePositionPatient[2])
    return img_list

def get_img_series_array(img_dir):
    #will return a shape [4 , (#z) , (#y), (#x)] array where first index has 4 specifications.
    #0 --> the regular image with pixel values
    #1 --> x value of each pixel
    #2 --> y value of each pixel 
    #3 --> z value of each pixel 
    #array is sorted from smallest to largest z. 
    #    
    meta_list = get_img_series_metadata(img_dir)
    

def get_all_nrrd_files(data_dir, patient_nums, subsegmentation=None, clear_dir=False):
    #clear dir true will erase all existing nrrd files found in save folders. 
    #if want features for subsegmtation of roi, then include the #slices [axial, y, x]
    for num in patient_nums:
        patient = "SIVIM" + num
        for image_type in ["ivim"]:
            if image_type == "ivim":
                modality = "CT"
            elif image_type == "dose":
                modality = "RTDOSE"    
            for scan_time in ["pre","post"]:
                img_dir = os.path.join(data_dir, patient, scan_time, "radiomics_data", "images", image_type,"dicom")
                img_save_dir = os.path.join(data_dir, patient, scan_time, "radiomics_data", "images", image_type,"stk")
                struct_dir = os.path.join(data_dir, patient, scan_time, "radiomics_data", "structs", image_type,"dicom")
                struct_save_dir = os.path.join(data_dir, patient, scan_time, "radiomics_data", "structs", image_type,"stk")
                save_paths = [img_save_dir, struct_save_dir]
                if clear_dir==True:
                    for file in glob.glob(os.path.join(img_save_dir, "*")):
                        os.remove(file)
                    for file in glob.glob(os.path.join(struct_save_dir, "*")):
                        os.remove(file)    

                dicom_to_nrrd.convert_all_dicoms_to_nrrd(img_dir, struct_dir, modality, subsegmentation=subsegmentation, save_paths=save_paths)

def get_all_radiomics_files(data_dir, patient_nums, clear_dir=False):
    for num in patient_nums:
        patient = "SIVIM" + num
        for image_type in ["ivim"]: 
            for scan_time in ["pre","post"]:
                img_dir = os.path.join(data_dir, patient, scan_time, "radiomics_data", "images", image_type,"stk")          
                struct_dir = os.path.join(data_dir, patient, scan_time, "radiomics_data", "structs", image_type,"stk")
                save_dir = os.path.join(data_dir, patient, scan_time, "radiomics_data", "features")

                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                else:
                    for dir in glob.glob(os.path.join(save_dir, "*")):
                        #temporary delete
                        shutil.rmtree(dir)     
                features_dict = {}
                for img_path in glob.glob(os.path.join(img_dir,"*")):
                    for struct_file in os.listdir(struct_dir):
                        structure_name = struct_file.split("__")[1]
                        segmentation_name = struct_file.split("__")[2]
                        mask_path = os.path.join(struct_dir, struct_file)
                        #img, header_img = nrrd.read(img_path)
                        #mask, header_mask = nrrd.read(mask_path)
                        if segmentation_name != "whole":
                            #remove shape features?
                            pass
                        #extract features 
                        # try:
                        features = extract_radiomics(img_path, mask_path)
                        # except:
                        #     print(f"Failed to calculate features for {mask_path}")
                        #     continue
                        if structure_name not in features_dict:
                            features_dict[structure_name] =  {}
                        features_dict[structure_name][segmentation_name] = features      
                        print(f"Calculated radiomics features for {structure_name}")
                with open(os.path.join(save_dir, "features_dict"), "wb") as fp:
                    pickle.dump(features_dict, fp)

if __name__ == "__main__":
    data_dir = os.path.join(os.getcwd(), "temp")
    

    # image_name, mask_name = radiomics.getTestCase('brain1',data_dir)

    # extract_radiomics(image_name, mask_name, None)

    ##now try for saved file 
    im_name = glob.glob("/media/sf_U_DRIVE/Profile/Desktop/Programs/IVIM_Code/Analysis_Data/SIVIM02/pre/radiomics_data/images/ivim/stk/*")[0]
    mask_name = glob.glob("/media/sf_U_DRIVE/Profile/Desktop/Programs/IVIM_Code/Analysis_Data/SIVIM02/pre/radiomics_data/structs/ivim/stk/*")[0]
    extract_radiomics(im_name, mask_name)
    print("Done")
    
