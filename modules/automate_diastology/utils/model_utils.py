import pydicom
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.densenet import densenet121
from torchvision.models.video import r2plus1d_18
from torchvision.models import convnext_base
import torch
import torchvision
import tqdm
import cv2 
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from utils.dicom_utils import change_dicom_color, get_doppler_region, find_horizontal_line, calculate_weighted_centroids_with_meshgrid
from utils import lav_mask
from utils.constants import *

cwd = Path.cwd()
weights_dir = cwd.parent

DOPPLER_WEIGHTS_DICT = {
    'medevel':weights_dir/"modules/measurement/weights/Doppler_models/medevel_weights.ckpt", # Download at: https://github.com/echonet/measurements/blob/main/weights/Doppler_models/medevel_weights.ckpt
    'latevel':weights_dir/"modules/measurement/weights/Doppler_models/latevel_weights.ckpt", # Download at: https://github.com/echonet/measurements/blob/main/weights/Doppler_models/latevel_weights.ckpt
    'trvmax':weights_dir/"modules/measurement/weights/Doppler_models/trvmax_weights.ckpt", # Download at: https://github.com/echonet/measurements/blob/main/weights/Doppler_models/trvmax_weights.ckpt
    'eovera':weights_dir/"modules/measurement/weights/Doppler_models/mvpeak_2c_weights.ckpt" # Download at: https://github.com/echonet/measurements/blob/main/weights/Doppler_models/mvpeak_2c_weights.ckpt
}

ALL_VIEWS = [
    'A2C','A3C','A4C','A5C',
    'Apical_Doppler',
    'Doppler_Parasternal_Long','Doppler_Parasternal_Short',
    'Parasternal_Long','Parasternal_Short',
    'SSN','Subcostal'
]

CLASS_LIST = ['BMode_A2C',
 'BMode_A2C_LV',
 'BMode_A3C',
 'BMode_A3C_LV',
 'BMode_A4C',
 'BMode_A4C_LA',
 'BMode_A4C_LV',
 'BMode_A4C_MV',
 'BMode_A4C_RV',
 'BMode_A5C',
 'BMode_DOPPLER_A2C',
 'BMode_DOPPLER_A3C',
 'BMode_DOPPLER_A3C_AV',
 'BMode_DOPPLER_A3C_MV',
 'BMode_DOPPLER_A4C_Apex',
 'BMode_DOPPLER_A4C_IAS',
 'BMode_DOPPLER_A4C_IVS',
 'BMode_DOPPLER_A4C_MV',
 'BMode_DOPPLER_A4C_Pulvns',
 'BMode_DOPPLER_A4C_TV',
 'BMode_DOPPLER_A5C',
 'BMode_DOPPLER_PLAX_AV_MV',
 'BMode_DOPPLER_PLAX_AV_zoomed',
 'BMode_DOPPLER_PLAX_Ascending_Aorta',
 'BMode_DOPPLER_PLAX_IVS',
 'BMode_DOPPLER_PLAX_MV_zoomed',
 'BMode_DOPPLER_PLAX_RVIT',
 'BMode_DOPPLER_PLAX_RVIT_CW',
 'BMode_DOPPLER_PLAX_RVOT',
 'BMode_DOPPLER_PSAX_IAS',
 'BMode_DOPPLER_PSAX_IVS',
 'BMode_DOPPLER_PSAX_MV',
 'BMode_DOPPLER_PSAX_level_great_vessels_AV',
 'BMode_DOPPLER_PSAX_level_great_vessels_PA',
 'BMode_DOPPLER_PSAX_level_great_vessels_TV',
 'BMode_DOPPLER_SC_4C_IAS',
 'BMode_DOPPLER_SC_4C_IVS',
 'BMode_DOPPLER_SC_IVC',
 'BMode_DOPPLER_SC_aorta',
 'BMode_DOPPLER_SSN_Aortic_Arch',
 'BMode_PLAX',
 'BMode_PLAX_AV_MV',
 'BMode_PLAX_Proximal_Ascending_Aorta',
 'BMode_PLAX_RV_inflow',
 'BMode_PLAX_RV_outflow',
 'BMode_PLAX_Zoom_out',
 'BMode_PLAX_zoomed_AV',
 'BMode_PLAX_zoomed_MV',
 'BMode_PSAX_(level_great_vessels)',
 'BMode_PSAX_(level_great_vessels)_focus_on_PV_and_PA',
 'BMode_PSAX_(level_great_vessels)_focus_on_TV',
 'BMode_PSAX_(level_great_vessels)_zoomed_AV',
 'BMode_PSAX_(level_of_MV)',
 'BMode_PSAX_(level_of_apex)',
 'BMode_PSAX_(level_of_papillary_muscles)',
 'BMode_SSN_aortic_arch',
 'BMode_Subcostal_4C',
 'BMode_Subcostal_Abdominal_Aorta',
 'BMode_Subcostal_IVC',
 'DOPPLER_A2C_MR_CW',
 'DOPPLER_A2C_MV_CW',
 'DOPPLER_A2C_MV_MR_CW',
 'DOPPLER_A3C _AV_AR_CW',
 'DOPPLER_A3C_AR_CW',
 'DOPPLER_A3C_AV_CW',
 'DOPPLER_A3C_AV_PW',
 'DOPPLER_A3C_LVOT_PW',
 'DOPPLER_A3C_MR_CW',
 'DOPPLER_A3C_MV_CW',
 'DOPPLER_A3C_MV_MR_CW',
 'DOPPLER_A4C_IVRT_PW',
 'DOPPLER_A4C_MR_CW',
 'DOPPLER_A4C_MV_CW',
 'DOPPLER_A4C_MV_MR_CW',
 'DOPPLER_A4C_MV_PW',
 'DOPPLER_A4C_PV_PW',
 'DOPPLER_A4C_TV_CW',
 'DOPPLER_A4C_TV_CW_Mayoview',
 'DOPPLER_A4C_TV_PW',
 'DOPPLER_A5C_AV_AR_CW',
 'DOPPLER_A5C_AV_CW',
 'DOPPLER_A5C_AV_PW',
 'DOPPLER_A5C_LVOT_PW',
 'DOPPLER_LV_midcavitary_PW',
 'DOPPLER_PLAX_RVIT_CW',
 'DOPPLER_PSAX_Great_vessel_level_PA_CW',
 'DOPPLER_PSAX_Great_vessel_level_PA_PW',
 'DOPPLER_PSAX_Great_vessel_level_TV_CW',
 'DOPPLER_SC_HV/IVC_PW',
 'DOPPLER_SC_TV_CWQ',
 'DOPPLER_SC_abdominal_AO_PW',
 'DOPPLER_SSN_descending_AO_CW',
 'DOPPLER_SSN_descending_AO_PW',
 'M-mode_A4C_TV_TAPSE',
 'M-mode_PSAX_AV',
 'M-mode_PSAX_LV_PM level',
 'M-mode_PSAX_MV level',
 'M_mode_A4C_RV_TAPSE',
 'M_mode_PLAX_Ao_LA',
 'M_mode_PLAX_LV',
 'M_mode_PLAX_MV',
 'M_mode_SC_IVC',
 'TDI_MV_Lateral e',
 'TDI_MV_Medial e',
 'TDI_TV_Lateral S','NON_ULTRASOUND'] 


'''
    EchoNet-Dynamic LVEF Model and Inference
'''
def ef_regressor(device=torch.device("cuda:0"),weights_path=weights_dir/'weights/lvef_weights.pt'):
    model = torchvision.models.video.__dict__["r2plus1d_18"](pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features,1)
    if device.type=='cuda':
        model = torch.nn.DataParallel(model,device_ids=[0])
    model.to(device)
    checkpoint = torch.load(weights_path,map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval(), checkpoint

def predict_lvef(x,ef_model,ef_checkpoint,device=torch.device("cuda:0"),dims=(112,112)):
    ### x is a 4D tensor of an echo
    mean = ef_checkpoint['mean'].reshape(3,1,1,1)
    std = ef_checkpoint['std'].reshape(3,1,1,1)
    device = torch.device("cuda:0")
    x = x.permute(1,0,2,3) # Change from F,C,H,W to C,F,H,W
    x -= mean
    x /= std
    c,f,h,w = x.shape
    x = x[:,np.arange(0,f,2),:,:] # Sample every other frame 
    x = torch.tensor(x).unsqueeze(0)
    x = x.to(device)
    ef_prediction = ef_model(x).item()
    return ef_prediction

'''
    Left Atrial Segmentation Model and Inference
'''
def load_la_model(device=torch.device("cuda:0"),weights_path=weights_dir/'weights/lav_weights.pt'):
    model = torchvision.models.segmentation.__dict__['deeplabv3_resnet50']()
    model.classifier[-1] = torch.nn.Conv2d(
        model.classifier[-1].in_channels,
        1,
        kernel_size=model.classifier[-1].kernel_size,
    )
    model = torch.nn.DataParallel(model,device_ids=[0])
    model.to(device)
    checkpoint = torch.load(weights_path,map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval()

def la_seg_inf(model,x,device=torch.device("cuda:0"),n=112):
    f,h,w,c = x.shape
    mean = GLOBAL_LA_MEAN.reshape(1,3,1,1)
    std = GLOBAL_LA_STD.reshape(1,3,1,1)
    x -= mean 
    x /= std 
    x = x.float()
    x = x.to(device)
    seg = model(x)['out']
    logits = seg.detach().cpu().numpy()
    logits = np.concatenate([logits])
    logits = logits[:,0,:,:]
    y = logits>0
    area = y.sum((1,2))
    area = area.astype(np.float64)
    return y,area

def calc_lav_from_a4c(mask,area):
    # mask,area = la_seg_inf(model,a4c)
    a4c_area = lav_mask.filter_areas(area)
    mask_a4c = mask[np.argmax(a4c_area)]
    contour,m_mitral,m_length,m_p,length,h = lav_mask.get_la_vals(mask_a4c)
    m_perpend,b_perpend,end = lav_mask.find_perpendicular(contour,m_mitral,m_p)
    h,length,axes,endpts = lav_mask.find_axes(contour,m_mitral,m_perpend,m_p,end)
    lav = lav_mask.calc_mod_volume(h,axes)
    return lav

def calc_lav_biplane(a4c_mask,a4c_area,a2c_mask,a2c_area):
    ### Segment LA from A4C 
    a4c_area = lav_mask.filter_areas(a4c_area)
    a4c_mask = a4c_mask[np.argmax(a4c_area)]
    ### Get geometric features of A4C LA mask
    a4c_contour,a4c_mmitral,a4c_mlength,a4c_mp,a4c_length,a4c_h = lav_mask.get_la_vals(a4c_mask)
    a4c_mperpend,a4c_bperpend,a4c_end = lav_mask.find_perpendicular(a4c_contour,a4c_mmitral,a4c_mp)
    h,length,a4c_axes,a4c_endpts = lav_mask.find_axes(a4c_contour,a4c_mmitral,a4c_mperpend,a4c_mp,a4c_end)

    ### Segment LA from A2C
    a2c_area = lav_mask.filter_areas(a2c_area)
    mask_a2c = a2c_mask[np.argmax(a2c_area)]
    a2c_contour,a2c_mmitral,a2c_mlength,a2c_mp,a2c_length,a2c_h = lav_mask.get_la_vals(mask_a2c)
    a2c_mperpend,a2c_bperpend,a2c_end = lav_mask.find_perpendicular(a2c_contour,a2c_mmitral,a2c_mp)
    h,length,a2c_axes,a2c_endpts = lav_mask.find_axes(a2c_contour,a2c_mmitral,a2c_mperpend,a2c_mp,a2c_end,a4c_end)

    ### Calculate disc axes on A4C again using information from A2C 
    ### Uses minimum disc height calculated from A2C vs A4C
    h,length,a4c_axes,a4c_endpts = lav_mask.find_axes(a4c_contour,a4c_mmitral,a4c_mperpend,a4c_mp,a4c_end,a2c_end)
    ### Calculate LAV using biplane MOD
    lav = lav_mask.calc_mod_volume(h,a4c_axes,a2c_axes)
    return lav#*scale
'''
    View Classification Model and Inference
'''
def load_view_classifier(device=torch.device("cuda:0"),weights_path=weights_dir/'EchoSky/modules/view_classification/weights/view_classifier.pt'): 
    view_classifier = torchvision.models.convnext_base()
    view_classifier.classifier[-1] = torch.nn.Linear(
        view_classifier.classifier[-1].in_features,len(ALL_VIEWS)
    )
    vc_state_dict = torch.load(weights_path, map_location='cpu')
    view_classifier.load_state_dict(vc_state_dict)
    view_classifier.to(device)
    view_classifier.eval()
    for param in view_classifier.parameters():
        param.requires_grad = False
    return view_classifier.eval()

def view_inference(view_input,view_classifier,filename,device=torch.device("cuda:0"),batch_size=512):
    view_labels = torch.zeros(len(view_input))
    view_ds = torch.utils.data.TensorDataset(view_input,view_labels)
    view_loader = torch.utils.data.DataLoader(
        view_ds,batch_size=batch_size,
        num_workers=0,shuffle=False
    )
    yhat = [""]*len(view_loader)
    with torch.no_grad():
        for idx,(image,views) in tqdm.tqdm(enumerate(view_loader)):
            image = image.to(device)
            out = view_classifier(image)
            out = torch.argmax(out,dim=1)
            preds = [ALL_VIEWS[o] for o in out]
            start = idx*batch_size
            end = min((idx+1)*batch_size,len(view_ds))
            yhat[start:end] = preds
    predicted_view = {key:val for (key,val) in zip(filename,yhat)}
    return predicted_view

'''
    106 View Classifier
'''
def load_view_106_model(device=torch.device('cuda:0'),weights_path=weights_dir/'weights/updated_view_classifer.pt'): 
    num_classes = len(CLASS_LIST)
    model = convnext_base(num_classes=num_classes+1)
    checkpoint = torch.load(weights_path,map_location="cpu")
    checkpoint = {k[2:]:v for k,v in checkpoint.items()}
    model.load_state_dict(checkpoint)
    model.to(device)
    return model.eval() 

def view_106_inference(dataset,model,filename,device=torch.device("cuda:0"),batch_size=64):
    view_labels = torch.zeros(len(dataset))
    view_ds = torch.utils.data.TensorDataset(dataset,view_labels)
    view_loader = torch.utils.data.DataLoader(
        view_ds,batch_size=batch_size,
        num_workers=0,shuffle=False
    )
    yhat = [""]*len(view_loader)
    with torch.no_grad():
        for idx,(image,views) in tqdm.tqdm(enumerate(view_loader)):
            image = image.to(device)
            logits = model(image)
            confidence_score = torch.nn.functional.softmax(logits[:,:106],dim=1).cpu()
            pred_views = [CLASS_LIST[score] for score in torch.argmax(torch.tensor(confidence_score),dim=1)]
            start = idx*batch_size
            end = min((idx+1)*batch_size,len(view_ds))
            yhat[start:end] = pred_views
    predicted_view = {key:val for (key,val) in zip(filename,yhat)}
    return predicted_view


'''
    Quality Control Model and Inference
'''
def load_quality_classifier(input_type,
                            weights_path,
                            device=torch.device('cuda:0')):
    if input_type=='image':
        weights = torch.load(weights_path, map_location=device)
        model = densenet121(num_classes=1)
        weights = {key.replace('m.','',1):val for key,val in weights.items()}
        model.load_state_dict(weights)
    elif input_type=='video':
        weights = torch.load(weights_path, map_location=device, weights_only=True)
        new_state_dict = {}
        for k, v in weights.items():
            new_key = k[2:] if k.startswith('m.') else k
            new_state_dict[new_key] = v

        model = r2plus1d_18(num_classes=1)
        model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    return model.eval()

def quality_inference(quality_input,quality_model,filename,device=torch.device("cuda:0"),batch_size=32):
    quality_labels = torch.zeros(len(quality_input))
    quality_ds = torch.utils.data.TensorDataset(quality_input,quality_labels)
    quality_dl = torch.utils.data.DataLoader(quality_ds,batch_size=batch_size,num_workers=1,shuffle=False)
    next(iter(quality_dl))
    yhat = np.zeros(len(quality_ds))
    with torch.no_grad():
        for idx,(x,label) in tqdm.tqdm(enumerate(quality_dl)):
            x = x.to(device)
            preds = quality_model(x)
            start = idx * batch_size
            end = min((idx + 1) * batch_size, len(quality_ds))
            activated = F.sigmoid(preds).squeeze().cpu().numpy()
            yhat[start:end] = F.sigmoid(preds).squeeze().cpu().numpy()
    predicted_quality = {key:val for (key,val) in zip(filename,yhat)}
    return predicted_quality

''' 
    EchoNet-Measurements Doppler Parameter Models and Inference 
'''
def load_doppler_model(parameter):
    device = 'cuda:0'
    weights_path = DOPPLER_WEIGHTS_DICT[parameter]
    if parameter == 'eovera':
        num_classes = 2 
    else: 
        num_classes = 1
    weights = torch.load(weights_path,map_location='cuda:0')
    backbone = deeplabv3_resnet50(num_classes=num_classes)
    weights = {k.replace('m.',''):v for k,v in weights.items()}
    backbone.load_state_dict(weights)
    backbone.eval()
    return backbone 

### Runs inference using weights from models for medial e', lateral e', or TR Vmax 
def doppler_inference(dicom_path,parameter,device=torch.device('cuda:0')):
    ds = pydicom.dcmread(dicom_path)
    input_image = change_dicom_color(dicom_path) #ds.pixel_array
    x0,x1,y0,y1,conversion_factor = get_doppler_region(ds)
    if (x0,x1,y0,y1,conversion_factor) == (-1,-1,-1,-1,-1):
        return [],-1,-1,-1
    horizontal_y = find_horizontal_line(ds.pixel_array[y0:y1, :])
    #Basically, the region where the Doppler signal starts is 342-345. We truncate the image from 342 to 768. Make 426*1024.
    input_dicom_doppler_area = input_image[342:,:,:] #ds.pixel_array[342 :,:, :] 
    doppler_area_tensor = torch.tensor(input_dicom_doppler_area)
    doppler_area_tensor = doppler_area_tensor.permute(2, 0, 1).unsqueeze(0)
    doppler_area_tensor = doppler_area_tensor.float() / 255.0
    doppler_area_tensor = doppler_area_tensor.to('cuda:0')
    backbone = load_doppler_model(parameter=parameter)
    backbone = backbone.to(device)
    param = next(iter(backbone.parameters()))
    # print('Device: ',param.device)
    with torch.no_grad():
        logit = forward_pass(backbone,doppler_area_tensor)
        max_val = logit.max().item()
        min_val = logit.min().item()
        logits_normalized = (logit - min_val) / (max_val - min_val)
        logits_normalized = logits_normalized.squeeze().cpu().numpy()
        max_coords = np.unravel_index(np.argmax(logits_normalized), logits_normalized.shape)
        
        X = max_coords[1]  # Max Logit X value
        Y = max_coords[0] # Max Logit Y value in the Doppler Region
        pred_x = int(X) 
        pred_y = int(Y + y0) #add y0 to get the actual y value in the original image to map
        
        peak_velocity = conversion_factor * (pred_y - (y0 + horizontal_y))
        if parameter == 'trvmax':
            peak_velocity /= 100 # Convert cm/s to m/s
        peak_velocity = round(peak_velocity, 2)
    return input_image,peak_velocity,pred_x,pred_y

def forward_pass(backbone, inputs):
    logits = backbone(inputs)["out"] # torch.Size([1, 2, 480, 640])
    # Step 1: Apply sigmoid if needed
    if DO_SIGMOID:
        logits = torch.sigmoid(logits)
    # Step 2: Apply segmentation threshold if needed
    if SEGMENTATION_THRESHOLD is not None:
        logits[logits < SEGMENTATION_THRESHOLD] = 0.0
    return logits

### Runs inference using weights from model for MV E and A
def eovera_forward_pass(backbone,inputs):
    logits = backbone(inputs)["out"]
    if DO_SIGMOID:
        logits = torch.sigmoid(logits)
    if SEGMENTATION_THRESHOLD is not None:
        logits[logits < SEGMENTATION_THRESHOLD] = 0.0
    
    logits_numpy = logits.squeeze().detach().cpu().numpy()
    logits_first = logits_numpy[1, :, :] #First channel is in 1. Sorry for the confusion.
    max_val_first, min_val_first = logits_first.max(), logits_first.min()
    logits_first = (logits_first - min_val_first) / (max_val_first - min_val_first)
    _, _, _, max_loc_first_channel = cv2.minMaxLoc(logits_first)
    
    logits_second = logits_numpy[0, :, :] #Second channel is in 0.
    max_val_second, min_val_second = logits_second.max(), logits_second.min()
    logits_second = (logits_second - min_val_second) / (max_val_second - min_val_second)
    _, _, _, max_loc_second_channel = cv2.minMaxLoc(logits_second)
    
    combine_logit = logits_first + logits_second
    _, _, _, max_loc_combine = cv2.minMaxLoc(combine_logit)
    
    #Check max_loc_combine is come from which channel
    #1. calculate difference between max_loc_combine and max_loc_first_channel / and max_loc_second_channel
    diff_maxloc_combine_first = np.sqrt((max_loc_combine[0] - max_loc_first_channel[0])**2 + (max_loc_combine[1] - max_loc_first_channel[1])**2)
    diff_maxloc_combine_second = np.sqrt((max_loc_combine[0] - max_loc_second_channel[0])**2 + (max_loc_combine[1] - max_loc_second_channel[1])**2)

    centroids_first, _ = calculate_weighted_centroids_with_meshgrid(logits_first)
    centroids_second, _ = calculate_weighted_centroids_with_meshgrid(logits_second)
    centroids, _ = calculate_weighted_centroids_with_meshgrid(combine_logit)
    
    #2. Pick the closest centeroid point to max_loc_combine
    #Among many points centeroids (basically, 2-4 points), get the closest point to max_loc_combine
    #Dictionary for distance between each centroid and maxlogits coordinate
    distance_centroid_btw_maxlogits = {}
    for centroid in centroids:
        distance = np.sqrt((max_loc_combine[0] - centroid[0])**2 + (max_loc_combine[1] - centroid[1])**2)
        distance_centroid_btw_maxlogits[centroid] = distance
    #Get the coordinates with minimum value of distance between maxlogits and centroid
    try:
        min_distance_coord  = min(distance_centroid_btw_maxlogits, key=distance_centroid_btw_maxlogits.get)
    except:
        print("Error: min_distance_coord is not found, due to low prediction score. Select Good quality MVPeak Doppler data")
        return 0,0,0,0
    
    #3. Calculate distance between min_distance_coord and other centroids
    distance_btw_centroids = {}
    if diff_maxloc_combine_second - diff_maxloc_combine_first > 15:
        # print("max_loc_combine is from the first channel. Pick Pair from the second channel")
        for centroid in centroids_second:
            distance = np.sqrt((min_distance_coord[0] - centroid[0])**2 + (min_distance_coord[1] - centroid[1])**2)
            distance_btw_centroids[centroid] = distance
            
    elif diff_maxloc_combine_first - diff_maxloc_combine_second > 15:
        # print("max_loc_combine is from the second channel. Pick Pair from the first channel")
        for centroid in centroids_first:
            distance = np.sqrt((min_distance_coord[0] - centroid[0])**2 + (min_distance_coord[1] - centroid[1])**2)
            distance_btw_centroids[centroid] = distance
    else:
        # print("Other. Get the coordinates from combined logit channel")
        distance_btw_centroids = {}
        for centroid in centroids:
            distance = np.sqrt((min_distance_coord[0] - centroid[0])**2 + (min_distance_coord[1] - centroid[1])**2)
            distance_btw_centroids[centroid] = distance
    
    #4. Get the closest Pair of Coordinates
    non_zero_distance_btw_centroids = {k:v for k, v in distance_btw_centroids.items() if v > 15}
    try:
        min_distance_paired_coord = min(non_zero_distance_btw_centroids, key=non_zero_distance_btw_centroids.get)
        pair_coords = [min_distance_coord, min_distance_paired_coord] 
    except:
        print("Error: min_distance_coord is not found, due to low prediction score. Please select higher quality MVPeak Doppler data")
        return 0, 0, 0, 0
    point_x1, point_y1= pair_coords[0][0], pair_coords[0][1] 
    point_x2, point_y2 = pair_coords[1][0], pair_coords[1][1]
            
    if point_x1 > point_x2:
        point_x1, point_y1, point_x2, point_y2 = point_x2, point_y2, point_x1, point_y1

    distance_x1_x2 = abs(point_x1 - point_x2)
    if distance_x1_x2 > 300:
        print("Error: The distance between two points is too far. Please select higher quality Doppler data.")
        return 0,0,0,0
            
    return point_x1, point_y1, point_x2, point_y2

def eovera_inference(dicom_path):
    device = 'cuda:0'
    ds = pydicom.dcmread(dicom_path)
    input_image = change_dicom_color(dicom_path)
    x0,x1,y0,y1,conversion_factor = get_doppler_region(ds)
    if (x0,x1,y0,y1,conversion_factor) == (-1,-1,-1,-1,-1):
        print(f'MV E/A was not calculated')
        return [],-1,0,0,0,0,0,0,0 
    #horizontal line means the line where the Doppler signal starts
    horizontal_y = find_horizontal_line(ds.pixel_array[y0:y1, :])
    #Basically, the region where the Doppler signal starts is 342-345. We truncate the image from 342 to 768. Make 426*1024.
    input_dicom_doppler_area = input_image[342:,:,:] #ds.pixel_array[342:,:, :] 
    doppler_area_tensor = torch.tensor(input_dicom_doppler_area)
    doppler_area_tensor = doppler_area_tensor.permute(2,0,1).unsqueeze(0)
    doppler_area_tensor = doppler_area_tensor.float()/255.0
    doppler_area_tensor = doppler_area_tensor.to(device)
 
    with torch.no_grad():
        backbone = load_doppler_model(parameter='eovera')
        backbone = backbone.to(device)
        point_x1, point_y1, point_x2, point_y2 = eovera_forward_pass(backbone,doppler_area_tensor)
    
    Inference_E_Vel = round(abs((point_y1 - horizontal_y) * conversion_factor),4)
    Inference_A_Vel = round(abs((point_y2 - horizontal_y) * conversion_factor),4)
    Inference_EperA = round(Inference_E_Vel / Inference_A_Vel, 3)
    return input_image,y0,point_x1,point_x2,point_y1,point_y2,Inference_A_Vel,Inference_E_Vel,Inference_EperA

def pad(avis,max_frames):
    padded = []
    for a in avis:
        pad_len = max_frames - a.shape[0]
        if pad_len > 0:
            pad = torch.zeros((pad_len,*a.shape[1:]),dtype=a.dtype,device=a.device)
            a = torch.cat([a,pad], dim=0)
        padded.append(a)
    return padded