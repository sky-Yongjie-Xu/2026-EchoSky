from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F
import torch
from peft import LoraConfig, get_peft_model, TaskType
import glob
from tqdm import tqdm
import numpy as np
import pydicom
import cv2
import torchvision
import math


class EchoGemma(nn.Module):
    def __init__(self,emb_dim=523, device=torch.device('cuda')):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("modules/report_generation/medgemma-1.5-4b-it", use_fast=True)

        self.medgemma = AutoModelForCausalLM.from_pretrained(
            "modules/report_generation/medgemma-1.5-4b-it",
            torch_dtype=torch.float,
            device_map="cpu"
        )

        lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"]

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=lora_target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.medgemma = get_peft_model(self.medgemma, lora_config)

        ## EchoPrime encoder
        self.echo_encoder = torchvision.models.video.mvit_v2_s()
        self.echo_encoder.head[-1] = torch.nn.Linear(self.echo_encoder.head[-1].in_features, 512)
        self.view_classifier = torchvision.models.convnext_base()
        self.view_classifier.classifier[-1] = torch.nn.Linear(
            self.view_classifier.classifier[-1].in_features, 11
        )
    
        # project EchoPrime embeddings to 2560 so that they can be merged
        self.visual_projection = torch.nn.Linear(emb_dim, 2560, dtype=torch.float)

        # video parameters
        self.frames_to_take=32
        self.frame_stride=2
        self.video_size=224
        self.mean = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
        self.std = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1, 1)
        self.device=device
    
        # load finetuned weights
        echogemma_checkpoint = torch.load("modules/report_generation/weights/echogemma.pt", map_location='cpu')
        self.load_state_dict(echogemma_checkpoint)
        self.to(device)
        self.eval()

    def process_dicoms(self,INPUT):
        """
        Reads DICOM video data from the specified folder and returns a tensor 
        formatted for input into the EchoPrime model.

        Args:
            INPUT (str): Path to the folder containing DICOM files.

        Returns:
            stack_of_videos (torch.Tensor): A float tensor of shape  (N, 3, 16, 224, 224)
                                            representing the video data where N is the number of videos,
                                            ready to be fed into EchoPrime.
        """

        dicom_paths = glob.glob(f'{INPUT}/**/*.dcm',recursive=True)
        stack_of_videos=[]
        for idx, dicom_path in tqdm(enumerate(dicom_paths),total=len(dicom_paths)):
            try:
                # simple dicom_processing
                dcm=pydicom.dcmread(dicom_path)
                pixels = pydicom.pixels.pixel_array(dcm, raw=True)
                
                # exclude images like (600,800) or (600,800,3)
                if pixels.ndim < 3 or pixels.shape[2]==3:
                    continue 
                    
                # if single channel repeat to 3 channels    
                if pixels.ndim==3:
                    pixels = np.repeat(pixels[..., None], 3, axis=3)
                
                # mask everything outside ultrasound region
                pixels=self.mask_outside_ultrasound(pixels)
                
                
                
                #model specific preprocessing
                x = np.zeros((len(pixels),224,224,3))
                for i in range(len(x)):
                    x[i] = self.crop_and_scale(pixels[i])
                
                x = torch.as_tensor(x, dtype=torch.float).permute([3,0,1,2])
                # normalize
                x.sub_(self.mean).div_(self.std)
            
                ## if not enough frames add padding
                if x.shape[1] < self.frames_to_take:
                    padding = torch.zeros(
                    (
                        3,
                        self.frames_to_take - x.shape[1],
                        self.video_size,
                        self.video_size,
                    ),
                    dtype=torch.float,
                    )
                    x = torch.cat((x, padding), dim=1)
                    
                start=0
                stack_of_videos.append(x[:, start : ( start + self.frames_to_take) : self.frame_stride, : , : ])
                
            except Exception as e:
                print("corrupt file")
                print(str(e))

        stack_of_videos=torch.stack(stack_of_videos)
        
        return stack_of_videos

    @torch.inference_mode()
    def generate(self, stack_of_videos, max_tokens=1024, temperature=0.0, bin_size=50):
        """Generate a report given study embeddings using HuggingFace's generate method
            stack_of_videos (torch.Tensor) [#N_videos, 3, 16, 224, 224]: preprocessed videos that belong to a the same echocardiography study 
            max_tokens (int): max tokens llm can generate
            temperature (float): llm temperature
            bin_size (int): 
        """
        prompt="<start_of_turn>user\nGenerate an echocardiography text report based on the study.<end_of_turn>\n<start_of_turn>model\n"

        # Prepare study embeddings
        n_bins=math.ceil(stack_of_videos.shape[0]/bin_size)
        stack_of_features_list=[]
        for bin_idx in range(n_bins):
            start_idx = bin_idx * bin_size
            end_idx = min( (bin_idx + 1) * bin_size, stack_of_videos.shape[0])
            bin_videos = stack_of_videos[start_idx:end_idx].to(self.device)
            bin_features = self.echo_encoder(bin_videos)
            stack_of_features_list.append(bin_features)
        stack_of_features=torch.cat(stack_of_features_list,dim=0)
        stack_of_features = F.normalize(stack_of_features, dim=-1)


        ## Get views   
        stack_of_first_frames = stack_of_videos[:,:,0,:,:].to(self.device)
        out_logits=self.view_classifier(stack_of_first_frames)
        out_views=torch.argmax(out_logits,dim=1)
        stack_of_view_encodings = torch.nn.functional.one_hot(out_views, num_classes=11).float().to(self.device)
    
        # Concat
        study_embeddings = torch.cat( (stack_of_features ,stack_of_view_encodings),dim=1)
        study_embeddings = study_embeddings.to(self.device)
        study_embeddings = study_embeddings.unsqueeze(0)

        # Project visual embeddings
        visual_embs = self.visual_projection(study_embeddings)  # (1, N_videos, 2560)

        # Encode prompt (includes BOS automatically)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt").to(self.device)
        token_embs = self.medgemma.get_input_embeddings()(prompt_ids)  # (1, T, 2560)

        # Combine visual + token embeddings
        inputs_embeds = torch.cat([visual_embs, token_embs], dim=1)

        # Create attention mask
        attention_mask = torch.ones(1, inputs_embeds.shape[1], dtype=torch.long, device=self.device)

        # Use HuggingFace generate with inputs_embeds
        if temperature == 0.0:
            outputs = self.medgemma.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        else:
            outputs = self.medgemma.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = outputs[0].tolist()
        output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return output

    @staticmethod
    def mask_outside_ultrasound(original_pixels: np.array) -> np.array:
        """
        Masks all pixels outside the ultrasound region in a video.

        Args:
        vid (np.ndarray): A numpy array representing the video frames. FxHxWxC

        Returns:
        np.ndarray: A numpy array with pixels outside the ultrasound region masked.
        """
        try:
            testarray=np.copy(original_pixels)
            vid=np.copy(original_pixels)
            ##################### CREATE MASK #####################
            # Sum all the frames
            frame_sum = testarray[0].astype(np.float32)  # Start off the frameSum with the first frame
            frame_sum = cv2.cvtColor(frame_sum, cv2.COLOR_YUV2RGB)
            frame_sum = cv2.cvtColor(frame_sum, cv2.COLOR_RGB2GRAY)
            frame_sum = np.where(frame_sum > 0, 1, 0) # make all non-zero values 1
            frames = testarray.shape[0]
            for i in range(frames): # Go through every frame
                frame = testarray[i, :, :, :].astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                frame = np.where(frame>0,1,0) # make all non-zero values 1
                frame_sum = np.add(frame_sum,frame)

            # Erode to get rid of the EKG tracing
            kernel = np.ones((3,3), np.uint8)
            frame_sum = cv2.erode(np.uint8(frame_sum), kernel, iterations=10)

            # Make binary
            frame_sum = np.where(frame_sum > 0, 1, 0)

            # Make the difference frame fr difference between 1st and last frame
            # This gets rid of static elements
            frame0 = testarray[0].astype(np.uint8)
            frame0 = cv2.cvtColor(frame0, cv2.COLOR_YUV2RGB)
            frame0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
            frame_last = testarray[testarray.shape[0] - 1].astype(np.uint8)
            frame_last = cv2.cvtColor(frame_last, cv2.COLOR_YUV2RGB)
            frame_last = cv2.cvtColor(frame_last, cv2.COLOR_RGB2GRAY)
            frame_diff = abs(np.subtract(frame0, frame_last))
            frame_diff = np.where(frame_diff > 0, 1, 0)

            # Ensure the upper left hand corner 20x20 box all 0s.
            # There is a weird dot that appears here some frames on Stanford echoes
            frame_diff[0:20, 0:20] = np.zeros([20, 20])

            # Take the overlap of the sum frame and the difference frame
            frame_overlap = np.add(frame_sum,frame_diff)
            frame_overlap = np.where(frame_overlap > 1, 1, 0)

            # Dilate
            kernel = np.ones((3,3), np.uint8)
            frame_overlap = cv2.dilate(np.uint8(frame_overlap), kernel, iterations=10).astype(np.uint8)

            # Fill everything that's outside the mask sector with some other number like 100
            cv2.floodFill(frame_overlap, None, (0,0), 100)
            # make all non-100 values 255. The rest are 0
            frame_overlap = np.where(frame_overlap!=100,255,0).astype(np.uint8)
            contours, hierarchy = cv2.findContours(frame_overlap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # contours[0] has shape (445, 1, 2). 445 coordinates. each coord is 1 row, 2 numbers
            # Find the convex hull
            for i in range(len(contours)):
                hull = cv2.convexHull(contours[i])
                cv2.drawContours(frame_overlap, [hull], -1, (255, 0, 0), 3)
            frame_overlap = np.where(frame_overlap > 0, 1, 0).astype(np.uint8) #make all non-0 values 1
            # Fill everything that's outside hull with some other number like 100
            cv2.floodFill(frame_overlap, None, (0,0), 100)
            # make all non-100 values 255. The rest are 0
            frame_overlap = np.array(np.where(frame_overlap != 100, 255, 0),dtype=bool)
            ################## Create your .avi file and apply mask ##################
            # Store the dimension values

            # Apply the mask to every frame and channel (changing in place)
            for i in range(len(vid)):
                frame = vid[i, :, :, :].astype('uint8')
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
                frame = cv2.bitwise_and(frame, frame, mask = frame_overlap.astype(np.uint8))
                vid[i,:,:,:]=frame
            return vid
        except Exception as e:
            print("Error masking returned as is.")
            return vid
    
    @staticmethod
    def crop_and_scale(img, res=(224, 224), interpolation=cv2.INTER_CUBIC, zoom=0.1):
        in_res = (img.shape[1], img.shape[0])
        r_in = in_res[0] / in_res[1]
        r_out = res[0] / res[1]

        if r_in > r_out:
            padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
            img = img[:, padding:-padding]
        if r_in < r_out:
            padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
            img = img[padding:-padding]
        if zoom != 0:
            pad_x = round(int(img.shape[1] * zoom))
            pad_y = round(int(img.shape[0] * zoom))
            img = img[pad_y:-pad_y, pad_x:-pad_x]

        img = cv2.resize(img, res, interpolation=interpolation)
        return img