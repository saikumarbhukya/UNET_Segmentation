import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from PIL import Image
import os
from torchvision.utils import save_image
from unet import Unet
import datetime
from math import ceil



class SegmentationModel:
    def __init__(self, model_config):

        try:
            self.model_config = model_config
            # self.belt_model = torch.hub.load('ultralytics/yolov5', 'custom',
            #         '/ext512/nvidia/pytorch_models/wear_and_tear_detection/belt_num_det.pt', source='local')
            self.device = torch.device(device="cuda:0")
            self.model = Unet().to(self.device)
            self.model.load_state_dict(
                torch.load(self.model_config['weight_path']))
            self.model.eval()

            self.threshold = 0.85
            self.data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.0, 0.0, 0.0),
                    std=(1.0, 1.0, 1.0)
                )
            ])
            print("Segmentation Network Initialized!")
        except Exception as e_:
            print("Exception occurred in SegmentationModel init")

    def adjustImage(self, input_img, multiple=32):
        """ Adjusts Dimensions of the Image for inference by padding with zeros along edges.

        Args:
            input_img (ndarray): RGB input.

        Returns:
            ndarray: Padded output.
        """
        new_shape = [None, None, 3]
        img_shape = input_img.shape
        if (input_img.shape[0] % multiple == 0):
            new_shape[0] = input_img.shape[0]
        else:
            new_shape[0] = img_shape[0] - (img_shape[0] % multiple) + multiple
        if (input_img.shape[1] % multiple == 0):
            new_shape[1] = input_img.shape[1]
        else:
            new_shape[1] = img_shape[1] - (img_shape[1] % multiple) + multiple

        zeros_mask = np.zeros(shape=(new_shape), dtype=input_img.dtype)
        zeros_mask[:img_shape[0], :img_shape[1]] = input_img
        return zeros_mask

    def preproc_packet(self, img):
        try:
            height, width, channels = img.shape
            half_width = width // 2

            l_img = img[:,:half_width]
            r_img = img[:,half_width:]
            l_img = self.adjustImage(l_img)
            r_img = self.adjustImage(r_img)
            return l_img, r_img
        except Exception as e_:
            print(" Exception occurred in SegmentationModel preproc_packet :")

    def img2tensor(self, img):
        try:
            tnsr = self.data_transform(img).unsqueeze(0)
            return tnsr
        except Exception as e_:
            print("Exception occurred in SegmentationModel img2tensor :")

    def inference(self, input_packet):
        try:
            # Preprocess incoming publisher packet
            l_img, r_img = self.preproc_packet(input_packet)
            print(" Image packets from publisher preprocessed.")

            # Convert images to torch tensors
            l_tnsr = self.img2tensor(l_img)
            r_tnsr = self.img2tensor(r_img)
            print(f" l_tnsr, r_tnsr: {l_tnsr.shape} {r_tnsr.shape}")
            with torch.no_grad():
                tnsr = torch.cat((l_tnsr, r_tnsr), dim=0).to(self.device)
                print("Inference] Image converted to tensor")
                t=tnsr.shape
                # Get Model Inference
                out_tnsr = self.model(tnsr)
                print(
                    f"[ - Inference] Model output inferred: {out_tnsr.shape}")

                # Format Model inference to processable output by R.E
                l_out, r_out = out_tnsr.cpu().detach()
                l_out = l_out.view(l_out.shape[-2], l_out.shape[-1]).numpy()
                r_out = r_out.view(r_out.shape[-2], r_out.shape[-1]).numpy()
            out = np.hstack((l_out, r_out))

            threshold = out.max() * self.threshold
            temp_out = out * 255
            temp_out = temp_out.astype(np.uint8)
            # print_stats(temp_out, 'temp_out')

            _, thresh = cv2.threshold(
                temp_out, int(threshold * 255), 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(
                thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            #print(contours)
            output_img = cv2.hconcat([l_img, r_img])
            # belt_numdet = self.belt_model([output_img]).xywh[0]
            # belt_dets = [[x.item() for x in det[:4]] for det in belt_numdet]
            output = list()
            height, width, _ = output_img.shape
            if len(contours) > 0:
                for c in contours:
                    rect = list(cv2.boundingRect(c))
                    x, y, w, h = rect
                    # print(rect)        # if rect[2] < 100 or rect[3] < 100: continue
                    if cv2.contourArea(c) < 200:
                        continue
                    if x < 280 or x > 1720:
                        continue

                    output.append(['Defect', 90.0, rect])
                    cv2.rectangle(output_img, (x, y),
                                  (x + w, y + h), (0, 0, 255), 1)
                    cv2.imwrite(
                        '/home/zestiot/Desktop/Zestiot/PROJECTS/JSW/data/training/RMHS Setup03_30_5_Sai/infer' + str(random.randint(0, 100000)) + 'cv2' + '.jpg',
                        output_img)




                if len(output) == 0:
                    output.append(['No_Defect', 0.0, []])
            else:
                output.append(['No_Defect', 0.0, []])

            print(f"Inference] Model output formatted:{output}")

            return output, output_img
        except Exception as e_:
            print("Exception occurred in SegmentationModel inference :")

if __name__ == "__main__":
    model_details = {
        'Wear_and_Tear': {
            'model_type': 'SegmentationModel',
            'model_params': {
                'weight_path': '/home/zestiot/Desktop/Zestiot/PROJECTS/JSW/models/label_2024-06-12T18_38_47/label_2024-06-12T18_38_47_weights.pth'
            }
        }
    }
    model = SegmentationModel(model_details['Wear_and_Tear']['model_params'])
    path = "/home/zestiot/Desktop/Zestiot/PROJECTS/JSW/data/training/RMHS Setup03_30_5_Sai/RMHS Setup03_30_5"
    #path = "/home/suchith/Desktop/Project_JSW/Belt_Defects/Detections/"

    list_ = os.listdir(path)
    for x, img in enumerate(list_):

        image_cv2 = cv2.imread(path + img)

        inference_output_cv2, inference_output_img_cv2 = model.inference(image_cv2)

        print("#inference_output_cv2", inference_output_cv2)

