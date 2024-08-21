import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights
from utils.utils import form_slices_32
import numpy as np
from CustomConfigParser import CustomConfigParser, CurrentTimeInterpolation
from configparser import ExtendedInterpolation

class Load_VIT():
    def __init__(self):
        config_parser = CustomConfigParser(interpolation=ExtendedInterpolation())
        self.device = config_parser.get('default', 'device')

    def extract_features(self, input_tensor):
        #print("Here: ", input_tensor.shape)
        # Set up the device
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load a pre-trained ViT model
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(self.device)

        # Remove the classification head
        model.heads = torch.nn.Identity()

        # Set the model to evaluation mode
        model.eval()

        input_tensor = torch.from_numpy(input_tensor)
        input_tensor = input_tensor.float() / 255.0

        input_tensor = input_tensor.permute(0, 3, 1, 2)
        slices = form_slices_32(input_tensor.size(0), input_tensor.size(0) // 32, input_tensor.size(0) // 32)

        # uncomment for averaged 32 stack features
        # vid_feats = []
        # uncomment for extract 768 sized features for all images in the video segment
        vid_feats = torch.empty((0, 768)).to(self.device)

        print("Num of slices: ", len(slices))
        for stack_idx, (start_idx, end_idx) in enumerate(slices):
            # inference
            rgb_stack = input_tensor[start_idx:end_idx, :, :, :]

            # below is to creates a duplicate frame for each frame for data augmentation and to
            # avoid errors with too small stack sizes
            # Select frames from start_idx to end_idx (inclusive) along the frame dimension
            #if (end_idx - start_idx < 10):
            #    #rgb_stack = self.augment_rgb_frame(rgb, start_idx, end_idx)
            #    print("Need to augment...")

            #print('Extracting for slices: ', start_idx, end_idx)

            #torch.no_grad()
            # torch.cuda.empty_cache()
            #print(torch.cuda.memory_summary())

            rgb_stack = rgb_stack.to(self.device)
            #input_tensor = input_tensor.permute(1, 0, 2, 3)
            # Resize the input tensor to match ViT's expected input size [N, 3, 224, 224]
            resized_input = torch.nn.functional.interpolate(rgb_stack, size=(224, 224), mode='bilinear',
                                                            align_corners=False)

            # Extract features
            with torch.no_grad():
                output = model(resized_input)
            # Uncomment the below for a (32,768) stack of average features
            #output = torch.mean(output, dim=0).unsqueeze(0)
            #vid_feats.extend(output.tolist())

            # Extract for each image
            vid_feats = torch.cat([vid_feats, output], dim=0).to(self.device)



        feats_dict = {
            # uncomment for 32 averaged features
            #'vit': np.array(vid_feats),
            'vit': vid_feats,
        }


        print(f"Extracted features shape: {output.shape}")

        del rgb_stack
        torch.cuda.empty_cache()

        return feats_dict


