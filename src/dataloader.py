import torchvision.transforms as transforms

import torch
from PIL import Image
import os
from tqdm import tqdm


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, frames_per_clip=16, transform=None):
        self.root_dir = root_dir
        self.frames_per_clip = frames_per_clip
        self.transform = transform

        self.classes = os.listdir(root_dir)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # following loops through each class directory and video file in the dataset, and extract a clip of frames from each video.
        #For each frame in the clip, it loads the image from file using the Image class from the Python Imaging Library (PIL),
        #applies the optional transform, and adds it to the clip. If the clip has at least frames_per_clip frames,
        #it is added to a list called clips along with the class name
        print('Transforming Data...')
        print(f"Total Classes = {len(self.classes)}" )
        self.clips = []
        for class_name in tqdm(self.classes, leave=False):
            print(f"working on: {class_name}")
            class_dir = os.path.join(root_dir, class_name)
            for video_file in tqdm(os.listdir(class_dir), desc='Videos', leave=False):
                video_path = os.path.join(class_dir, video_file)
                clip = []
                for frame_file in tqdm(os.listdir(video_path),desc='Frames', leave=False):
                    frame_path = os.path.join(video_path, frame_file)
                    frame = Image.open(frame_path)
                    if self.transform:
                        frame = self.transform(frame)
                    clip.append(frame)
                if len(clip) >= frames_per_clip:
                    self.clips.append((clip[:frames_per_clip], class_name))
        print('Done Transformation.')

    def __len__(self):
        return len(self.clips) #returns the number of clips in the dataset

    def __getitem__(self, idx):
        clip, class_name = self.clips[idx]
        clip = torch.stack(clip, dim=0)
        label = self.class_to_idx[class_name]
        return clip, label
        #returns the idx-th clip in the dataset, along with its label.
        #The clip is returned as a tensor with dimensions (frames_per_clip, C, H, W), where C, H, and W are the number of channels, height, and width of each frame, respectively