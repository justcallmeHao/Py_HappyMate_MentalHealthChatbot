from typing import Any

import numpy as np
import pygame
from PIL.ImageFile import Image
import time
import cv2
import os

from numpy import ndarray

import ad_enum
from dooh_player.schedule_manager import ScheduleManager
from ad_enum import AdType

class Preprocessor:
    def __init__(self):
        self.schedule_manager = ScheduleManager()
        self.schedule = self.schedule_manager.load_schedule("schedule.json")

    # ---------------PREPROCESSING------------------
    def schedule_processing(self) -> list[tuple[AdType,list[list[np.ndarray]], int]] | None:
        try:
            sequence = []
            for ad_item in self.schedule:
                if ad_item.get("type") == "image":
                    sequence.append(
                        (AdType.IMAGE,
                         [self.image_preprocessing(
                             ad_item.get("file"))],
                         ad_item.get("duration")))
                elif ad_item.get("type") == "video":
                    sequence.append(
                        (AdType.VIDEO,
                         [self.video_preprocessing(
                             ad_item.get("file"),
                             ad_item.get("duration"))],
                         ad_item.get("duration")))
                elif ad_item.get("type") == "split":
                    sequence.append(
                        (AdType.SPLIT,
                         self.split_preprocessing(
                             ad_item.get("image_file"),
                             ad_item.get("video_file"),
                             ad_item.get("duration")),
                         ad_item.get("duration")))
            return sequence
        except Exception as e:
            print(f"Error processing schedule: {e}")
            return None

    def image_preprocessing(self, file_path: str) -> list[ndarray] | None:
        try:
            return [pygame.surfarray.array3d(pygame.image.load(file_path))]
        except Exception as e:
            print(f"Skipped preprocessing image at {file_path}: {e}")
    def video_preprocessing(self, file_path: str, duration: int) -> list[ndarray] | None:
        try:
            video = cv2.VideoCapture(file_path)
            if not video.isOpened():
                raise Exception(f"Failed to open")
            success, frame = video.read()
            playing = success
            fps = 30 # To be edited for customising
            max_frames = int(duration * fps)
            frames = []
            frame_count = 0
            # Render until the end of duration
            while playing and frame_count < max_frames:
                success, frame = video.read()
                if not success:
                    break
                # Convert from BGR (OpenCV default) to RGB (for display or Pygame use)
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_count += 1
            return frames
        except Exception as e:
            print(f"Skipped preprocessing video at {file_path}: {e}")

    def split_preprocessing(self, img_paths: list[str], vid_paths: list[str], duration) -> list[list[ndarray]] | None:
        try:
            split_elements = []
            for img in img_paths:
                split_elements.append(self.image_preprocessing(img))
            for vid in vid_paths:
                split_elements.append(self.video_preprocessing(vid, duration))
            return split_elements
        except Exception as e:
            print(f"Error displaying split-screen: {e}")
