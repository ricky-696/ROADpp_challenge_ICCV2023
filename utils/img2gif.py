import imageio
from moviepy.editor import *

def convert_mp4_to_gif(mp4_file, gif_file):
    video = VideoFileClip(mp4_file)
    video = video.resize((320, 240))
    video = video.set_fps(10)
    video.write_gif(gif_file)

mp4_file = "/home/Ricky/0_Project/ROADpp_challenge_ICCV2023/test.mp4"
gif_file = "T1_demo_epoch_20.gif"

convert_mp4_to_gif(mp4_file, gif_file)
