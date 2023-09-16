import os.path
#
# from moviepy.video.io.VideoFileClip import VideoFileClip
#
# # Path to the input video clip
# input_video_path = 'archery.mp4'
#
# # Time stamp to start the segment (in seconds)
# start_time = 5  # Adjust this to your desired starting time
#
# # Time duration of the segment to extract (in seconds)
# duration = 2   # Adjust this to your desired segment duration
#
# # Output path for the extracted segment
# output_segment_path = 'output_segment.mp4'
#
# # Load the video clip
# video_clip = VideoFileClip(input_video_path)
#
# # Extract the segment based on the time stamp and duration
# segment = video_clip.subclip(start_time, start_time + duration)
#
# # Write the extracted segment to the output file
# segment.write_videofile(output_segment_path, codec='libx264')
#
# import torchvision
# reader = torchvision.io.VideoReader(input_video_path, "video")
# reader_md = reader.get_metadata()
# l = torchvision.io.read_video_timestamps(input_video_path, pts='sec')
# print(float(l[0][299]))
#
# t = torchvision.io.read_video(input_video_path,5, 7, pts_unit='sec')
#
# torchvision.io.write_video("1_"+output_segment_path, t, 30, video_codec='libx264')
#
# torchvision.io.write_video("1_"+output_segment_path, t[0], 30, video_codec='libx264')

import decord as de
import decord
import io
import torchvision
import pandas
import Data_Loader
#
# vr = de.VideoReader(input_video_path)
#
# key_indices = vr.get_key_indices()
# key_frames = vr.get_batch(key_indices)
# print(key_frames.shape)
#
# plt.imshow(key_frames.asnumpy()[0])
# plt.show()
#
# frames = vr.get_batch(range(0, len(vr) - 1, 5))
# print(frames.shape)
#
# # // read byte array into vr
# # // calculate start and end frame count_num_classes
# # // load part 1, part (anomaly) and part 3 tensors
# # // write 3 tensors to a file
#
# fps = 30
# frames = vr.get_batch(range(5*fps, 7*fps)).asnumpy()
# torchvision.io.write_video("1a_"+output_segment_path, frames, 30, video_codec='libx264')

import pandas as pd
df_test_times = pd.read_csv('.\data\Temporal_Anomaly_Annotation_for_Testing_Videos.txt', sep= '  ', header=None)

folder_path = "test_videos"

def split_videos():
    # run Data_Loader before this

    dataset, _ = Data_Loader.get_dataset_and_loader()
    # Iterate through all rows using iterrows()
    for index, row in df_test_times.iterrows():
        # print(index, row[0], row[1])
        samples = dataset.samples
        for s in samples:
            if s[0].name == row[0] and row[1] != 'Normal':
                print(index, row[0], row[1])
                v = s[0]
                video = v.read_bytes()
                file_obj = io.BytesIO(video)
                vr = decord.VideoReader(file_obj)

                # begin of first anomaly segment
                b_1 = row[2]
                # end of first anomaly segment
                e_1 = row[3]
                # begin of second anomaly segment, if one exists i.e. is not -1
                b_2 = row[4]
                # end of second anomaly segment, if one exists i.e. is not -1
                e_2 = row[5]

                # correct end frames if needed
                if e_1 > len(vr)-1:
                    e_1 = len(vr)-1
                if e_2 > len(vr)-1:
                    e_2 = len(vr)-1

                # all frames before start of first anomaly
                frames_b = vr.get_batch(range(0, b_1)).asnumpy()
                # all frames in anomaly segment
                frames_e = vr.get_batch(range(e_1 + 1, len(vr))).asnumpy()
                # all frames in normal segment after anomaly segment above
                frames_a = vr.get_batch(range(b_1, e_1 + 1)).asnumpy()

                torchvision.io.write_video(folder_path + "\\begin_1_" + row[0], frames_b, 30, video_codec='libx264')
                torchvision.io.write_video(folder_path + "\\anomaly_1_" + row[0], frames_a, 30, video_codec='libx264')
                torchvision.io.write_video(folder_path + "\\end_1_" + row[0], frames_e, 30, video_codec='libx264')

                if b_2 != -1:
                    frames_b = vr.get_batch(range(e_1+1, b_2)).asnumpy()
                    frames_a = vr.get_batch(range(b_2, e_2 + 1)).asnumpy()
                    # Only if the end of the second anomaly segment does not span till the end of the video, then another normal segment exists after the second anomaly
                    if e_2 < len(vr) - 1:
                        frames_e = vr.get_batch(range(e_2 + 1, len(vr))).asnumpy()
                        torchvision.io.write_video(folder_path + "\\end_2_" + row[0], frames_e, 30, video_codec='libx264')

                    torchvision.io.write_video(folder_path + "\\begin_2_+" + row[0], frames_b, 30, video_codec='libx264')
                    torchvision.io.write_video(folder_path + "\\anomaly_2_" + row[0], frames_a, 30, video_codec='libx264')



def main():
    print("Calling split videos...")
    split_videos()
    # You can put your main code here

# Check if the script is being run directly (not imported as a module)
if __name__ == "__main__":
    main()
#
# samples = dataset.samples
# for s in samples:
#     if s[0].name == 'Abuse028_x264.mp4':
#         v=s[0]
#
# b=165; e=240
#
# video = v.read_bytes()
# file_obj = io.BytesIO(video)
# vr = decord.VideoReader(file_obj)
#
# frames_a = vr.get_batch(range(b,e)).asnumpy()
# frames_b = vr.get_batch(range(0,b)).asnumpy()
# frames_e = vr.get_batch(range(e+1,1412)).asnumpy()
# frames_a = vr.get_batch(range(b,e+1)).asnumpy()
# torchvision.io.write_video("b_"+ "Abuse028_x264.mp4", frames_b, 30, video_codec='libx264')
# torchvision.io.write_video("a_"+ "Abuse028_x264.mp4", frames_a, 30, video_codec='libx264')
# #torchvision.io.write_video(r'.\test_videos\' + "e_"+ "Abuse028_x264.mp4", frames_e, 30, video_codec='libx264')