from moviepy.video.io.VideoFileClip import VideoFileClip

# Path to the input video clip
input_video_path = 'archery.mp4'

# Time stamp to start the segment (in seconds)
start_time = 5  # Adjust this to your desired starting time

# Time duration of the segment to extract (in seconds)
duration = 2   # Adjust this to your desired segment duration

# Output path for the extracted segment
output_segment_path = 'output_segment.mp4'

# Load the video clip
video_clip = VideoFileClip(input_video_path)

# Extract the segment based on the time stamp and duration
segment = video_clip.subclip(start_time, start_time + duration)

# Write the extracted segment to the output file
segment.write_videofile(output_segment_path, codec='libx264')

import torchvision
reader = torchvision.io.VideoReader(input_video_path, "video")
reader_md = reader.get_metadata()
l = torchvision.io.read_video_timestamps(input_video_path, pts='sec')
print(float(l[0][299]))

t = torchvision.io.read_video(input_video_path,5, 7, pts_unit='sec')

torchvision.io.write_video("1_"+output_segment_path, t, 30, video_codec='libx264')

torchvision.io.write_video("1_"+output_segment_path, t[0], 30, video_codec='libx264')

import decord as de

vr = de.VideoReader(input_video_path)

key_indices = vr.get_key_indices()
key_frames = vr.get_batch(key_indices)
print(key_frames.shape)

plt.imshow(key_frames.asnumpy()[0])
plt.show()

frames = vr.get_batch(range(0, len(vr) - 1, 5))
print(frames.shape)

// read byte array into vr
// calculate start and end frame count_num_classes
// load part 1, part (anomaly) and part 3 tensors
// write 3 tensors to a file
