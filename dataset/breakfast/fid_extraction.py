import os
import shutil

video_path = '/home/tuan/Documents/Code/video2command/IIT-V2C/avi_video'
train_path = '/home/tuan/Documents/Code/video2command/IIT-V2C/train.txt'
test_path = '/home/tuan/Documents/Code/video2command/IIT-V2C/test.txt'
video_list = [video.replace(".avi","") for video in os.listdir(video_path)]
# print(video_list)

def combine():
    # Extract train and test text files
    with open(train_path, 'r') as train_file:
        train = train_file.read()

    with open(test_path, 'r') as test_file:
        test = test_file.read()

    # Create combined file
    with open("combine.txt", 'w') as file:
        file.write(train + test)


def fid_ex():
    combine_path = '/home/tuan/Documents/Code/video2command/dataset/breakfast/combine.txt'
    fid_root_path = '/home/tuan/Documents/Code/video2command/subtitle/l30_srt'

    if os.path.isdir(fid_root_path):
        shutil.rmtree(fid_root_path)
    os.makedirs(fid_root_path)

    if os.path.isfile(combine_path):
        with open(combine_path, 'r') as file:
            combine = file.readlines()

    for video_name in video_list:
        fid_path = os.path.join(fid_root_path, f'{video_name}.fid')
        sort_list = [
            combine[idx:idx + 4]
            for idx in range(0, len(combine), 4)
            if combine[idx].startswith(video_name)
        ]

        sort_list.sort(key=lambda x: int(x[0].split('_')[4].strip()), reverse=False)
        flattened_list = [item for sublist in sort_list for item in sublist]

        with open(fid_path, 'w') as f:
            f.writelines(flattened_list)


if __name__ == '__main__':
    fid_ex()
    with open('/home/tuan/Documents/Code/video2command/subtitle/l30_srt/P05_cam01_P05_juice.fid', 'r') as test_file:
        test = test_file.readlines()
    test = "".join(test)
    print(test)