import pickle
import pprint
import glob
from read_data import *
from feature_module import *


def detect_lane_change(lane_center, cur_y, lane_width, car_height):
    delta_y = abs(lane_center - cur_y)
    relative_diff = delta_y / car_height
    if(relative_diff < 0.25):
        return True
    else:
        return False


tracks_csv = read_tracks_csv("data/01_tracks.csv")
tracks_meta = read_tracks_meta("data/01_tracksMeta.csv")
recording_meta = read_recording_meta("data/01_recordingMeta.csv")

lane_changing_ids = []
lane_keeping_ids = []
for key in tracks_meta:
    if(tracks_meta[key][NUMBER_LANE_CHANGES] > 0):
        lane_changing_ids.append(key)
    else:
        lane_keeping_ids.append(key)

print(lane_changing_ids)

lanes_info = {}
lane_num = len(recording_meta[UPPER_LANE_MARKINGS]) + \
    len(recording_meta[LOWER_LANE_MARKINGS]) - 2
if lane_num == 4:
    lanes_info[2] = recording_meta[UPPER_LANE_MARKINGS][0]
    lanes_info[3] = recording_meta[UPPER_LANE_MARKINGS][1]
    lanes_info[5] = recording_meta[LOWER_LANE_MARKINGS][0]
    lanes_info[6] = recording_meta[LOWER_LANE_MARKINGS][1]
    lane_width = ((lanes_info[3] - lanes_info[2]) +
                  (lanes_info[6] - lanes_info[5])) / 2
elif lane_num == 6:
    lanes_info[2] = recording_meta[UPPER_LANE_MARKINGS][0]
    lanes_info[3] = recording_meta[UPPER_LANE_MARKINGS][1]
    lanes_info[4] = recording_meta[UPPER_LANE_MARKINGS][2]
    lanes_info[6] = recording_meta[LOWER_LANE_MARKINGS][0]
    lanes_info[7] = recording_meta[LOWER_LANE_MARKINGS][1]
    lanes_info[8] = recording_meta[LOWER_LANE_MARKINGS][2]
    lane_width = ((lanes_info[3] - lanes_info[2]) + (lanes_info[4] - lanes_info[3]) +
                  (lanes_info[7] - lanes_info[6]) + (lanes_info[8] - lanes_info[7])) / 4
else:
    raise Exception("Damn it")

lane_width = round(lane_width, 2)

# id: list of list of feature module (multiple lane changes)
result = {}

for i in lane_changing_ids:
    print("for car:", i)
    # for each car:
    last_boundary = 0
    changing_pairs_list = []
    for frame_num in range(1, len(tracks_csv[i][FRAME])):
        if tracks_csv[i][LANE_ID][frame_num] != tracks_csv[i][LANE_ID][frame_num-1]:
            original_lane = tracks_csv[i][LANE_ID][frame_num-1]
            new_lane = tracks_csv[i][LANE_ID][frame_num]
            # calculate the starting frame
            starting_frame = frame_num - 1
            while starting_frame > last_boundary:
                if detect_lane_change(lanes_info[original_lane], tracks_csv[i][Y][starting_frame], lane_width, tracks_meta[i][HEIGHT]):
                    break
                starting_frame -= 1
            # calculate the ending frae
            ending_frame = frame_num
            last_boundary = ending_frame
            print(starting_frame, ending_frame)
            changing_pairs_list.append((starting_frame, ending_frame))

    for pair in changing_pairs_list:
        start_idx = pair[0]
        end_idx = pair[1]
        original_lane = tracks_csv[i][LANE_ID][start_idx]
        print("=================================================")
        for frame_num in range(start_idx, end_idx+1):
            # construct the object
            # unique_id, left_lane_exist, right_lane_exist, delta_y, x_velocity, y_velocity, car_type
            # TODO: update index 01
            unique_id = "01-" + str(i)
            # TODO: left lane exist & right lane exist
            delta_y = abs(tracks_csv[i][Y][frame_num] -
                          lanes_info[original_lane])
            x_velocity = tracks_csv[i][X_VELOCITY][frame_num]
            y_velocity = tracks_csv[i][Y_VELOCITY][frame_num]
            car_type = tracks_meta[i][CLASS]
            print(unique_id, delta_y, x_velocity, y_velocity, car_type)
