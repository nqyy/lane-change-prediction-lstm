import pickle
import pprint
import glob
from read_data import *
from feature_module import *


# read from 3 files
tracks_csv = read_tracks_csv("data/01_tracks.csv")
tracks_meta = read_tracks_meta("data/01_tracksMeta.csv")
recording_meta = read_recording_meta("data/01_recordingMeta.csv")

# figure out the lane changing cars and lane keeping cars
lane_changing_ids = []
lane_keeping_ids = []
for key in tracks_meta:
    if(tracks_meta[key][NUMBER_LANE_CHANGES] > 0):
        lane_changing_ids.append(key)
    else:
        lane_keeping_ids.append(key)

print("lane changing cars:", lane_changing_ids)

# get the lane information
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


def determine_lane_exist(cur_lane):
    '''
    return: left_exist, right_exist 
    Have to do this shit in a hardcoded way
    '''
    if lane_num == 4:
        if cur_lane == 2 or cur_lane == 6:
            # right lane
            return True, False
        else:
            # left lane
            return False, True
    elif lane_num == 6:
        if cur_lane == 2 or cur_lane == 8:
            # right lane
            return True, False
        elif cur_lane == 3 or cur_lane == 7:
            # middle lane
            return True, True
        else:
            # left lane
            return False, True
    else:
        raise Exception("Damn it")


def construct_features(i, frame_num, original_lane):
    cur_feature = {}
    # TODO: update index 01
    cur_feature["unique_id"] = "01-" + str(i)
    cur_feature["left_lane_exist"], cur_feature["right_lane_exist"] = determine_lane_exist(
        original_lane)
    cur_feature["delta_y"] = abs(
        tracks_csv[i][Y][frame_num] - lanes_info[original_lane])
    cur_feature["x_velocity"] = tracks_csv[i][X_VELOCITY][frame_num]
    cur_feature["y_velocity"] = tracks_csv[i][Y_VELOCITY][frame_num]
    cur_feature["x_acceleration"] = tracks_csv[i][X_ACCELERATION][frame_num]
    cur_feature["y_acceleration"] = tracks_csv[i][Y_ACCELERATION][frame_num]
    cur_feature["car_type"] = tracks_meta[i][CLASS]

    def calculate_dx(target_car_id):
        """
        Calculate x pos difference between target car and current car
        """
        if target_car_id != 0:
            # target frame for target car
            target_frame = tracks_meta[i][INITIAL_FRAME] + \
                frame_num - tracks_meta[target_car_id][INITIAL_FRAME]
            preceding_x = tracks_csv[target_car_id][X][target_frame]
            cur_x = tracks_csv[i][X][frame_num]
            return abs(preceding_x - cur_x)
        else:
            return None

    # surrounding cars info
    cur_feature["preceding_dx"] = calculate_dx(
        tracks_csv[i][PRECEDING_ID][frame_num])
    cur_feature["following_dx"] = calculate_dx(
        tracks_csv[i][FOLLOWING_ID][frame_num])
    cur_feature["left_preceding_dx"] = calculate_dx(
        tracks_csv[i][LEFT_PRECEDING_ID][frame_num])
    cur_feature["left_alongside_dx"] = calculate_dx(
        tracks_csv[i][LEFT_ALONGSIDE_ID][frame_num])
    cur_feature["left_following_dx"] = calculate_dx(
        tracks_csv[i][LEFT_FOLLOWING_ID][frame_num])
    cur_feature["right_preceding_dx"] = calculate_dx(
        tracks_csv[i][RIGHT_PRECEDING_ID][frame_num])
    cur_feature["right_alongside_dx"] = calculate_dx(
        tracks_csv[i][RIGHT_ALONGSIDE_ID][frame_num])
    cur_feature["right_following_dx"] = calculate_dx(
        tracks_csv[i][RIGHT_FOLLOWING_ID][frame_num])

    return cur_feature


# list of list of feature module (multiple lane changes)
result = []


def detect_lane_change(lane_center, cur_y, lane_width, car_height):
    delta_y = abs(lane_center - cur_y)
    relative_diff = delta_y / car_height
    if(relative_diff < 0.25):
        return True
    else:
        return False


for i in lane_changing_ids:
    # print("for car:", i)
    # for each car:
    last_boundary = 0
    changing_pairs_list = []
    # 1. determine the frame we want to use
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
            # calculate the ending frame
            ending_frame = frame_num
            last_boundary = ending_frame
            # print(starting_frame, ending_frame)
            changing_pairs_list.append((starting_frame, ending_frame))
    # add those frames' features
    for pair in changing_pairs_list:
        # for each lane change instance
        cur_change = []
        start_idx = pair[0]
        end_idx = pair[1]
        original_lane = tracks_csv[i][LANE_ID][start_idx]
        # print("=================================================")
        for frame_num in range(start_idx, end_idx+1):
            # construct the object
            cur_change.append(construct_features(i, frame_num, original_lane))
        # add to the result
        result.append(cur_change)

# the stuff we want is in result
f = open('result.pickle', 'wb')
pickle.dump(result, f)
f.close()
