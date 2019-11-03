from read_data import *
import random


def run(number):
    # read from 3 files
    tracks_csv = read_tracks_csv("data/" + number + "_tracks.csv")
    tracks_meta = read_tracks_meta("data/" + number + "_tracksMeta.csv")
    recording_meta = read_recording_meta(
        "data/" + number + "_recordingMeta.csv")

    FRAME_TAKEN = recording_meta[FRAME_RATE]
    # figure out the lane changing cars and lane keeping cars
    lane_changing_ids = []
    lane_keeping_ids = []
    for key in tracks_meta:
        if(tracks_meta[key][NUMBER_LANE_CHANGES] > 0):
            lane_changing_ids.append(key)
        else:
            lane_keeping_ids.append(key)

    # print("lane changing cars:", lane_changing_ids)

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
                return 1, -1
            else:
                # left lane
                return -1, 1
        elif lane_num == 6:
            if cur_lane == 2 or cur_lane == 8:
                # right lane
                return 1, -1
            elif cur_lane == 3 or cur_lane == 7:
                # middle lane
                return 1, 1
            else:
                # left lane
                return -1, 1
        else:
            raise Exception("Damn it")

    def construct_features(i, frame_num, original_lane):
        cur_feature = {}
        # cur_feature["unique_id"] = "01-" + str(i)
        cur_feature["left_lane_exist"], cur_feature["right_lane_exist"] = determine_lane_exist(
            original_lane)
        cur_feature["delta_y"] = abs(
            tracks_csv[i][Y][frame_num] - lanes_info[original_lane])
        cur_feature["x_velocity"] = tracks_csv[i][X_VELOCITY][frame_num]
        cur_feature["y_velocity"] = tracks_csv[i][Y_VELOCITY][frame_num]
        cur_feature["x_acceleration"] = tracks_csv[i][X_ACCELERATION][frame_num]
        cur_feature["y_acceleration"] = tracks_csv[i][Y_ACCELERATION][frame_num]
        cur_feature["car_type"] = 1 if tracks_meta[i][CLASS] == "Car" else -1

        def calculate_ttc(target_car_id):
            """
            Calculate time to collision of target car and current car
            """
            if target_car_id != 0:
                target_frame = tracks_meta[i][INITIAL_FRAME] + \
                    frame_num - tracks_meta[target_car_id][INITIAL_FRAME]
                target_x = tracks_csv[target_car_id][X][target_frame]
                cur_x = tracks_csv[i][X][frame_num]
                target_v = tracks_csv[target_car_id][X_VELOCITY][target_frame]
                cur_v = tracks_csv[i][X_VELOCITY][frame_num]
                if target_v == cur_v:
                    return 99999
                if cur_x > target_x:
                    # if cur car is in front of target car
                    ttc = (cur_x - target_x) / (target_v - cur_v)
                else:
                    # if target car is in front of cur car
                    ttc = (target_x - cur_x) / (cur_v - target_v)
                return ttc
            else:
                return 99999

        # surrounding cars info
        cur_feature["preceding_ttc"] = calculate_ttc(
            tracks_csv[i][PRECEDING_ID][frame_num])
        cur_feature["following_ttc"] = calculate_ttc(
            tracks_csv[i][FOLLOWING_ID][frame_num])
        cur_feature["left_preceding_ttc"] = calculate_ttc(
            tracks_csv[i][LEFT_PRECEDING_ID][frame_num])
        cur_feature["left_alongside_ttc"] = calculate_ttc(
            tracks_csv[i][LEFT_ALONGSIDE_ID][frame_num])
        cur_feature["left_following_ttc"] = calculate_ttc(
            tracks_csv[i][LEFT_FOLLOWING_ID][frame_num])
        cur_feature["right_preceding_ttc"] = calculate_ttc(
            tracks_csv[i][RIGHT_PRECEDING_ID][frame_num])
        cur_feature["right_alongside_ttc"] = calculate_ttc(
            tracks_csv[i][RIGHT_ALONGSIDE_ID][frame_num])
        cur_feature["right_following_ttc"] = calculate_ttc(
            tracks_csv[i][RIGHT_FOLLOWING_ID][frame_num])

        ret = tuple(cur_feature.values())
        return ret

    # list of list of features
    result = []

    def detect_lane_change(lane_center, cur_y, lane_width, car_height):
        delta_y = abs(lane_center - cur_y)
        relative_diff = delta_y / car_height
        if(relative_diff < 0.5):
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
                starting_change = frame_num - 1
                while starting_change > last_boundary:
                    if detect_lane_change(lanes_info[original_lane], tracks_csv[i][Y][starting_change], lane_width, tracks_meta[i][HEIGHT]):
                        break
                    starting_change -= 1
                # calculate the starting and ending frame
                starting_point = starting_change - FRAME_TAKEN
                ending_point = starting_change
                if starting_point > last_boundary:
                    # print(starting_point, ending_point)
                    # print(tracks_csv[i][Y][starting_point], tracks_csv[i][Y][ending_point])
                    changing_pairs_list.append((starting_point, ending_point))
                last_boundary = frame_num

        # add those frames' features
        for pair in changing_pairs_list:
            # for each lane change instance
            cur_change = []
            start_idx = pair[0]
            end_idx = pair[1]
            original_lane = tracks_csv[i][LANE_ID][start_idx]
            # print("=================================================")
            for frame_num in range(start_idx, end_idx):
                # construct the object
                cur_change.append(construct_features(
                    i, frame_num, original_lane))
            # add to the result
            result.append((cur_change, 1))

    if len(lane_keeping_ids) > len(result):
        # make the lane keeping size the same as lane changing
        lane_keeping_ids = random.sample(lane_keeping_ids, len(result))

    for i in lane_keeping_ids:
        cur_change = []
        original_lane = tracks_csv[i][LANE_ID][0]
        for frame_num in range(1, FRAME_TAKEN+1):
            cur_change.append(construct_features(i, frame_num, original_lane))
        result.append((cur_change, 0))

    return result
