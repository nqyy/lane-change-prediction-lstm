from read_data import *
import random


FRAME_TAKEN = 50  # number of states to construct features
FRAME_BEFORE = 12  # frame taken before the lane change
FRAME_BEFORE_FLAG = False # use FRAME_BEFORE or not

def run(number):
    '''
    This function runs the data processing code and output to pickle files
    '''
    # read from 3 files
    tracks_csv = read_tracks_csv("data/" + number + "_tracks.csv")
    tracks_meta = read_tracks_meta("data/" + number + "_tracksMeta.csv")
    recording_meta = read_recording_meta(
        "data/" + number + "_recordingMeta.csv")

    # figure out the lane changing cars and lane keeping cars
    lane_changing_ids = []
    lane_keeping_ids = []
    for key in tracks_meta:
        if(tracks_meta[key][NUMBER_LANE_CHANGES] > 0):
            lane_changing_ids.append(key)
        else:
            lane_keeping_ids.append(key)

    # get the lane information
    lanes_info = {}
    lane_num = len(recording_meta[UPPER_LANE_MARKINGS]) + \
        len(recording_meta[LOWER_LANE_MARKINGS]) - 2
    if lane_num == 4:
        # 4 lanes
        lanes_info[2] = recording_meta[UPPER_LANE_MARKINGS][0]
        lanes_info[3] = recording_meta[UPPER_LANE_MARKINGS][1]
        lanes_info[5] = recording_meta[LOWER_LANE_MARKINGS][0]
        lanes_info[6] = recording_meta[LOWER_LANE_MARKINGS][1]
        lane_width = ((lanes_info[3] - lanes_info[2]) +
                      (lanes_info[6] - lanes_info[5])) / 2
    elif lane_num == 6:
        # 6 lanes
        lanes_info[2] = recording_meta[UPPER_LANE_MARKINGS][0]
        lanes_info[3] = recording_meta[UPPER_LANE_MARKINGS][1]
        lanes_info[4] = recording_meta[UPPER_LANE_MARKINGS][2]
        lanes_info[6] = recording_meta[LOWER_LANE_MARKINGS][0]
        lanes_info[7] = recording_meta[LOWER_LANE_MARKINGS][1]
        lanes_info[8] = recording_meta[LOWER_LANE_MARKINGS][2]
        lane_width = ((lanes_info[3] - lanes_info[2]) + (lanes_info[4] - lanes_info[3]) +
                      (lanes_info[7] - lanes_info[6]) + (lanes_info[8] - lanes_info[7])) / 4
    elif lane_num == 7:
        # 7 lanes: track 58 ~ 60
        lanes_info[2] = recording_meta[UPPER_LANE_MARKINGS][0]
        lanes_info[3] = recording_meta[UPPER_LANE_MARKINGS][1]
        lanes_info[4] = recording_meta[UPPER_LANE_MARKINGS][2]
        lanes_info[5] = recording_meta[UPPER_LANE_MARKINGS][3]
        lanes_info[7] = recording_meta[LOWER_LANE_MARKINGS][0]
        lanes_info[8] = recording_meta[LOWER_LANE_MARKINGS][1]
        lanes_info[9] = recording_meta[LOWER_LANE_MARKINGS][2]
        lane_width = ((lanes_info[3] - lanes_info[2]) + (lanes_info[4] - lanes_info[3]) + (
            lanes_info[5] - lanes_info[4]) + (lanes_info[8] - lanes_info[7]) + (lanes_info[9] - lanes_info[8])) / 5
    else:
        print("Error: Invalid input -", number)

    def determine_lane_exist(cur_lane):
        '''
        return: left_exist, right_exist 
        Have to do this shit in a hardcoded way to determine the existence of neighbor lanes.
        '''
        if lane_num == 4:
            if cur_lane == 2 or cur_lane == 6:
                return 1, 0
            else:
                return 0, 1
        elif lane_num == 6:
            if cur_lane == 2 or cur_lane == 8:
                return 1, 0
            elif cur_lane == 3 or cur_lane == 7:
                return 1, 1
            else:
                return 0, 1
        elif lane_num == 7:
            if cur_lane == 2 or cur_lane == 9:
                return 1, 0
            elif cur_lane == 3 or cur_lane == 4 or cur_lane == 8:
                return 1, 1
            else:
                return 0, 1

    def construct_features(i, frame_num, original_lane):
        '''
        Construct all the features for the RNN to train:
        Here is the list:
        Difference of the ego car’s Y position and the lane center: ΔY
        Ego car’s X velocity: Vx
        Ego car’s Y velocity: Vy
        Ego car’s X acceleration: Ax
        Ego car’s Y acceleration: Ay
        Ego car type: T
        TTC of preceding car: TTCp
        TTC of following car: TTCf
        TTC of left preceding car: TTClp
        TTC of left alongside car: TTCla
        TTC of left following car: TTClf
        TTC of right preceding car: TTCrp
        TTC of right alongside car: TTCra
        TTC of right following car: TTCrf
        '''
        going = 0  # 1 left, 2 right
        if lane_num == 4:
            if original_lane == 2 or original_lane == 3:
                going = 1
            else:
                going = 2
        else:
            if original_lane == 2 or original_lane == 3 or original_lane == 4 or original_lane == 5:
                going = 1
            else:
                going = 2
        cur_feature = {}
        cur_feature["left_lane_exist"], cur_feature["right_lane_exist"] = determine_lane_exist(
            original_lane)

        # We need to consider the fact that right/left are different for top/bottom lanes.
        # top lanes are going left      <----
        # bottom lanes are going right  ---->
        # left -> negative, right -> positive
        if going == 1:
            cur_feature["delta_y"] = tracks_csv[i][Y][frame_num] - \
                lanes_info[original_lane]  # up
            cur_feature["y_velocity"] = -tracks_csv[i][Y_VELOCITY][frame_num]
            cur_feature["y_acceleration"] = - \
                tracks_csv[i][Y_ACCELERATION][frame_num]
        else:
            cur_feature["delta_y"] = lanes_info[original_lane] - \
                tracks_csv[i][Y][frame_num]  # down
            cur_feature["y_velocity"] = tracks_csv[i][Y_VELOCITY][frame_num]
            cur_feature["y_acceleration"] = tracks_csv[i][Y_ACCELERATION][frame_num]

        cur_feature["x_velocity"] = tracks_csv[i][X_VELOCITY][frame_num]
        cur_feature["x_acceleration"] = tracks_csv[i][X_ACCELERATION][frame_num]
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
                if going == 1:
                    # going left (up)
                    if cur_x > target_x:
                        ttc = (cur_x - target_x) / (cur_v - target_v)
                    else:
                        ttc = (target_x - cur_x) / (target_v - cur_v)
                else:
                    # going right (down)
                    if cur_x > target_x:
                        ttc = (cur_x - target_x) / (target_v - cur_v)
                    else:
                        ttc = (target_x - cur_x) / (cur_v - target_v)
                if ttc < 0:
                    return 99999
                else:
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

    def detect_lane_change(lane_center, cur_y, lane_width, car_height):
        delta_y = abs(lane_center - cur_y)
        relative_diff = delta_y / car_height
        if(relative_diff < 0.5):
            return True
        else:
            return False

    def determine_change_direction(ori_laneId, new_laneId):
        '''
        return 1 upon left change
        return 2 upon right change
        '''
        if lane_num == 4:
            if (ori_laneId == 2 and new_laneId == 3) or (ori_laneId == 6 and new_laneId == 5):
                return 1
            else:
                return 2
        else:
            # left:
            if (ori_laneId == 2 and new_laneId == 3) or (ori_laneId == 4 and new_laneId == 5) \
                or (ori_laneId == 3 and new_laneId == 4) or (ori_laneId == 7 and new_laneId == 6) \
                    or (ori_laneId == 8 and new_laneId == 7) or (ori_laneId == 9 and new_laneId == 8):
                return 1
            else:
                return 2

    # list of list of features
    result = []

    for i in lane_changing_ids:
        # for each car:
        last_boundary = 0
        # list of (starting index, ending index, direction)
        changing_tuple_list = []
        # 1. determine the frame we want to use
        for frame_num in range(1, len(tracks_csv[i][FRAME])):
            if tracks_csv[i][LANE_ID][frame_num] != tracks_csv[i][LANE_ID][frame_num-1]:
                original_lane = tracks_csv[i][LANE_ID][frame_num-1]
                new_lane = tracks_csv[i][LANE_ID][frame_num]
                direction = determine_change_direction(original_lane, new_lane)
                # calculate the starting frame
                starting_change = frame_num - 1
                while starting_change > last_boundary:
                    if detect_lane_change(lanes_info[original_lane], tracks_csv[i][Y][starting_change], lane_width, tracks_meta[i][HEIGHT]):
                        break
                    starting_change -= 1
                # calculate the starting and ending frame
                if FRAME_BEFORE_FLAG:
                    starting_point = starting_change - FRAME_TAKEN - FRAME_BEFORE
                    ending_point = starting_change - FRAME_BEFORE
                else:
                    starting_point = starting_change - FRAME_TAKEN
                    ending_point = starting_change
                if starting_point > last_boundary:
                    changing_tuple_list.append(
                        (starting_point, ending_point, direction))
                last_boundary = frame_num
        # add those frames' features
        for pair in changing_tuple_list:
            # for each lane change instance
            cur_change = []
            start_idx = pair[0]
            end_idx = pair[1]
            direction = pair[2]
            original_lane = tracks_csv[i][LANE_ID][start_idx]
            # continue for out of boundary cases
            if original_lane not in lanes_info:
                continue
            for frame_num in range(start_idx, end_idx):
                # construct the object
                cur_change.append(construct_features(
                    i, frame_num, original_lane))
            # add to the result
            result.append((cur_change, direction))

    change_num = len(result)

    if len(lane_keeping_ids) > len(result):
        # make the lane keeping size the same as lane changing
        lane_keeping_ids = random.sample(lane_keeping_ids, len(result))

    for i in lane_keeping_ids:
        cur_change = []
        original_lane = tracks_csv[i][LANE_ID][0]
        fail = False
        for frame_num in range(1, FRAME_TAKEN+1):
            try:
                cur_change.append(construct_features(
                    i, frame_num, original_lane))
            except:
                # handle exception where the total frame is less than FRAME_TAKEN
                fail = True
                break
        if not fail:
            result.append((cur_change, 0))

    return result, change_num
