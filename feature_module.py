

class feature_module():
    def __init__(self, unique_id, left_lane_exist, right_lane_exist, delta_y, x_velocity, y_velocity, car_type):
        self.unique_id = unique_id
        self.left_lane_exist = left_lane_exist
        self.right_lane_exist = right_lane_exist
        self.delta_y = delta_y
        self.x_velocity = x_velocity
        self.y_velocity = y_velocity
        self.car_type = car_type