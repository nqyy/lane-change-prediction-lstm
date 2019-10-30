

class feature_module():
    """
    Features to be used for training
    """

    def __init__(self, unique_id, left_lane_exist, right_lane_exist,
                 delta_y, x_velocity, y_velocity, x_acceleration, y_acceleration, car_type,
                 preceding_dx, following_dx, 
                 left_preceding_dx, left_alongside_dx, left_following_dx,
                 right_preceding_dx, right_alongside_dx, right_following_dx):
        self.unique_id = unique_id
        self.left_lane_exist = left_lane_exist
        self.right_lane_exist = right_lane_exist
        self.delta_y = delta_y # diff between the car and the lane
        self.x_velocity = x_velocity
        self.y_velocity = y_velocity
        self.x_acceleration = x_acceleration
        self.y_acceleration = y_acceleration
        self.car_type = car_type
        # surrounding car info
        self.preceding_dx = preceding_dx
        self.following_dx = following_dx
        self.left_preceding_dx = left_preceding_dx
        self.left_alongside_dx = left_alongside_dx
        self.left_following_dx = left_following_dx
        self.right_preceding_dx = right_preceding_dx
        self.right_alongside_dx = right_alongside_dx
        self.right_following_dx = right_following_dx
