"""
Author: Dikshant Gupta
Time: 25.07.21 09:57
"""


class Config:
    PI = 3.14159

    simulation_step = 0.05  # 0.008
    # sensor_simulation_step = '0.5'
    synchronous = True
    segcam_fov = '90'
    segcam_image_x = '400'  # '1280'
    segcam_image_y = '400'  # '720'

    # # grid_size = 2  # grid size in meters
    # speed_limit = 50
    # max_steering_angle = 1.22173  # 70 degrees in radians
    # occupancy_grid_width = '1920'
    # occupancy_grid_height = '1080'

    # location_threshold = 1.0

    ped_speed_range = [0.6, 2.0]
    ped_distance_range = [0, 40]
    # car_speed_range = [6, 9]
    # scenarios = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
    scenarios = ['01']

    # val_scenarios = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
    val_scenarios = ['01']
    val_ped_speed_range = ([0.2, 0.5], [2.1, 2.8])
    val_ped_distance_range = [4.25, 49.25]
    # val_car_speed_range = [6, 9]

    # test_scenarios = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    test_scenarios = ['01']
    test_ped_speed_range = [0.25, 2.85]

    test_ped_distance_range = [4.75, 49.75]

    # Simulator Parameters
    host = 'localhost'
    port = 2000
    width = 1280
    height = 720
    # TODO: remove it from here to commmand line parameter
    display = True
    filter = 'vehicle.audi.tt'
    rolename = 'hero'
    gama = 1.7
    despot_port = 1245
    N_DISCRETE_ACTIONS = 3
    min_speed = 3 # in kmph
    max_speed = 25   # in kmph
    hit_penalty = 1000
    goal_reward = 1000
    braking_penalty = 1
    over_speeding_penalty = -0.5


    # utils_parameters
    model_checkpointing_interval = 4
    max_checkpoints = 20

    # Target Entropy Scheduler Parameter
    exp_win_discount = 0.999
    avg_threshold = 0.01
    std_threshold = 0.05
    entropy_discount_factor =0.9