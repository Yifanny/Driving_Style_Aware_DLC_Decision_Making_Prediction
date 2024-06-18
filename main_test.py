# XX_tracks.csv: This file contains all time dependent values for each track,
# 0: The current frame.
# 1: The track's id.
# 2: The x position of the upper left corner of the vehicle's bounding box.
# 3: The y position of the upper left corner of the vehicle's bounding box.
# 4: The width of the bounding box of the vehicle.
# 5: The height of the bounding box of the vehicle.
# 6: The longitudinal velocity in the image coordinate system.
# 7: The lateral velocity in the image coordinate system.
# 8: The longitudinal acceleration in the image coordinate system.
# 9: The lateral acceleration in the image coordinate system
# 10: frontSightDistance - The distance to the end of the recorded highway section in driving direction from the vehicle's center.
# 11: backSightDistance - The distance to the end of the recorded highway section in the opposite driving direction from the vehicle's center.
# 12: The Distance Headway. This value is set to 0, if no preceding vehicle exists.
# 13: The Time Headway. This value is set to 0, if no preceding vehicle exists.
# 14: The Time-to-Collision. This value is set to 0, if no preceding vehicle or valid TTC exists.
# 15: precedingXVelocity - The longitudinal velocity of the preceding in the image coordinate system. This value is set to 0, if no preceding vehicle exists.
# 16: precedingId - The id of the preceding vehicle in the same lane. This value is set to 0, if no preceding vehicle exists.
# 17: followingId - The id of the following vehicle in the same lane. This value is set to 0, if no following vehicle exists.
# 18: leftPrecedingId - The id of the preceding vehicle on the adjacent lane on the left in the direction of travel. This value is set to 0, if no such a vehicle exists.
# 19: leftAlongsideId - The id of the adjacent vehicle on the adjacent lane on the left in the direction of travel. In order for a vehicle to be adjacent and not e.g. preceding, the vehicles must overlap in the longitudinal direction. This value is set to 0, if no such a vehicle exists.
# 20: leftFollowingId - The id of the following vehicle on the adjacent lane on the left in the direction of travel. This value is set to 0, if no such a vehicle exists.
# 21: rightPrecedingId - The id of the preceding vehicle on the adjacent lane on the right in the direction of travel. This value is set to 0, if no such a vehicle exists.
# 22: rightAlsongsideId - The id of the adjacent vehicle on the adjacent lane on the right in the direction of travel. In order for a vehicle to be adjacent and not e.g. preceding, the vehicles must overlap in the longitudinal direction. This value is set to 0, if no such a vehicle exists.
# 23: rightFollowingId - The id of the following vehicle on the adjacent lane on the right in the direction of travel. This value is set to 0, if no such a vehicle exists.
# 24: laneId - The IDs start at 1 and are assigned in ascending order. Since the Lane ids are derived from the positions of the lane markings, the first and last ids typically do not describe any useable lanes.

import numpy as np
import random
import matplotlib.pyplot as plt

"""
tracks = []
tracksMeta = []
for i in range(1, 58):
    this_tracks = []
    if i < 10:
        file_id = "0" + str(i)
    else:
        file_id = str(i)
    print(file_id)
    recordingFile = "data/" + file_id + "_recordingMeta.csv"
    trackMetaFile = "data/" + file_id + "_tracksMeta.csv"
    trackFile = "data/" + file_id + "_tracks.csv"
    trackFile = open(trackFile, "r")
    trackMetaFile = open(trackMetaFile, "r")
    for row in trackFile.readlines():
        tmp = []
        for item in row.split(','):
            try:
                tmp.append(float(item))
            except ValueError:
                continue
        this_tracks.append(tmp)
    del this_tracks[0]
    trackFile.close()
    tracks.append(this_tracks)
    print(len(this_tracks))
    this_trackMeta = []
    for row in trackMetaFile.readlines():
        tmp = []
        for item in row.split(','):
            try:
                tmp.append(float(item))
            except ValueError:
                continue
        this_trackMeta.append(tmp)
    del this_trackMeta[0]
    trackMetaFile.close()
    tracksMeta.append(this_trackMeta)
    print(len(this_trackMeta))
print(len(tracks))
print(len(tracksMeta))



num_lanes = []
new_y_o = []
for track in tracks:
    lane_id = 0
    y_o = 0
    for row in track:
        if row[24] > lane_id:
            lane_id = row[24]
        if row[3] > y_o:
            y_o = row[3]
    num_lanes.append(lane_id)
    new_y_o.append(y_o)
print(num_lanes)
print(np.max(np.array(new_y_o)))

new_x_o = 450.0
new_y_o = 50.0


def takeSecond(elem):
    return elem[1]

for this_tracks in tracks:
    this_tracks.sort(key=takeSecond)
# print(tracks[100000:100010])


# tracks = np.array(tracks)
vehicles = []
# print(tracks.shape)

n = 0
for this_tracks in tracks:
    this_section_vehicles = []
    this_vehicle = []
    for i in range(len(this_tracks) - 1):
        if i % 100000 == 0:
            print(i)
        v_id = this_tracks[i][1]
        if v_id != tracksMeta[n][int(v_id) - 1][0]:
            print("error")
        v_class = tracksMeta[n][int(v_id) - 1][6]
        direction = 2
        if this_tracks[i][24] <= (num_lanes[n] / 2):
            direction = 1
        if direction == 1:
            this_tracks[i][6] = -this_tracks[i][6]
            this_tracks[i][7] = -this_tracks[i][7]
            this_tracks[i][8] = -this_tracks[i][8]
            this_tracks[i][9] = -this_tracks[i][9]
            this_tracks[i][2] = new_x_o - this_tracks[i][2]
            this_tracks[i][3] = new_y_o - this_tracks[i][3]
        if this_tracks[i][2] < 0 or this_tracks[i][3] < 0:
            print(direction, this_tracks[i][2], this_tracks[i][3])

        if this_tracks[i][1] == this_tracks[i + 1][1] and this_tracks[i][1] != 0:
            this_vehicle.append(this_tracks[i])
            if i + 2 == len(this_tracks):
                this_vehicle.append(this_tracks[i + 1])
                this_vehicle.sort()
                this_section_vehicles.append(this_vehicle)
        else:
            if this_tracks[i][1] != 0:
                this_vehicle.append(this_tracks[i])
                # print(type(this_vehicle))
                this_vehicle.sort()
                this_section_vehicles.append(this_vehicle)
                # print(len(vehicles))
            this_vehicle = []
    vehicles.append(this_section_vehicles)
    print(len(this_section_vehicles))
    n += 1
print(len(vehicles))


lc_cases = []
lc_labels = []
n = 0
for this_vehicles in vehicles:
    lc_v_id_list = []
    this_lc_cases = []
    this_lc_labels = []
    for v in this_vehicles:
        v = np.array(v)
        for i in range(len(v)-1):
            if v[i][24] != v[i+1][24]:
                if not (v[i][1], v[i][0]) in lc_v_id_list:
                    if i-100 >= 0 :
                        this_lc_cases.append(v[i-100:i+50])
                        lc_v_id_list.append((v[i][1], v[i][0]))
                        if v[i][24] > v[i+1][24] and v[i][24] > (num_lanes[n]/2):
                            this_lc_labels.append(1)
                        elif v[i][24] < v[i+1][24] and v[i][24] > (num_lanes[n]/2):
                            this_lc_labels.append(2)
                        elif v[i][24] > v[i+1][24] and v[i][24] <= (num_lanes[n]/2):
                            this_lc_labels.append(2)
                        elif v[i][24] < v[i+1][24] and v[i][24] <= (num_lanes[n]/2):
                            this_lc_labels.append(1)
                        else:
                            print("error")
    lc_cases.append(this_lc_cases)
    lc_labels.append(this_lc_labels)
    n += 1
print(len(lc_cases))
s = 0
for v in lc_cases:
    print(len(v))
    s += len(v)
    #plt.plot(v[:,2], v[:,3])
#plt.show()
#print(np.array(lc_cases).shape)
left = 0
right = 0
for v in lc_labels:
    print(len(v))
    left += np.sum(np.array(v)==1)
    right += np.sum(np.array(v)==2)
print(len(lc_labels))
print(s)
print(left)
print(right)


lk_cases = []
lk_labels = []
n = 0
for this_vehicles in vehicles:
    lk_v_id_list = []
    this_lk_cases = []
    this_lk_labels = []
    for v in this_vehicles:
        v = np.array(v)
        for i in range(len(v)-1):
            if v[i][24] != v[i+1][24]:
                break
            else:
                if i == 300:
                    if not (v[i][1], v[i][0]) in lk_v_id_list:
                        this_lk_cases.append(v[i-200:i-150])
                        lk_v_id_list.append((v[i][1], v[i][0]))
                        this_lk_labels.append(0)
    lk_cases.append(this_lk_cases)
    lk_labels.append(this_lk_labels)
    n += 1
print(len(lk_cases))
for v in lk_cases:
    print(len(v))
    # plt.plot(v[:,2], v[:,3])
# plt.show()
print(np.array(lk_cases).shape)
print(len(lk_labels))


def distance(target_v_id, ego_v, frame_id, vehicles):
    target_v_id = int(target_v_id)
    if target_v_id != 0:
        target_v = vehicles[target_v_id - 1]
        for i in range(len(target_v)):
            frame = target_v[i]
            if frame[0] == frame_id:
                if i - 50 >= 0:
                    return np.sqrt((frame[2] - ego_v[2]) ** 2 + (frame[3] - ego_v[3]) ** 2), target_v[i - 50: i]
                elif i == 0:
                    return np.sqrt((frame[2] - ego_v[2]) ** 2 + (frame[3] - ego_v[3]) ** 2), [target_v[0]]
                else:
                    return np.sqrt((frame[2] - ego_v[2]) ** 2 + (frame[3] - ego_v[3]) ** 2), target_v[0: i]
    else:
        return 99999, []


def find_neighbors(ego_v, vehicles):
    frame_id = ego_v[-1][0]

    PrecedingID = ego_v[-1][16]
    leftPrecedingID = ego_v[-1][18]
    leftAlongsideID = ego_v[-1][19]
    leftFollowingID = ego_v[-1][20]
    rightPrecedingID = ego_v[-1][21]
    rightAlongsideID = ego_v[-1][22]
    rightFollowingID = ego_v[-1][23]

    # print(ego_v[-1])
    PrecedingD, Preceding_v = distance(PrecedingID, ego_v[-1], frame_id, vehicles)
    # print(len(Preceding_v))
    # print(lll)
    leftPrecedingD, leftPreceding_v = distance(leftPrecedingID, ego_v[-1], frame_id, vehicles)
    rightPrecedingD, rightPreceding_v = distance(rightPrecedingID, ego_v[-1], frame_id, vehicles)
    leftFollowingD, leftFollowing_v = distance(leftFollowingID, ego_v[-1], frame_id, vehicles)
    rightFollowingD, rightFollowing_v = distance(rightFollowingID, ego_v[-1], frame_id, vehicles)
    leftAlongsideD, leftAlongside_v = distance(leftAlongsideID, ego_v[-1], frame_id, vehicles)
    rightAlongsideD, rightAlongside_v = distance(rightAlongsideID, ego_v[-1], frame_id, vehicles)

    if PrecedingD < 99999:
        Preceding = [PrecedingD, Preceding_v[-1][4], Preceding_v[-1][5], Preceding_v[-1][6], Preceding_v[-1][7],
                     Preceding_v[-1][8],
                     Preceding_v[-1][9]]
    else:
        Preceding = [0, 0, 0, 0, 0, 0, 0]

    if leftPrecedingD < 99999:
        leftPreceding = [leftPrecedingD, leftPreceding_v[-1][4], leftPreceding_v[-1][5], leftPreceding_v[-1][6],
                         leftPreceding_v[-1][7],
                         leftPreceding_v[-1][8], leftPreceding_v[-1][9]]
    else:
        leftPreceding = [0, 0, 0, 0, 0, 0, 0]

    if rightPrecedingD < 99999:
        rightPreceding = [rightPrecedingD, rightPreceding_v[-1][4], rightPreceding_v[-1][5], rightPreceding_v[-1][6],
                          rightPreceding_v[-1][7],
                          rightPreceding_v[-1][8], rightPreceding_v[-1][9]]
    else:
        rightPreceding = [0, 0, 0, 0, 0, 0, 0]

    if leftFollowingD < 99999:
        leftFollowing = [leftFollowingD, leftFollowing_v[-1][4], leftFollowing_v[-1][5], leftFollowing_v[-1][6],
                         leftFollowing_v[-1][7],
                         leftFollowing_v[-1][8], leftFollowing_v[-1][9]]
    else:
        leftFollowing = [0, 0, 0, 0, 0, 0, 0]

    if rightFollowingD < 99999:
        rightFollowing = [rightFollowingD, rightFollowing_v[-1][4], rightFollowing_v[-1][5], rightFollowing_v[-1][6],
                          rightFollowing_v[-1][7],
                          rightFollowing_v[-1][8], rightFollowing_v[-1][9]]
    else:
        rightFollowing = [0, 0, 0, 0, 0, 0, 0]

    if leftAlongsideD < 99999:
        leftAlongside = [leftAlongsideD, leftAlongside_v[-1][4], leftAlongside_v[-1][5], leftAlongside_v[-1][6],
                         leftAlongside_v[-1][7],
                         leftAlongside_v[-1][8], leftAlongside_v[-1][9]]
    else:
        leftAlongside = [0, 0, 0, 0, 0, 0, 0]

    if rightAlongsideD < 99999:
        rightAlongside = [rightAlongsideD, rightAlongside_v[-1][4], rightAlongside_v[-1][5], rightAlongside_v[-1][6],
                          rightAlongside_v[-1][7],
                          rightAlongside_v[-1][8], rightAlongside_v[-1][9]]
    else:
        rightAlongside = [0, 0, 0, 0, 0, 0, 0]

    host_v = [ego_v[-1][2], ego_v[-1][3], ego_v[-1][6], ego_v[-1][7], ego_v[-1][8], ego_v[-1][9], ego_v[-1][12],
              ego_v[-1][13], ego_v[-1][4], ego_v[-1][5]]

    Preceding_DOP = get_DOP(Preceding_v)
    leftPreceding_DOP = get_DOP(leftPreceding_v)
    rightPreceding_DOP = get_DOP(rightPreceding_v)
    leftFollowing_DOP = get_DOP(leftFollowing_v)
    rightFollowing_DOP = get_DOP(rightFollowing_v)
    leftAlongside_DOP = get_DOP(leftAlongside_v)
    rightAlongside_DOP = get_DOP(rightAlongside_v)
    host_DOP = get_DOP(ego_v)

    return (
    host_v, Preceding, leftPreceding, rightPreceding, leftFollowing, rightFollowing, leftAlongside, rightAlongside,
    host_DOP,
    Preceding_DOP, leftPreceding_DOP, rightPreceding_DOP, leftFollowing_DOP, rightFollowing_DOP, leftAlongside_DOP,
    rightAlongside_DOP)


# mean, minimum, maximum, median, 25% percentile, 75% percentile, standard deviation
# position, speed, acceleration, space headway, time headway
def get_mean(l):
    return np.mean(np.array(l))


def get_min(l):
    return np.min(np.array(l))


def get_max(l):
    return np.max(np.array(l))


def get_median(l):
    return np.median(np.array(l))


def get_25per(l):
    return np.percentile(np.array(l), 25)


def get_75per(l):
    return np.percentile(np.array(l), 75)


def get_std(l):
    return np.std(np.array(l))


def get_DOP(vehicles):
    if len(vehicles) == 0:
        return np.zeros((7, 8))

    start_x = vehicles[0][2]
    start_y = vehicles[0][3]
    traj = [[]]
    for v in vehicles:
        x = v[2] - start_x
        y = v[3] - start_y
        long_vel = v[6]
        lat_vel = v[7]
        long_acc = v[8]
        lat_acc = v[9]
        space_hw = v[12]
        time_hw = v[13]
        # position vel acc space_hw time_hw
        traj.append([x, y, long_vel, lat_vel, long_acc, lat_acc, space_hw, time_hw])

    del traj[0]
    traj = np.array(traj)
    mean = []
    min_l = []
    max_l = []
    median = []
    per25 = []
    per75 = []
    std = []
    for bf in range(8):
        mean.append(get_mean(traj[:, bf]))
    for bf in range(8):
        min_l.append(get_min(traj[:, bf]))
    for bf in range(8):
        max_l.append(get_max(traj[:, bf]))
    for bf in range(8):
        median.append(get_median(traj[:, bf]))
    for bf in range(8):
        per25.append(get_25per(traj[:, bf]))
    for bf in range(8):
        per75.append(get_75per(traj[:, bf]))
    for bf in range(8):
        std.append(get_std(traj[:, bf]))

    DOP = np.array([mean, min_l, max_l, median, per25, per75, std])

    return DOP


lc_neighbors = []
i = 0
for this_lc_cases in lc_cases:
    this_lc_neighbors = []
    for v in this_lc_cases:
        ego_v = v[0:50]
        # print(ego_v)
        neighbors = find_neighbors(ego_v, vehicles[i])
        this_lc_neighbors.append(neighbors)
    lc_neighbors.append(this_lc_neighbors)
    print(len(this_lc_neighbors))
    i += 1

print(len(lc_neighbors))
print(lc_neighbors[0][20])
# print(lc_neighbors[0][70])


lk_neighbors=[]
i = 0
for this_lk_cases in lk_cases:
    this_lk_neighbors = []
    for v in this_lk_cases:
    # if i in lk_cases_id:
        # print(len(v))
        ego_v = v
        # print(ego_v)
        neighbors = find_neighbors(ego_v, vehicles[i])
        this_lk_neighbors.append(neighbors)
    lk_neighbors.append(this_lk_neighbors)
    print(len(this_lk_neighbors))
    i += 1

print(len(lk_neighbors))

s = 0
for tmp in lc_neighbors:
    s += len(tmp)
print(s)

s = 0
for tmp in lk_neighbors:
    s += len(tmp)
print(s)


neighbors = lc_neighbors[0]
labels = lc_labels[0]
for i in range(1, len(tracks)):
    neighbors = neighbors + lc_neighbors[i]
    labels = labels + lc_labels[i]
print(len(neighbors))
for i in range(len(tracks)):
    neighbors = neighbors+ lk_neighbors[i]
    labels = labels + lk_labels[i]
print(len(neighbors))
print(len(labels))

data = []
DOPs = []
host_DOPs = []
size = []

# (host_v, Preceding, leftPreceding, rightPreceding, leftFollowing, rightFollowing, leftAlongside, rightAlongside, host_DOP,
# Preceding_DOP, leftPreceding_DOP, rightPreceding_DOP, leftFollowing_DOP, rightFollowing_DOP, leftAlongside_DOP, rightAlongside_DOP)

# rightAlongside = [distance, width, height, longitudinal v, lateral v, longitudinal acc, lateral acc]
#                      0       1       2          3              4            5                6

# host_v = [x_pos, y_pos, longitudinal v, lateral v, longitudinal acc, lateral acc, space hw, time hw, width, height]
#            0      1         2              3              4              5            6        7       8      9

for v in neighbors:
    host_v = v[0]
    p_v = v[1]
    left_p_v = v[2]
    right_p_v = v[3]
    left_r_v = v[4]
    right_r_v = v[5]
    left_as_v = v[6]
    right_as_v = v[7]

    host_v_w = host_v[8]
    host_v_h = host_v[9]

    p_v_w = p_v[1]
    p_v_h = p_v[2]

    left_p_v_w = left_p_v[1]
    left_p_v_h = left_p_v[2]

    right_p_v_w = right_p_v[1]
    right_p_v_h = right_p_v[2]

    left_r_v_w = left_r_v[1]
    left_r_v_h = left_r_v[2]

    right_r_v_w = right_r_v[1]
    right_r_v_h = right_r_v[2]

    left_as_v_w = left_as_v[1]
    left_as_v_h = left_as_v[2]

    right_as_v_w = right_as_v[1]
    right_as_v_h = right_as_v[2]

    host_DOP = v[8]
    min_front_DOP = v[9]
    min_left_front_DOP = v[10]
    min_right_front_DOP = v[11]
    min_left_back_DOP = v[12]
    min_right_back_DOP = v[13]
    min_left_as_DOP = v[14]
    min_right_as_DOP = v[15]

    v_E = np.sqrt(host_v[2] ** 2 + host_v[3] ** 2)
    a_E = np.sqrt(host_v[4] ** 2 + host_v[5] ** 2)
    v_P = np.sqrt(p_v[3] ** 2 + p_v[4] ** 2)
    a_P = np.sqrt(p_v[5] ** 2 + p_v[6] ** 2)

    v_LP = np.sqrt(left_p_v[3] ** 2 + left_p_v[4] ** 2)
    a_LP = np.sqrt(left_p_v[5] ** 2 + left_p_v[6] ** 2)
    v_LR = np.sqrt(left_r_v[3] ** 2 + left_r_v[4] ** 2)
    a_LR = np.sqrt(left_r_v[5] ** 2 + left_r_v[6] ** 2)
    G_LP_G_P = left_p_v[0] - p_v[0]
    G_LR = left_r_v[0]
    v_E_v_LR = v_E - v_LR

    v_RP = np.sqrt(right_p_v[3] ** 2 + right_p_v[4] ** 2)
    a_RP = np.sqrt(right_p_v[5] ** 2 + right_p_v[6] ** 2)
    v_RR = np.sqrt(right_r_v[3] ** 2 + right_r_v[4] ** 2)
    a_RR = np.sqrt(right_r_v[5] ** 2 + right_r_v[6] ** 2)
    G_RP_G_P = right_p_v[0] - p_v[0]
    G_RR = right_r_v[0]
    v_E_v_RR = v_E - v_RR

    G_P_t_h = p_v[0] - v_E * host_v[7]

    if v_E - v_P > v_LP - v_P:
        v_Lbenefit = v_LP - v_P
    else:
        v_Lbenefit = v_E - v_P

    if v_E - v_P > v_RP - v_P:
        v_Rbenefit = v_RP - v_P
    else:
        v_Rbenefit = v_E - v_P
        # l_this_data.append([v_E - v_P, v_LP - v_P, G_LP_G_P, G_LR, v_E_v_LR, G_P_t_h]) # 0 1 3 4 5 9
        # r_this_data.append([v_E - v_P, v_RP - v_P, G_RP_G_P, G_RR, v_E_v_RR, G_P_t_h]) # 0 2 6 7 8 9
        #                    0            1           2          3       4       5         6       7        8        9
    this_size = [[host_v_w, host_v_h], [p_v_w, p_v_h], [left_p_v_w, left_p_v_h], [right_p_v_w, right_p_v_h],
                 [left_r_v_w, left_r_v_h], [right_r_v_w, right_r_v_h], [left_as_v_w, left_as_v_h],
                 [right_as_v_w, right_as_v_h]]
    this_data = [v_E - v_P, v_LP - v_P, v_RP - v_P, G_LP_G_P, G_LR, v_E_v_LR, G_RP_G_P, G_RR, v_E_v_RR, G_P_t_h]
    this_DOP = [min_front_DOP, min_left_front_DOP, min_right_front_DOP, min_left_back_DOP, min_right_back_DOP,
                min_left_as_DOP, min_right_as_DOP]
    this_host_DOP = host_DOP
    # data.append([v_Lbenefit, v_Rbenefit, G_LP_G_P, G_LR, v_E_v_LR, G_RP_G_P, G_RR, v_E_v_RR, G_P_t_h])
    # l_data.append(l_this_data)
    # r_data.append(r_this_data)
    data.append(this_data)
    DOPs.append(this_DOP)
    host_DOPs.append(this_host_DOP)
    size.append(this_size)
"""


# data = np.array(data)
# DOPs = np.array(DOPs)
# host_DOPs = np.array(host_DOPs)
import pickle
f = open("data1_58.pkl", "rb")
data = pickle.load(f)
# pickle.dump(np.array(data), f)
f.close()
print(np.array(data).shape)

f = open("DOPs1_58.pkl", "rb")
# pickle.dump(np.array(DOPs), f)
DOPs = pickle.load(f)
f.close()
print(np.array(DOPs).shape)

f = open("host_DOPs1_58.pkl", "rb")
# pickle.dump(np.array(host_DOPs), f)
host_DOPs = pickle.load(f)
f.close()
print(np.array(host_DOPs).shape)

f = open("labels1_58.pkl", "rb")
# pickle.dump(np.array(labels), f)
labels = pickle.load(f)
f.close()
print(np.array(labels).shape)

f = open("size1_58.pkl", "rb")
# pickle.dump(np.array(size), f)
size = pickle.load(f)
f.close()
print(np.array(size).shape)

# data = sklearn.preprocessing.normalize(data)
# print(data[0:5])


# test = random.sample(range(82343),4343)
test = np.loadtxt("test_cases.txt")


print(len(labels))
print(len(data))
print(len(test))

train_data = []
test_data = []
train_DOPs = []
test_DOPs = []
train_host_DOPs = []
test_host_DOPs = []
train_labels = []
test_labels = []

for i in range(len(data)):
    if i in test:
        if labels[i] == 0:
            test_data.append(data[i])
            test_DOPs.append(DOPs[i])
            test_host_DOPs.append(host_DOPs[i])
            # test_DOPs.append(np.zeros((7,7,8)))
            # test_host_DOPs.append(np.zeros((7,8)))
            test_labels.append(labels[i])
        else:
            # for _ in range(16):
            test_data.append(data[i])
            test_DOPs.append(DOPs[i])
            test_host_DOPs.append(host_DOPs[i])
            # test_DOPs.append(np.zeros((7,7,8)))
            # test_host_DOPs.append(np.zeros((7,8)))
            test_labels.append(labels[i])
    else:
        if labels[i] == 0:
            train_data.append(data[i])
            train_DOPs.append(DOPs[i])
            train_host_DOPs.append(host_DOPs[i])
            # train_DOPs.append(np.zeros((7,7,8)))
            # train_host_DOPs.append(np.zeros((7,8)))
            train_labels.append(labels[i])
        else:
            # for _ in range(16):
            train_data.append(data[i])
            train_DOPs.append(DOPs[i])
            train_host_DOPs.append(host_DOPs[i])
            # train_DOPs.append(np.zeros((7,7,8)))
            # train_host_DOPs.append(np.zeros((7,8)))
            train_labels.append(labels[i])

train_data = np.array(train_data).reshape(len(train_data), 10).astype(float)
train_DOPs = np.array(train_DOPs).reshape(len(train_data), 7, 7, 8).astype(float)
train_host_DOPs = np.array(train_host_DOPs).reshape(len(train_data), 7, 8).astype(float)
train_labels = np.array(train_labels)

test_data = np.array(test_data).reshape(len(test_data), 10).astype(float)
test_DOPs = np.array(test_DOPs).reshape(len(test_data), 7, 7, 8).astype(float)
test_host_DOPs = np.array(test_host_DOPs).reshape(len(test_data), 7, 8).astype(float)
test_labels = np.array(test_labels)

print("train data size:")
print(np.array(train_data).shape)
print("test data size:")
print(np.array(test_data).shape)
print("train DOP size:")
print(np.array(train_DOPs).shape)
print("test DOP size:")
print(np.array(test_DOPs).shape)
print("train host DOP size:")
print(np.array(train_host_DOPs).shape)
print("test host DOP size:")
print(np.array(test_host_DOPs).shape)
print(np.sum(test_labels == 1))
print(np.sum(test_labels == 2))
print(np.sum(test_labels == 0))


import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_source, data_DOPs, host_DOPs, labels, transform=None):
        self.data = data_source
        self.DOPs = data_DOPs
        self.host_DOPs = host_DOPs
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        if self.transform is not None:
            img = self.transform(torch.from_numpy(self.data[index]).long())
        else:
            img = self.data[index]
            DOP = self.DOPs[index]
            host_DOP = self.host_DOPs[index]
        return img, DOP, host_DOP, self.labels[index]

    def __len__(self):
        return len(self.data)


# Hyper Parameters
EPOCH = 50            # train the training data n times
BATCH_SIZE = 16
LR = 0.001               # learning rate

train_dataset = Dataset(train_data, train_DOPs, train_host_DOPs, train_labels)
test_dataset = Dataset(test_data, test_DOPs, test_host_DOPs, test_labels)

print(len(train_dataset.data))
print(len(train_dataset.labels))

# Data Loader for easy mini-batch return in training
train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# convert test data into Variable
print(test_dataset.data.shape)
test_x = Variable(torch.from_numpy(test_dataset.data))
test_x_DOPs = Variable(torch.from_numpy(test_dataset.DOPs))
test_x_host_DOPs = Variable(torch.from_numpy(test_dataset.host_DOPs))
print(test_x.shape)
test_y = test_dataset.labels


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden_1, n_hidden_2, n_hidden_3, n_output):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=7,
                out_channels=16,
                kernel_size=4,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.host_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=4,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.host_conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=8,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.hidden_1 = torch.nn.Linear(n_feature + 32 + 8, n_hidden_1)
        self.hidden_2 = torch.nn.Linear(n_hidden_1, n_hidden_2)
        self.hidden_3 = torch.nn.Linear(n_hidden_2, n_hidden_3)
        self.out = torch.nn.Linear(n_hidden_3, n_output)

    def forward(self, x, DOP, host_DOP):
        x = x.type(torch.FloatTensor)
        DOP = DOP.type(torch.FloatTensor)
        host_DOP = host_DOP.type(torch.FloatTensor).unsqueeze(1)
        DOP = self.conv1(DOP)
        DOP = self.conv2(DOP)
        DOP = DOP.view(DOP.size(0), -1)
        host_DOP = self.host_conv1(host_DOP)
        host_DOP = self.host_conv2(host_DOP)
        host_DOP = host_DOP.view(host_DOP.size(0), -1)
        x = torch.cat((x, DOP, host_DOP), dim=1)
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = F.relu(self.hidden_3(x))
        x = self.out(x)
        x = torch.sigmoid(x)
        return x


net = Net(n_feature=10, n_hidden_1=128, n_hidden_2=32, n_hidden_3=16, n_output=3)

print(net)

net.load_state_dict(torch.load("all_tracks_0927_DLC_1frame_3output_neighbor_2cnn_test_acc99_no_bal_test_data.pkl"))
exit()

optimizer = torch.optim.Adam(net.parameters(), lr=LR)  # optimize all parameters
loss_func = nn.CrossEntropyLoss()
acc = []

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_DOP, b_host_DOP, b_y) in enumerate(train_data_loader):  # gives batch data
        # print(b_x.shape)
        output = net(b_x, b_DOP, b_host_DOP)  # output
        # print(output)
        # print(b_y)
        b_y = b_y.long()
        loss = loss_func(output, b_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if step % 1 == 0:
            test_output = net(test_x, test_x_DOPs, test_x_host_DOPs)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            acc.append(accuracy)

            if accuracy >= 0.90:
                torch.save(net.state_dict(), "all_tracks_0927_DLC_1frame_3output_neighbor_2cnn_test_acc" + str(
                    round(accuracy * 100)) + "_no_bal_test_data.pkl")

        if step % 300 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test_accuracy: %.2f' % accuracy)

print(np.max(np.array(acc)))