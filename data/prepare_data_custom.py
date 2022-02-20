from genericpath import exists
import os
import numpy as np
from glob import glob
import json
from tqdm import tqdm
import cv2

keypoints_root = '/home/data/scut/Label_3d/'
depth_root = '/home/data/scut/Depth_frame/'

save_npz_path = '/home/wt/py_projects/Human-Pose-Estimation-3D/data/'
output_filename = save_npz_path + 'data_3d_custom'
output_filename_2d = save_npz_path + 'data_2d_custom_gt'
output_filename_dep = save_npz_path + 'data_dep_custom'


target_subjects = ['P000', 'P001', 'P002', 'P003', 'P004']
target_cameras = ['C000', 'C001', 'C002', 'C003']
target_actions = ['A%03d' % i for i in range(0, 20)]

njoints = 21


def get_seg_dir(subjects, cameras, actions):
    skeletons = {}
    num = 0
    for subject in subjects:
        skeletons[subject] = {}
        for action in actions:
            skeletons[subject][action] = {}
            for cam in cameras:
                match = cam + subject + '*' + action
                print("finding %s ..." % match)
                seg_names = glob(keypoints_root + match)

                seg_names.sort()
                skeletons[subject][action][cam] = seg_names
                num += len(seg_names)
    return skeletons, num


def read_json(file) -> dict:
    data = {}
    with open(file, 'r') as f:
        j = json.load(f)
        data['points'] = []
        data['points_3d'] = []
        for joint in j['shapes']:
            data['points'].append(joint['points'][0])
            data['points_3d'].append(joint['points_3d'][0])

        assert len(data['points']) == len(data['points_3d']) == njoints
    return data


def process_segment(seg_path):
    json_files = os.listdir(seg_path)
    json_files.sort()
    frames = len(json_files)

    joints_vec = np.zeros((frames, njoints, 3), dtype=np.float32)
    keypoints_vec = np.zeros((frames, njoints, 2), dtype=np.float32)
    depth_vec = np.zeros((frames, njoints, 1), dtype=np.float32)

    for frame, file in enumerate(json_files):
        data = read_json(os.path.join(seg_path, file))
        dep_path = os.path.basename(file).replace(
            'RF', 'DF').replace('json', 'png')
        seg_name = os.path.basename(seg_path)
        dep_img = cv2.imread(os.path.join(
            depth_root, seg_name, dep_path), cv2.IMREAD_ANYDEPTH)
        keypoints = data['points'].copy()

        positions = [(round(y), round(x)) for x, y in keypoints]

        for joint, position in enumerate(positions):
            depth = dep_img[position]
            if depth > 5500:
                print("***** warning *****: find depth {}".format(depth))
            depth_vec[frame, joint] = depth

        joints_vec[frame] = np.array(data['points_3d'], dtype=np.float32)
        keypoints_vec[frame] = np.array(data['points'], dtype=np.float32)
    return joints_vec, keypoints_vec, depth_vec, frames


def run():
    skeletons, num_seg = get_seg_dir(
        target_subjects, target_cameras, target_actions)

    print("Find {} video segs.".format(num_seg))

    output_3d = {}
    output_2d = {}
    output_dep = {}

    total_frames = 0

    for subject in skeletons.keys():
        output_3d[subject] = {}
        output_2d[subject] = {}
        output_dep[subject] = {}

        for action in skeletons[subject].keys():
            output_3d[subject][action] = {}
            output_2d[subject][action] = {}
            output_dep[subject][action] = {}

            for cam in skeletons[subject][action].keys():
                output_3d[subject][action][cam] = {}
                output_2d[subject][action][cam] = {}
                output_dep[subject][action][cam] = {}

                files = skeletons[subject][action][cam]
                # print(files)
                # exit(1)
                seg = 0

                for f in tqdm(files):
                    # basename = os.path.basename(f)
                    # name = basename.split('.')[-2]
                    joints_vec, keypoints_vec, dep_vec, frames = process_segment(
                        f)

                    total_frames += frames

                    output_3d[subject][action][cam][seg] = joints_vec.astype(
                        'float32')
                    output_2d[subject][action][cam][seg] = keypoints_vec.astype(
                        'float32')
                    output_dep[subject][action][cam][seg] = dep_vec.astype(
                        'float32')

                    seg += 1

    print("Finish reading all files.\nSaving...")
    print("Average frames: {}.".format(total_frames / num_seg))
    np.savez_compressed(output_filename, positions_3d=output_3d)
    np.savez_compressed(output_filename_2d, positions_2d=output_2d)
    np.savez_compressed(output_filename_dep, depths=output_dep)
    print("Done.")


if __name__ == '__main__':
    run()
