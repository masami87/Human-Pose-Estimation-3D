import os
import sys
import errno
from time import time
import numpy as np
import torch
import torch.nn as nn

from common.camera import world_to_camera, normalize_screen_coordinates
from common.utils import deterministic_random
from common.loss import mpjpe
from model.VideoPose3D import TemporalModel, TemporalModelOptimized1f


def load_dataset(data_dir: str, dataset_type: str, keypoints_type: str):
    print('Loading dataset...')
    dataset_path = data_dir + 'data_3d_' + dataset_type + '.npz'

    if dataset_type == "h36m":
        from datasets.h36m import Human36mDataset
        dataset = Human36mDataset(dataset_path)
    else:
        raise KeyError('Invalid dataset')

    print('Preparing data')
    # TODO ?
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            if 'positions' in anim:
                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(
                        anim['positions'], R=cam['orientation'], t=cam['translation'])
                    # Remove global offset, but keep trajectory in first position
                    pos_3d[:, 1:] -= pos_3d[:, :1]
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d

    print('Loading 2D detections...')
    keypoints = np.load(data_dir + 'data_2d_' + dataset_type +
                        '_' + keypoints_type + '.npz', allow_pickle=True)
    keypoints_metadata = keypoints['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(
        keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(
        dataset.skeleton().joints_right())
    keypoints = keypoints['positions_2d'].item()

    for subject in dataset.subjects():
        assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(
            subject)
        for action in dataset[subject].keys():
            assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(
                action, subject)
            if 'positions_3d' not in dataset[subject][action]:
                continue

            for cam_idx in range(len(keypoints[subject][action])):

                # We check for >= instead of == because some videos in H3.6M contain extra frames
                mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    # Shorten sequence
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

            assert len(keypoints[subject][action]) == len(
                dataset[subject][action]['positions_3d'])

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(
                    kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps
    return dataset, keypoints, keypoints_metadata, kps_left, kps_right, joints_left, joints_right


def fetch(subjects, dataset, keypoints, action_filter=None, downsample=5, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    stride = downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
            start = deterministic_random(
                0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]
    return out_camera_params, out_poses_3d, out_poses_2d


def create_model(cfg, dataset, poses_valid_2d):
    filter_widths = [int(x) for x in cfg.architecture.split(",")]

    if not cfg.disable_optimizations and not cfg.dense and cfg.stride == 1:
        # Use optimized model for single-frame predictions
        model_pos_train = TemporalModelOptimized1f(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                                                   filter_widths=filter_widths, causal=cfg.causal, dropout=cfg.dropout, channels=cfg.channels)
    else:
        # When incompatible settings are detected (stride > 1, dense filters, or disabled optimization) fall back to normal model
        model_pos_train = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                                        filter_widths=filter_widths, causal=cfg.causal, dropout=cfg.dropout, channels=cfg.channels,
                                        dense=cfg.dense)

    model_pos = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                              filter_widths=filter_widths, causal=cfg.causal, dropout=cfg.dropout, channels=cfg.channels,
                              dense=cfg.dense)

    receptive_field = model_pos.receptive_field()
    pad = (receptive_field - 1) // 2  # padding on each side
    if cfg.causal:
        causal_shift = pad
    else:
        causal_shift = 0

    return model_pos_train, model_pos, pad, causal_shift


def load_weight(cfg, model_pos_train, model_pos):
    checkpoint = dict()
    if cfg.resume or cfg.evaluate:
        chk_filename = os.path.join(
            cfg.checkpoint, cfg.resume if cfg.resume else cfg.evaluate)
        print("Loading checkpoint", chk_filename)
        checkpoint = torch.load(chk_filename)
        # print("This model was trained for {} epochs".format(checkpoint["epoch"]))
        model_pos_train.load_state_dict(checkpoint["model_pos"])
        model_pos.load_state_dict(checkpoint["model_pos"])

    return model_pos_train, model_pos, checkpoint


def train(model_pos_train, train_generator, optimizer):
    epoch_loss_3d_train = 0
    N = 0

    # TODO dataloader and tqdm
    for _, batch_3d, batch_2d in train_generator.next_epoch():
        inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
        inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
        if torch.cuda.is_available():
            inputs_3d = inputs_3d.cuda()
            inputs_2d = inputs_2d.cuda()

        # TODO
        inputs_3d[:, :, 0] = 0

        optimizer.zero_grad()

        # Predict 3D poses
        predicted_3d_pos = model_pos_train(inputs_2d)
        loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)

        epoch_loss_3d_train += inputs_3d.shape[0] * \
            inputs_3d.shape[1] * loss_3d_pos.item()
        N += inputs_3d.shape[0] * inputs_3d.shape[1]

        loss_total = loss_3d_pos
        loss_total.backward()

        optimizer.step()

    epoch_losses_eva = epoch_loss_3d_train / N

    return epoch_losses_eva
