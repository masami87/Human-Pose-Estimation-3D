import logging
from typing import Iterator
from omegaconf import DictConfig, OmegaConf
import hydra
import os
import errno
from time import time

import torch

from common.utils import summary
from common.generators import UnchunkedGenerator, ChunkedGenerator
from trainval import create_model, load_dataset, fetch, load_weight, train

log = logging.getLogger('hpe-3d')


@hydra.main(config_path="config/", config_name="conf")
def main(cfg: DictConfig):
    log.info('Config:\n' + OmegaConf.to_yaml(cfg))

    try:
        # Create checkpoint directory if it does not exist
        os.makedirs(cfg.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError(
                'Unable to create checkpoint directory:', cfg.checkpoint)

    dataset, keypoints, keypoints_metadata, kps_left, kps_right, joints_left, joints_right = load_dataset(cfg.data_dir,
                                                                                                          cfg.dataset, cfg.keypoints)

    subjects_train = cfg.subjects_train.split(',')
    subjects_test = cfg.subjects_test.split(',')

    action_filter = None if cfg.actions == '*' else cfg.actions.split(',')
    if action_filter is not None:
        log.info('Selected actions:', action_filter)

    cameras_valid, poses_valid, poses_valid_2d = fetch(
        subjects_test, dataset, keypoints, action_filter, cfg.downsample, cfg.subset)

    model_pos_train, model_pos, pad, causal_shift = create_model(
        cfg, dataset, poses_valid_2d)
    receptive_field = model_pos.receptive_field()
    log.info(" Receptive field: {} frames".format(receptive_field))
    if cfg.causal:
        log.info("Using causal convolutions")

    # TODO
    # if torch.cuda.is_available():
    #     model_pos = model_pos.cuda()
    #     model_pos_train = model_pos_train.cuda()

    # Loading weight
    model_pos_train, model_pos, checkpoint = load_weight(
        cfg, model_pos_train, model_pos)

    test_generator = UnchunkedGenerator(cameras_valid, poses_valid, poses_valid_2d,
                                        pad=pad, causal_shift=causal_shift, augment=False,
                                        kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    log.info("Testing on {} frames".format(test_generator.num_frames()))

    if not cfg.evaluate:
        cameras_train, poses_train, poses_train_2d = fetch(subjects_train,  dataset, keypoints, action_filter,
                                                           cfg.downsample, subset=cfg.subset)
        lr = cfg.learning_rate
        optimizer = torch.optim.Adam(
            model_pos_train.parameters(), lr=lr, amsgrad=True)
        lr_decay = cfg.lr_decay

        losses_3d_train = []
        losses_3d_train_eval = []
        losses_3d_valid = []

        epoch = 0
        initial_momentum = 0.1
        final_momentum = 0.01

        train_generator = ChunkedGenerator(cfg.batch_size // cfg.stride, cameras_train, poses_train, poses_train_2d,
                                           cfg.stride,
                                           pad=pad, causal_shift=causal_shift, shuffle=True, augment=cfg.data_augmentation,
                                           kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                           joints_right=joints_right)
        train_generator_eval = UnchunkedGenerator(cameras_train, poses_train, poses_train_2d,
                                                  pad=pad, causal_shift=causal_shift, augment=False)
        log.info('Training on {} frames'.format(
            train_generator_eval.num_frames()))
        _, _, sample_inputs_2d = next(
            train_generator.next_epoch())
        log.info('Input 2d shape: {}'.format(list(sample_inputs_2d.shape)))
        summary(log, model_pos,
                sample_inputs_2d.shape[1:], sample_inputs_2d.shape[0], device='cpu')

        if cfg.resume:
            epoch = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
                train_generator.set_random_state(checkpoint['random_state'])
            else:
                log.info(
                    'WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')

            lr = checkpoint['lr']

        log.info(
            '** Note: reported losses are averaged over all frames and test-time augmentation is not used here.')
        log.info(
            '** The final evaluation will be carried out after the last training epoch.')

        loss_min = 49.5
        # Pos model only
        while epoch < cfg.epochs:
            start_time = time()
            epoch_loss_3d_train = 0
            model_pos_train.train()

            # Regular supervised scenario
            epoch_loss_3d = train(model_pos_train, train_generator, optimizer)
            losses_3d_train.append(epoch_loss_3d)


if __name__ == '__main__':
    main()
