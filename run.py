import logging
from typing import Iterator
from omegaconf import DictConfig, OmegaConf
import hydra
import os
import sys
import errno
from time import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.utils import summary
from common.dataset_generators import UnchunkedGeneratorDataset, ChunkedGeneratorDataset
from trainval import create_model, load_dataset, fetch, load_weight, train, eval

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

    # Loading weight
    model_pos_train, model_pos, checkpoint = load_weight(
        cfg, model_pos_train, model_pos)

    test_dataset = UnchunkedGeneratorDataset(cameras_valid, poses_valid, poses_valid_2d,
                                             pad=pad, causal_shift=causal_shift, augment=False,
                                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    test_loader = DataLoader(test_dataset, 1, shuffle=False)
    log.info("Testing on {} frames".format(test_dataset.num_frames()))

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

        train_dataset = ChunkedGeneratorDataset(cameras_train, poses_train, poses_train_2d,
                                                cfg.stride,
                                                pad=pad, causal_shift=causal_shift, shuffle=True, augment=cfg.data_augmentation,
                                                kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                                joints_right=joints_right)
        train_loader = DataLoader(
            train_dataset, cfg.batch_size, shuffle=True, num_workers=4)

        train_dataset_eval = UnchunkedGeneratorDataset(cameras_train, poses_train, poses_train_2d,
                                                       pad=pad, causal_shift=causal_shift, augment=False)
        train_loader_eval = DataLoader(train_dataset_eval, 1, shuffle=False)

        log.info('Training on {} frames'.format(
            train_dataset_eval.num_frames()))
        _, _, sample_inputs_2d = train_dataset[0]
        input_shape = [cfg.batch_size]
        input_shape += list(sample_inputs_2d.shape)
        log.info('Input 2d shape: {}'.format(
            input_shape))
        summary(log, model_pos,
                sample_inputs_2d.shape, cfg.batch_size, device='cpu')

        if cfg.resume:
            epoch = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                log.info(
                    'WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')

            lr = checkpoint['lr']

        log.info(
            '** Note: reported losses are averaged over all frames and test-time augmentation is not used here.')
        log.info(
            '** The final evaluation will be carried out after the last training epoch.')

        loss_min = 49.5

        # TODO
        if torch.cuda.is_available():
            model_pos = model_pos.cuda()
            model_pos_train = model_pos_train.cuda()

        # Pos model only
        while epoch < cfg.epochs:
            start_time = time()
            epoch_loss_3d_train = 0
            model_pos_train.train()

            # Regular supervised scenario
            epoch_loss_3d = train(model_pos_train, train_loader, optimizer)
            losses_3d_train.append(epoch_loss_3d)

            # After training an epoch, whether to evaluate the loss of the training and validation set
            if not cfg.no_eval:
                model_train_dict = model_pos_train.state_dict()
                losses_3d_valid_ave, losses_3d_train_eval_ave = eval(
                    model_train_dict, model_pos, test_loader, train_loader_eval)
                losses_3d_valid.append(losses_3d_valid_ave)
                losses_3d_train_eval.append(losses_3d_train_eval_ave)

            elapsed = (time() - start_time) / 60

            if cfg.no_eval:
                log.info('[%d] time %.2f lr %f 3d_train %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses_3d_train[-1] * 1000))
            else:
                log.info('[%d] time %.2f lr %f 3d_train %f 3d_eval %f 3d_valid %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses_3d_train[-1] * 1000,
                    losses_3d_train_eval[-1] * 1000,
                    losses_3d_valid[-1] * 1000))

                # Saving the best result
                if losses_3d_valid[-1]*1000 < loss_min:
                    chk_path = os.path.join(cfg.checkpoint, 'epoch_best.bin')
                    log.info('Saving checkpoint to', chk_path)

                    torch.save({
                        'epoch': epoch,
                        'lr': lr,
                        'optimizer': optimizer.state_dict(),
                        'model_pos': model_pos_train.state_dict()
                    }, chk_path)

                    loss_min = losses_3d_valid[-1]*1000

            # Decay learning rate exponentially
            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay
            epoch += 1

            # Save checkpoint if necessary
            if epoch % cfg.checkpoint_frequency == 0:
                chk_path = os.path.join(
                    cfg.checkpoint, 'epoch_{}.bin'.format(epoch))
                log.info('Saving checkpoint to', chk_path)

                torch.save({
                    'epoch': epoch,
                    'lr': lr,
                    'optimizer': optimizer.state_dict(),
                    'model_pos': model_pos_train.state_dict()
                }, chk_path)

            # Save training curves after every epoch, as .png images (if requested)
            if cfg.export_training_curves and epoch > 3:
                if 'matplotlib' not in sys.modules:
                    import matplotlib

                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt

                plt.figure()
                epoch_x = np.arange(3, len(losses_3d_train)) + 1
                plt.plot(epoch_x, losses_3d_train[3:], '--', color='C0')
                plt.plot(epoch_x, losses_3d_train_eval[3:], color='C0')
                plt.plot(epoch_x, losses_3d_valid[3:], color='C1')
                plt.legend(['3d train', '3d train (eval)', '3d valid (eval)'])
                plt.ylabel('MPJPE (m)')
                plt.xlabel('Epoch')
                plt.xlim((3, epoch))
                plt.savefig(os.path.join(cfg.checkpoint, 'loss_3d.png'))
                plt.close('all')


if __name__ == '__main__':
    main()
