import torch


def get_pose3dbyBoneVec(bones, num_joints=17):
    '''
    convert bone vect to pose3d， inverse function of get_bone_vector
    :param bones:
    :return:
    '''
    Ctinverse = torch.Tensor([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 basement
        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 1
        [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1 2
        [-1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2 3
        [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 4
        [0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4 5
        [0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5 6
        [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 7
        [0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],  # 7 8
        [0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0],  # 8 9
        [0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0],  # 9 10
        [0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, 0, 0, 0, 0, 0],  # 8 11
        [0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 0, 0],  # 11 12
        [0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, -1, 0, 0, 0],  # 12 13
        [0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, -1, 0, 0],  # 8 14
        [0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, -1, -1, 0],  # 14 15
        [0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1],  # 15 16
    ]).transpose(1, 0)

    Ctinverse = Ctinverse.to(bones.device)
    C = Ctinverse.repeat([bones.size(0), bones.size(1), 1, 1])
    bonesT = bones.permute(0, 1, 3, 2).contiguous()
    pose3d = torch.matmul(bonesT, C)
    pose3d = pose3d.permute(0, 1, 3, 2).contiguous()  # back to N x f x 17 x 3
    return pose3d


def get_BoneVecbypose3d(x, num_joints=17):
    '''
    convert 3D point to bone vector
    :param x: N x frames x number of joint x 3
    :return: N x frames x number of bone x 3  number of bone = number of joint - 1
    '''
    Ct = torch.Tensor([
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 1
        [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1 2
        [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2 3
        [1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 4
        [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4 5
        [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5 6
        [1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 7
        [0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],  # 7 8
        [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0],  # 8 9
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],  # 9 10
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0],  # 8 11
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],  # 11 12
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0],  # 12 13
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0],  # 8 14
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],  # 14 15
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1],  # 15 16
    ]).transpose(1, 0)  # 17 * 16

    Ct = Ct.to(x.device)
    C = Ct.repeat([x.size(0), x.size(1), 1, 1])  # N * f * J * 3
    pose3 = x.permute(0, 1, 3, 2).contiguous()  # 这里17x3变成3x17的话 应该用permute吧
    B = torch.matmul(pose3, C)
    B = B.permute(0, 1, 3, 2)  # back to N x f x 16 x 3
    return B


def get_BoneVecbypose2d(x, num_joints=17):
    '''
    convert 2D point to bone vector
    :param x: N x number of joint x 2
    :return: N x number of bone x 2  number of bone = number of joint - 1
    '''
    Ct = torch.Tensor([
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 1
        [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1 2
        [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2 3
        [1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 4
        [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4 5
        [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5 6
        [1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 7
        [0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],  # 7 8
        [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0],  # 8 9
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],  # 9 10
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0],  # 8 11
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],  # 11 12
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0],  # 12 13
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0],  # 8 14
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],  # 14 15
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1],  # 15 16
    ]).transpose(1, 0)

    Ct = Ct.to(x.device)
    C = Ct.repeat([x.size(0), x.size(1), 1, 1])  # N * f * J * 3
    pose2 = x.permute(0, 1, 3, 2).contiguous()  # 这里17x2变成2x17的话 应该用permute吧
    B = torch.matmul(pose2, C)
    B = B.permute(0, 1, 3, 2)  # back to N x f x 16 x 2
    return B


def get_bone_lengthbypose3d(x, bone_dim=2):
    '''
    :param bone_dim: dim=2
    :return:
    '''
    bonevec = get_BoneVecbypose3d(x)
    bones_length = torch.norm(bonevec, dim=2, keepdim=True)
    return bones_length


def get_bone_unit_vecbypose3d(x, num_joints=16, bone_dim=2):
    bonevec = get_BoneVecbypose3d(x)
    bonelength = get_bone_lengthbypose3d(x)
    bone_unitvec = bonevec / bonelength
    return bone_unitvec
