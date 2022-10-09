import torch

#! 90
def rotation1():
    def _transform(images):
        size = images.shape[1:]
        return torch.stack([torch.rot90(images, k, (2, 3)) for k in [1]], 1).view(-1, *size)
    
    return _transform, 1

#! 0/180
def rotation2():
    def _transform(images):
        size = images.shape[1:]
        return torch.stack([torch.rot90(images, k, (2, 3)) for k in [0, 2]], 1).view(-1, *size)

    return _transform, 2

#! 0/90/270
def rotation3():
    def _transform(images):
        size = images.shape[1:]
        return torch.stack([torch.rot90(images, k, (2, 3)) for k in [0, 1, 3]], 1).view(-1, *size)

    return _transform, 3

#! 90/180/270
def rotation33():
    def _transform(images):
        size = images.shape[1:]
        return torch.stack([torch.rot90(images, k, (2, 3)) for k in [1, 2, 3]], 1).view(-1, *size)

    return _transform, 3

#! 0/90/180/270
def rotation4():
    def _transform(images):
        size = images.shape[1:]
        return torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1).view(-1, *size)

    return _transform, 4

#! GBR
def color_perm1():
    def _transform(images):
        size = images.shape[1:]
        images = torch.stack([images[:, 1, :, :], images[:, 2, :, :], images[:, 0, :, :]], 1).view(-1, *size)
        return images.contiguous()

    return _transform, 1

#! GBR,BRG
def color_perm2():
    def _transform(images):
        size = images.shape[1:]
        images = torch.stack([torch.stack([images[:, 1, :, :], images[:, 2, :, :], images[:, 0, :, :]], 1),
                              torch.stack([images[:, 2, :, :], images[:, 0, :, :], images[:, 1, :, :]], 1)], 1).view(-1, *size)
        return images.contiguous()

    return _transform, 2

#! RGB,GBR,BRG
def color_perm3():
    def _transform(images):
        size = images.shape[1:]
        images = torch.stack([images,
                              torch.stack([images[:, 1, :, :], images[:, 2, :, :], images[:, 0, :, :]], 1),
                              torch.stack([images[:, 2, :, :], images[:, 0, :, :], images[:, 1, :, :]], 1)], 1).view(-1, *size)
        return images.contiguous()

    return _transform, 3

#! RGB,RBG,GRB,GBR,BRG,BGR  
def color_perm6():
    def _transform(images):
        size = images.shape[1:]
        images = torch.stack([images,
                              torch.stack([images[:, 0, :, :], images[:, 2, :, :], images[:, 1, :, :]], 1),
                              torch.stack([images[:, 1, :, :], images[:, 0, :, :], images[:, 2, :, :]], 1),
                              torch.stack([images[:, 1, :, :], images[:, 2, :, :], images[:, 0, :, :]], 1),
                              torch.stack([images[:, 2, :, :], images[:, 0, :, :], images[:, 1, :, :]], 1),
                              torch.stack([images[:, 2, :, :], images[:, 1, :, :], images[:, 0, :, :]], 1)], 1).view(-1, *size)
        return images.contiguous()

    return _transform, 6

