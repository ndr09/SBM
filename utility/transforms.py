import torch

class PermutePixels:
    def __init__(self, permutation):
        """
        Initializes the transform with a specific permutation.
        
        :param permutation: A 1D tensor or list with the indices of the permuted pixels.
                            It should have a size of H*W for a single channel, and the permutation
                            will be applied identically across channels.
        """
        self.permutation = permutation

    def __call__(self, x):
        """
        Apply the permutation to the image tensor.
        
        :param x: A tensor representing an image, with shape (C, H, W).
        :return: A tensor of the same shape with permuted pixels.
        """
        if len(x.shape) == 3:
            C, H, W = x.shape
            # Flatten spatial dimensions and apply the permutation
            x_permuted = x.view(C, H*W)[:, self.permutation]
            # Reshape back to the original dimensions
            return x_permuted.view(C, H, W)
        elif len(x.shape) == 2:
            W, H = x.shape
            # Flatten spatial dimensions and apply the permutation
            x_permuted = x.view(W*H)[:, self.permutation]
            # Reshape back to the original dimensions
            return x_permuted.view(W, H)
        
    @staticmethod
    def random_permutation(size):
        """
        Generates a random permutation of a given size.
        
        :param size: The size of the permutation.
        :return: A 1D tensor with the indices of the permutation.
        """
        return torch.randperm(size)
