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
    

class RandomBoxOverlay():
    def __init__(self, probability=0.5, min_box_size=5, max_box_size=28):
        """
        Initializes the transform with the given parameters.
        
        :param probability: The probability of applying the black box.
        :param min_box_size: The minimum size of the black box.
        :param max_box_size: The maximum size of the black box.
        """
        self.probability = probability
        self.min_box_size = min_box_size
        self.max_box_size = max_box_size

    def __call__(self, x):
        """
        Apply the random black box to the image tensor.
        
        :param x: A tensor representing an image, with shape (C, H, W).
        :return: A tensor of the same shape with a random black box overlay.
        """
        if torch.rand(1) < self.probability:
            C, H, W = x.shape
            box_width = torch.randint(self.min_box_size, self.max_box_size, (1,))
            box_height = torch.randint(self.min_box_size, self.max_box_size, (1,))

            # Ensure the box fits within the image dimensions
            box_width = torch.min(box_width[0], torch.tensor(W))
            box_height = torch.min(box_height[0], torch.tensor(H))


            x1 = torch.randint(0, W - box_width, (1,))
            y1 = torch.randint(0, H - box_height, (1,))

            x[:, y1:y1+box_height, x1:x1+box_width] = 0
        return x