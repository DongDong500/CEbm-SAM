import torch
import torch.nn as nn

class ClassConditionalInstanceNorm2d(nn.Module):
    def __init__(
            self, 
            num_features, 
            num_classes, 
            affine: bool = False
    ):
        """
        Class conditional instance normalization layer

        Args:
            num_features (int): C from an expected input size (BxCxHxW)
            num_classes (int): 
            affine (bool):  a boolean value that when set to True, 
                `nn.InstanceNorm2d` module has learnable affine parameters, 
                initialized the same way as done for batch normalization. 
                (Default: False)
        """
        super().__init__()
        self.num_features = num_features
        self.isn = nn.InstanceNorm2d(num_features, affine=affine)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        
        self.mlp = nn.Sequential(
            nn.Linear(num_features * 2, num_features * 2),
            nn.ReLU(inplace=True),
            nn.Linear(num_features * 2, num_features * 2)
        )

        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x, y):
        
        out = self.isn(x)
        
        gamma, beta = self.mlp(self.embed(y)).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
    
        return out

class CEmb(nn.Module):

    def __init__(
            self, 
            num_features, 
            emb_classes, 
            CIN_affine=True, 
            mid_channels=None
    ):
        """
        Condition Embedding block

        Arguments:
            num_feature (int): 
            emb_classes (int): 
        """
        super().__init__()

        if mid_channels is None:
            mid_channels = num_features

        self.conv1 = nn.Conv2d(num_features, mid_channels, kernel_size=3, padding=1, bias=False)
        self.cin1 = ClassConditionalInstanceNorm2d(num_features, emb_classes, CIN_affine)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, num_features, kernel_size=3, padding=1, bias=False)
        self.cin2 = ClassConditionalInstanceNorm2d(num_features, emb_classes, CIN_affine)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, y):
        
        x = self.conv1(x)
        x = self.cin1(x, y)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.cin2(x, y)
        out = self.relu2(x)

        return out



if __name__ == "__main__":

    m = CEmb(num_features=2, emb_classes=2, CIN_affine=False)

    print(
        m(
            torch.rand(size=(5, 2, 12, 15)), 
            torch.randint(low=0, high=2, size=(5, ))
        ).shape
    )