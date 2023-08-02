import torch
import torch.nn as nn

from transformers import SwinConfig, SwinModel


class Swin_t(nn.Module):
    def __init__(self, num_class, num_channels, input_size=224):
        super(Swin_t, self).__init__()
        configuration = SwinConfig(image_size=input_size, patch_size=4, num_channels=num_channels, embed_dim=96,
            depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, drop_path_rate=0.2)
        
        # Initializing a model (with random weights) from the vit-base-patch16-224 style configuration
        self.swin = SwinModel(configuration)

        self.classifier = nn.Linear(self.swin.num_features, num_class)

    def forward(self, x):
        outputs = self.swin(pixel_values=x)
        
        # see https://github.com/huggingface/transformers/blob/main/src/transformers/models/swin/modeling_swin.py#L1166
        pooled_output = outputs[1]
        output = self.classifier(pooled_output)

        return output
    

class Swin_s(nn.Module):
    def __init__(self, num_class, num_channels, input_size=224):
        super(Swin_s, self).__init__()
        configuration = SwinConfig(image_size=input_size, patch_size=4, num_channels=num_channels, embed_dim=96,
            depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24], window_size=7, drop_path_rate=0.3)
        
        # Initializing a model (with random weights) from the vit-base-patch16-224 style configuration
        self.swin = SwinModel(configuration)

        self.classifier = nn.Linear(self.swin.num_features, num_class)

    def forward(self, x):
        outputs = self.swin(pixel_values=x)
        
        # see https://github.com/huggingface/transformers/blob/main/src/transformers/models/swin/modeling_swin.py#L1166
        pooled_output = outputs[1]
        output = self.classifier(pooled_output)

        return output


class Swin_b_224(nn.Module):
    def __init__(self, num_class, num_channels, input_size=224):
        super(Swin_b_224, self).__init__()
        configuration = SwinConfig(image_size=input_size, patch_size=4, num_channels=num_channels, embed_dim=128,
            depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=7, drop_path_rate=0.5)
        
        # Initializing a model (with random weights) from the vit-base-patch16-224 style configuration
        self.swin = SwinModel(configuration)

        self.classifier = nn.Linear(self.swin.num_features, num_class)

    def forward(self, x):
        outputs = self.swin(pixel_values=x)
        
        # see https://github.com/huggingface/transformers/blob/main/src/transformers/models/swin/modeling_swin.py#L1166
        pooled_output = outputs[1]
        output = self.classifier(pooled_output)

        return output
    

class Swin_b_384(nn.Module):
    def __init__(self, num_class, num_channels, input_size=384):
        super(Swin_b_384, self).__init__()
        configuration = SwinConfig(image_size=input_size, patch_size=4, num_channels=num_channels, embed_dim=128,
            depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=12, drop_path_rate=0.5)
        
        # Initializing a model (with random weights) from the vit-base-patch16-224 style configuration
        self.swin = SwinModel(configuration)

        self.classifier = nn.Linear(self.swin.num_features, num_class)

    def forward(self, x):
        outputs = self.swin(pixel_values=x)
        
        # see https://github.com/huggingface/transformers/blob/main/src/transformers/models/swin/modeling_swin.py#L1166
        pooled_output = outputs[1]
        output = self.classifier(pooled_output)

        return output


class Swin_l_224(nn.Module):
    def __init__(self, num_class, num_channels, input_size=224):
        super(Swin_l_224, self).__init__()
        configuration = SwinConfig(image_size=input_size, patch_size=4, num_channels=num_channels, embed_dim=192,
            depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48], window_size=7, drop_path_rate=0.2)
        
        # Initializing a model (with random weights) from the vit-base-patch16-224 style configuration
        self.swin = SwinModel(configuration)

        self.classifier = nn.Linear(self.swin.num_features + 512, num_class)

        self.mlp = nn.Sequential(
            nn.Linear(4, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )


    def forward(self, x, loc):
        outputs = self.swin(pixel_values=x)
        
        # see https://github.com/huggingface/transformers/blob/main/src/transformers/models/swin/modeling_swin.py#L1166
        pooled_output = outputs[1]
        mlp_output = self.mlp(loc)
        output = torch.cat((pooled_output, mlp_output), dim=1)

        output = self.classifier(output)

        return output
    

class Swin_l_384(nn.Module):
    def __init__(self, num_class, num_channels, input_size=384):
        super(Swin_l_384, self).__init__()
        configuration = SwinConfig(image_size=input_size, patch_size=4, num_channels=num_channels, embed_dim=192,
            depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48], window_size=12, drop_path_rate=0.2)
        
        # Initializing a model (with random weights) from the vit-base-patch16-224 style configuration
        self.swin = SwinModel(configuration)

        self.classifier = nn.Linear(self.swin.num_features, num_class)

    def forward(self, x):
        outputs = self.swin(pixel_values=x)
        
        # see https://github.com/huggingface/transformers/blob/main/src/transformers/models/swin/modeling_swin.py#L1166
        pooled_output = outputs[1]
        output = self.classifier(pooled_output)

        return output
    

if __name__ == "__main__":
    model = Swin_l_224(num_class=14, num_channels=3, input_size=224)

    x = torch.randn(2, 3, 224, 224)
    loc = torch.randn(2, 4)
    print(model(x, loc).shape)

    # print(torch.hub.list('pytorch/'))