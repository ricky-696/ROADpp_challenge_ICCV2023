import torch
import torch.nn as nn

from transformers import ViTConfig, ViTModel


class ViT_b(nn.Module):
    def __init__(self, num_class, num_channels, input_size=224):
        super(ViT_b, self).__init__()
        configuration = ViTConfig(hidden_size = 768, num_hidden_layers = 12, num_attention_heads = 12,
            intermediate_size = 3072, image_size = input_size, num_channels = num_channels)

        # Initializing a model (with random weights) from the vit-base-patch16-224 style configuration
        self.vit = ViTModel(configuration)

        self.classifier = nn.Linear(self.vit.config.hidden_size, num_class)

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        
        # see https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py#L813
        sequence_output = outputs[0]
        output = self.classifier(sequence_output[:, 0, :])

        return output
    

class ViT_l(nn.Module):
    def __init__(self, num_class, num_channels, input_size=224):
        super(ViT_l, self).__init__()
        configuration = ViTConfig(hidden_size = 1024, num_hidden_layers = 24, num_attention_heads = 16,
            intermediate_size = 4096, image_size = input_size, num_channels = num_channels)

        # Initializing a model (with random weights) from the vit-base-patch16-224 style configuration
        self.vit = ViTModel(configuration)

        self.classifier = nn.Linear(self.vit.config.hidden_size, num_class)

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        
        # see https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py#L813
        sequence_output = outputs[0]
        output = self.classifier(sequence_output[:, 0, :])

        return output
    

class ViT_h(nn.Module):
    def __init__(self, num_class, num_channels, input_size=224):
        super(ViT_h, self).__init__()
        configuration = ViTConfig(hidden_size = 1280, num_hidden_layers = 32, num_attention_heads = 16,
            intermediate_size = 5120, image_size = input_size, num_channels = num_channels)

        # Initializing a model (with random weights) from the vit-base-patch16-224 style configuration
        self.vit = ViTModel(configuration)

        self.classifier = nn.Linear(self.vit.config.hidden_size, num_class)

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        
        # see https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py#L813
        sequence_output = outputs[0]
        output = self.classifier(sequence_output[:, 0, :])

        return output


if __name__ == "__main__":
    model = ViT_l('', num_class=22, num_channels=24, input_size=224)

    x = torch.randn(2, 24, 224, 224)
    print(model(x).shape)

    # print(torch.hub.list('pytorch/'))