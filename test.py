# import torch

# from src.stage1.modules.dynamic_modules.decoder import (
#     Decoder, PositionEmbedding2DLearned)


# def test_PositionEmbedding2DLearned():
#     n_row, feats_dim, n_col = 10, 64, 10
#     model = PositionEmbedding2DLearned(n_row, feats_dim, n_col)
#     x = torch.randn(1, feats_dim, n_row, n_col)
#     output = model(x)
#     print(f"PositionEmbedding2DLearned output shape: {output.shape}")


# def test_Decoder():
#     ch, in_ch, out_ch = 64, 3, 3
#     ch_mult = [1, 2, 4]
#     num_res_blocks = 2
#     resolution = 64
#     attn_resolutions = [16]
#     latent_size = 64
#     window_size = 2
#     position_type = "rope"

#     model = Decoder(
#         ch,
#         in_ch,
#         out_ch,
#         ch_mult,
#         num_res_blocks,
#         resolution,
#         attn_resolutions,
#         latent_size=latent_size,
#         window_size=window_size,
#         position_type=position_type,
#     )
#     h = torch.randn(1, in_ch, resolution, resolution)
#     output = model(h)
#     print(f"Decoder output shape: {output.shape}")


# if __name__ == "__main__":
#     test_PositionEmbedding2DLearned()
#     test_Decoder()




# import torch
# from src.stage1.modules.dynamic_modules.router_dual import (
#     DualGrainFeatureRouter,
#     DualGrainFixedEntropyRouter,
#     DualGrainDynamicEntropyRouter,
# )

# def test_feature_router():
#     num_channels = 16
#     num_groups = 2
#     router = DualGrainFeatureRouter(num_channels, num_groups)
#     h_coarse = torch.randn(1, num_channels, 8, 8)
#     h_fine = torch.randn(1, num_channels,16 , 16)
#     gate = router.forward(h_coarse, h_fine)
#     print(f"Feature Router output shape: {gate.shape}")

# def test_fixed_entropy_router():
#     entropy_router_fixed = DualGrainFixedEntropyRouter("D:\\AwesomeCV\\DynamicVectorQuantization\\scripts\\tools\\thresholds\\entropy_thresholds_imagenet_train_patch-16.json", 0.5)
#     entropy = torch.tensor([0.3, 0.7, 0.5])
#     gate = entropy_router_fixed.forward(entropy)
#     print(f"Fixed Entropy Router output shape: {gate.shape}")

# def test_dynamic_entropy_router():
#     entropy_router_dynamic = DualGrainDynamicEntropyRouter()
#     entropy = torch.tensor([0.3, 0.7, 0.5])
#     gate = entropy_router_dynamic.forward(entropy)
#     print(f"Dynamic Entropy Router output shape: {gate.shape}")

# def test_get_gate_from_threshold():
#     entropy_router_fixed = DualGrainFixedEntropyRouter("D:\\AwesomeCV\\DynamicVectorQuantization\\scripts\\tools\\thresholds\\entropy_thresholds_imagenet_train_patch-16.json", 0.5)
#     entropy = torch.tensor([0.3, 0.7, 0.5])
#     threshold = 0.5
#     gate = entropy_router_fixed._get_gate_from_threshold(entropy, threshold)
#     print(f"Gate from threshold output shape: {gate.shape}")

# if __name__ == '__main__':
#     test_feature_router()
#     test_fixed_entropy_router()
#     test_dynamic_entropy_router()
#     test_get_gate_from_threshold()



import torch
from src.stage1.modules.dynamic_modules.encoder_dual import DualGrainEncoder

# Define the configuration as per the provided YAML
# config = {
#     'ch': 128,
#     'ch_mult': [1, 1, 2, 2, 4],
#     'num_res_blocks': 2,
#     'attn_resolutions': [16, 32],
#     'dropout': 0.0,
#     'resamp_with_conv': True,
#     'in_channels': 3,
#     'resolution': 256,
#     'z_channels': 256,
#     'router_config': {
#         'target': 'src.stage1.modules.dynamic_modules.router_dual.DualGrainFeatureRouter',
#         'params': {
#             'num_channels': 256,
#             'num_groups': 32,
#         }
#     }
# }


config = {
    'ch': 128,
    'ch_mult': [1, 1, 2, 2, 4],
    'num_res_blocks': 2,
    'attn_resolutions': [16, 32],
    'dropout': 0.0,
    'resamp_with_conv': True,
    'in_channels': 3,
    'resolution': 256,
    'z_channels': 256,
    'router_config': {
        'target': 'src.stage1.modules.dynamic_modules.router_dual.DualGrainFixedEntropyRouter',
        'params': {   
            'json_path': 'D:\\AwesomeCV\\DynamicVectorQuantization\\scripts\\tools\\thresholds\\entropy_thresholds_imagenet_train_patch-16.json',
            'fine_grain_ratito': 0.5
        }
    }
}


# Instantiate the encoder
encoder = DualGrainEncoder(**config)

# Create a sample input tensor
x = torch.randn(1, 3, 256, 256)  # Batch size of 1, 3 channels, 256x256 resolution
x_entropy = torch.randn(1, 1, 256, 256)  # Sample entropy tensor

# Forward pass through the encoder
output = encoder(x, x_entropy)

# Print the outputs
print("h_dual:", output['h_dual'].shape)
print("indices:", output['indices'].shape)
print("codebook_mask:", output['codebook_mask'].shape)
print("gate:", output['gate'].shape)