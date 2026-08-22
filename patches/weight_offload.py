"""Weight Offloading for MiniMax H3.

Implements SSD streaming-style weight offloading for ComfyUI:
- Text encoder: process layer-by-layer, only 1 layer on GPU at a time
- DiT: offload inactive blocks to CPU, prefetch upcoming blocks
- VAE: tiled decode with minimal memory footprint
"""

import gc
import torch
import comfy.model_management


class TextEncoderOffloader:
    def __init__(self, text_encoder, device=None):
        self.text_encoder = text_encoder
        self.device = device or comfy.model_management.text_encoder_device()
        self.layers = []
        self._extract_layers()

    def _extract_layers(self):
        model = self.text_encoder
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            self.layers = list(model.model.layers)
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
            self.layers = list(model.transformer.layers)
        else:
            for attr_name in ['layers', 'block', 'encoder_layer']:
                if hasattr(model, attr_name):
                    self.layers = list(getattr(model, attr_name))
                    break

    def encode_layer_by_layer(self, input_ids, attention_mask=None):
        if not self.layers:
            raise RuntimeError("No transformer layers found in text encoder")

        model = self.text_encoder
        device = self.device

        if hasattr(model, 'get_input_embeddings'):
            hidden_states = model.get_input_embeddings()(input_ids)
        else:
            raise RuntimeError("Cannot get input embeddings")

        for i, layer in enumerate(self.layers):
            layer.to(device)
            layer.eval()

            with torch.no_grad():
                layer_output = layer(hidden_states.to(device))
                if hasattr(layer_output, 'last_hidden_state'):
                    hidden_states = layer_output.last_hidden_state
                elif isinstance(layer_output, tuple):
                    hidden_states = layer_output[0]
                else:
                    hidden_states = layer_output

            layer.cpu()
            hidden_states = hidden_states.cpu()

            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

        return hidden_states.to(device)


class DiTBlockOffloader:
    def __init__(self, model, prefetch_depth=1):
        self.model = model
        self.prefetch_depth = prefetch_depth
        self._offloaded = set()

    def offload_all(self):
        for block in self.model.blocks:
            block.cpu()
            self._offloaded.add(id(block))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def prefetch_blocks(self, indices):
        device = next(self.model.parameters()).device
        for idx in indices:
            if idx < len(self.model.blocks):
                self.model.blocks[idx].to(device)
                self._offloaded.discard(id(self.model.blocks[idx]))

    def offload_blocks(self, indices):
        for idx in indices:
            if idx < len(self.model.blocks):
                self.model.blocks[idx].cpu()
                self._offloaded.add(id(self.model.blocks[idx]))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
