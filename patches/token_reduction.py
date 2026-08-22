"""Token Reduction for MiniMax H3 DiT.

Ports the h3-metal token reduction technique: in middle DiT blocks, spatial tokens
are pooled (2x2 average) to reduce compute, then upsampled back before the final
layers. This saves ~40% compute in the middle blocks with minimal quality loss.

Algorithm (from h3-metal h3_dit.c):
1. After block N, detect spatial tokens (video segments)
2. Apply 2x2 average pooling to reduce token count by 4x
3. Continue through remaining blocks with fewer tokens
4. Before final_layer, upsample back to original resolution
"""

import torch
import torch.nn.functional as F


class TokenReducer:
    """Manages token reduction state during DiT forward pass.

    Usage:
        reducer = TokenReducer(enabled=True, start_block=10, end_block=40)
        for i, block in enumerate(model.blocks):
            if reducer.should_reduce(i):
                h = reducer.reduce(h, latent_h, latent_w)
            h = block(h, ...)
            if reducer.should_restore(i):
                h = reducer.restore(h, original_latent_h, original_latent_w)
    """

    def __init__(self, enabled=True, start_block=10, end_block=40, pool_factor=2):
        self.enabled = enabled
        self.start_block = start_block
        self.end_block = end_block
        self.pool_factor = pool_factor
        self._original_shape = None
        self._reduced = False

    def should_reduce(self, block_index):
        return self.enabled and block_index == self.start_block and not self._reduced

    def should_restore(self, block_index):
        return self.enabled and block_index == self.end_block - 1 and self._reduced

    def get_reduced_seq_len(self, seq_len, latent_h, latent_w, n_frames=1):
        """Compute reduced sequence length after pooling."""
        if not self.enabled:
            return seq_len
        spatial_tokens = latent_h * latent_w * n_frames
        audio_tokens = seq_len - spatial_tokens
        reduced_spatial = spatial_tokens // (self.pool_factor ** 2)
        return audio_tokens + reduced_spatial

    def reduce(self, h, mod_segments, latent_h, latent_w, n_frames=1, patch_size=(1, 2, 2)):
        """Reduce spatial tokens via average pooling.

        h: [seq_len, hidden_size]
        Returns: [reduced_seq_len, hidden_size]
        """
        self._original_shape = h.shape
        self._reduced = True

        # Identify video segment (last segment is always video)
        video_start = None
        video_end = None
        audio_start = None
        audio_end = None
        for i, (a, b, kind) in enumerate(mod_segments):
            if kind == "video":
                video_start, video_end = a, b
            elif kind == "audio":
                audio_start, audio_end = a, b

        if video_start is None:
            return h

        hidden = h.shape[-1]
        pf = self.pool_factor

        # Extract video tokens
        video_tokens = h[video_start:video_end]  # [n_video, hidden]
        n_video = video_tokens.shape[0]

        # Reshape to spatial grid: [n_frames, latent_h//patch_h, latent_w//patch_w, hidden]
        # Video has patch_size=(1,2,2), so spatial per frame = (latent_h//2) * (latent_w//2)
        patch_h, patch_w = patch_size[1], patch_size[2]
        frame_h = latent_h // patch_h
        frame_w = latent_w // patch_w
        tokens_per_frame = frame_h * frame_w

        if n_video != n_frames * tokens_per_frame:
            # Fallback: can't reshape cleanly, skip reduction
            return h

        # Reshape: [n_frames, frame_h, frame_w, hidden]
        vf = video_tokens.reshape(n_frames, frame_h, frame_w, hidden)

        # 2x2 average pooling on spatial dims
        vf = vf.permute(0, 3, 1, 2)  # [n_frames, hidden, frame_h, frame_w]
        vf = F.avg_pool2d(vf.unsqueeze(0), kernel_size=pf, stride=pf).squeeze(0)
        vf = vf.permute(0, 2, 3, 1)  # [n_frames, frame_h//pf, frame_w//pf, hidden]

        reduced_video = vf.reshape(-1, hidden)

        # Build new sequence: non-video segments + reduced video
        parts = []
        for a, b, kind in mod_segments:
            if kind == "video":
                parts.append(reduced_video)
            else:
                parts.append(h[a:b])

        return torch.cat(parts, dim=0)

    def restore(self, h, mod_segments, original_latent_h, original_latent_w,
                n_frames=1, patch_size=(1, 2, 2)):
        """Restore spatial tokens to original resolution via nearest-upsample.

        h: [reduced_seq_len, hidden_size]
        Returns: [original_seq_len, hidden_size]
        """
        if not self._reduced or self._original_shape is None:
            return h

        self._reduced = False
        hidden = h.shape[-1]
        pf = self.pool_factor

        # Find reduced video segment
        video_start = None
        video_end = None
        for a, b, kind in mod_segments:
            if kind == "video":
                video_start, video_end = a, b

        if video_start is None:
            return h

        reduced_video = h[video_start:video_end]
        patch_h, patch_w = patch_size[1], patch_size[2]
        frame_h = original_latent_h // patch_h
        frame_w = original_latent_w // patch_w

        reduced_frame_h = frame_h // pf
        reduced_frame_w = frame_w // pf

        n_reduced = reduced_video.shape[0]
        if n_reduced != n_frames * reduced_frame_h * reduced_frame_w:
            return h

        # Reshape: [n_frames, reduced_frame_h, reduced_frame_w, hidden]
        vf = reduced_video.reshape(n_frames, reduced_frame_h, reduced_frame_w, hidden)
        vf = vf.permute(0, 3, 1, 2)  # [n_frames, hidden, rfh, rfw]

        # Nearest-neighbor upsample
        vf = F.interpolate(vf, size=(frame_h, frame_w), mode='nearest')
        vf = vf.permute(0, 2, 3, 1)  # [n_frames, frame_h, frame_w, hidden]
        restored_video = vf.reshape(-1, hidden)

        # Rebuild full sequence
        parts = []
        for a, b, kind in mod_segments:
            if kind == "video":
                parts.append(restored_video)
            else:
                parts.append(h[a:b])

        return torch.cat(parts, dim=0)


def create_token_reduction_hook(reducer, latent_h, latent_w, n_frames=1, patch_size=(1, 2, 2)):
    """Create a hook that applies token reduction after a specific DiT block.

    Returns a function suitable for model patcher's patches_replace mechanism.
    """
    def hook(args):
        h = args["img"]
        t_emb = args["t_emb"]
        mod_segments = args["mod_segments"]
        rope_freqs = args["rope_freqs"]
        transformer_options = args["transformer_options"]

        block_idx = args.get("block_index", 0)

        if reducer.should_reduce(block_idx):
            h = reducer.reduce(h, mod_segments, latent_h, latent_w, n_frames, patch_size)

        return {"img": h, "t_emb": t_emb, "mod_segments": mod_segments,
                "rope_freqs": rope_freqs, "transformer_options": transformer_options}

    return hook
