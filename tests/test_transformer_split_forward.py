import pytest
import torch

from diffusion_policy.model.diffusion.transformer_for_diffusion import (
    TransformerForDiffusion,
)


def make_decoder() -> TransformerForDiffusion:
    model = TransformerForDiffusion(
        input_dim=4,
        output_dim=4,
        horizon=6,
        n_obs_steps=2,
        cond_dim=8,
        n_layer=12,
        n_head=4,
        n_emb=32,
        p_drop_emb=0.0,
        p_drop_attn=0.0,
        causal_attn=True,
        time_as_cond=True,
        obs_as_cond=True,
        n_cond_layers=1,
    )
    model.eval()
    return model


def test_decoder_9_10_split_matches_original_forward():
    torch.manual_seed(7)
    model = make_decoder()
    sample = torch.randn(2, 6, 4)
    timestep = torch.tensor([11, 73])
    cond = torch.randn(2, 2, 8)

    expected = model(sample, timestep, cond)
    state = model.forward_decoder_pre(
        sample,
        timestep,
        cond,
        split_layer=9,
    )
    actual = model.forward_decoder_post(state)

    assert state.hidden.shape == (2, 6, 32)
    assert state.memory.shape == (2, 3, 32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_frozen_post_block_preserves_feedback_input_gradient():
    torch.manual_seed(13)
    model = make_decoder()
    model.requires_grad_(False)
    sample = torch.randn(2, 6, 4)
    cond = torch.randn(2, 2, 8)

    with torch.no_grad():
        state = model.forward_decoder_pre(sample, 25, cond, split_layer=9)
    feedback = torch.randn_like(state.hidden, requires_grad=True)
    prediction = model.forward_decoder_post(
        state,
        hidden=state.hidden + feedback,
    )
    prediction.square().mean().backward()

    assert feedback.grad is not None
    assert torch.count_nonzero(feedback.grad).item() > 0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_split_forward_rejects_encoder_only_model():
    model = TransformerForDiffusion(
        input_dim=4,
        output_dim=4,
        horizon=6,
        n_layer=2,
        n_head=2,
        n_emb=16,
        time_as_cond=False,
    )
    with pytest.raises(RuntimeError, match="decoder mode"):
        model.forward_decoder_pre(torch.randn(1, 6, 4), 1)
