import torch

from zoey.models.gpt import GPT, GPTConfig


def tiny():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=300, block_size=32, n_layer=2, n_head=2, d_model=32)).eval()


def test_causal_no_peeking():
    """Changing a future token must not change predictions for earlier positions."""
    model = tiny()
    a = torch.randint(0, 300, (1, 16))
    b = a.clone()
    b[0, 10:] = torch.randint(0, 300, (6,))
    la, _ = model(a)
    lb, _ = model(b)
    assert torch.allclose(la[0, :10], lb[0, :10], atol=1e-5)


def test_recorded_attention_matches_fast_path():
    """The step-by-step attention used for the viewer computes the same thing."""
    model = tiny()
    idx = torch.randint(0, 300, (1, 20))
    fast, _ = model(idx)
    views = model.inspect(idx)
    # The last layer of the logit lens is the model's real output.
    final = torch.softmax(fast[0], dim=-1)
    top_p, _ = final.topk(5, dim=-1)
    assert torch.allclose(views["lens_top_probs"][-1], top_p, atol=1e-5)
    att = views["attention"]
    assert att.shape == (2, 2, 20, 20)
    assert torch.allclose(att.sum(-1), torch.ones(2, 2, 20), atol=1e-5)  # rows sum to 1
    assert att.triu(diagonal=1).abs().max() == 0  # nothing attends to the future


def test_induction_scores_shape():
    assert tiny().induction_scores().shape == (2, 2)
