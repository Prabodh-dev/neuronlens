import itertools
from neuronlens.activations import ActivationStream, token_sequences
from neuronlens.evals import evaluate


class IdentitySAE:
    d_model = 768
    d_sae = 768

    def __call__(self, x):
        return x, (x != 0).float(), x


def test_identity_sae_recovers_everything():
    s = ActivationStream(buffer_size=100_000)
    seqs = list(itertools.islice(
        token_sequences(s.tok, "~/data/fineweb/sample/10BT", seed=123), 16))
    r = evaluate(s.model, IdentitySAE(), seqs, 8, s.mu, s.scale)
    print(r)
    assert r["fvu"] < 1e-3
    assert r["ce_zero"] > r["ce_orig"] + 1.0
    assert r["ce_recovered"] > 0.99


if __name__ == "__main__":
    test_identity_sae_recovers_everything()
    print("ok")