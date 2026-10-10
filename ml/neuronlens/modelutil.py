"""The few places where the code touches the model's layout, in one file.

GPT-2 (GPT2LMHeadModel), DistilGPT-2 (the same class, 6 blocks) and GPT-Neo (GPTNeoForCausalLM)
all have the transformer blocks at model.transformer.h[i], the backbone at model.transformer,
and the unembedding at model.lm_head (checked in the transformers source: modeling_gpt2.py and
modeling_gpt_neo.py). Everything else in neuronlens goes through these helpers, so a model with
a different layout needs a change here only.
"""
DEFAULT_MODEL = "gpt2"


def backbone(model):
    """The module that maps token ids to the final hidden states (no language-model head)."""
    return model.transformer


def get_block(model, layer):
    """Transformer block `layer`; the hook point is the output of this block."""
    return backbone(model).h[layer]


def n_layers(model):
    return len(backbone(model).h)


def d_model_of(model):
    return model.config.hidden_size


def unembedding(model):
    """Output embedding weight (vocab, d_model)."""
    return model.get_output_embeddings().weight


def model_name_from_ck(ck):
    """Hugging Face name of the model a checkpoint was trained on. Checkpoints keep the training
    config in ck["cfg"]; older ones say model_name: gpt2 there, and a missing key means gpt2."""
    return (ck.get("cfg") or {}).get("model_name") or DEFAULT_MODEL


def slug(model_name):
    """Short folder-friendly name: "EleutherAI/gpt-neo-125M" -> "gpt-neo-125m"."""
    return model_name.split("/")[-1].lower()
