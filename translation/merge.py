import sentencepiece as spm
from yaml import dump
from pathlib import Path

# def load_spm_model(model_path: Path):
#     sp = spm.SentencePieceProcessor()
#     sp.LoadFromFile(model_path.as_posix())
#     return sp

# def extract_vocab(sp_model):
#     vocab = [sp_model.id_to_piece(i) for i in range(sp_model.vocab_size())]
#     return vocab

# def merge_vocabs(src_vocab, trg_vocab):
#     merged_vocab = src_vocab + [token for token in trg_vocab if token not in src_vocab]
#     return merged_vocab

# def save_vocab_as_yaml(vocab, file_path):
#     vocab_dict = {token: idx for idx, token in enumerate(vocab)}
#     with open(file_path, 'w', encoding='utf-8') as file:
#         dump(vocab_dict, file, allow_unicode=True, sort_keys=False)

# def adjust_token_indices(sp_model, new_vocab):
#     # Create a new SentencePiece model with the adjusted vocabulary
#     new_sp = spm.SentencePieceProcessor()
#     new_sp.load_from_serialized_proto(sp_model.serialized_model_proto())
#     new_sp.set_vocabulary(new_vocab)
#     return new_sp

# basePath = Path("./tmp/downloads/opusTCv20210807+bt-2021-11-10")

# # Load source and target vocabularies
# src_sp = load_spm_model(basePath / "source.spm")
# trg_sp = load_spm_model(basePath / "target.spm")

# # Extract the vocabularies
# src_vocab = extract_vocab(src_sp)
# trg_vocab = extract_vocab(trg_sp)

# save_vocab_as_yaml(src_sp, basePath / "src.vocab.yml")
# save_vocab_as_yaml(src_sp, basePath / "trg.vocab.yml")

basePath = Path("./tmp/downloads/opusTCv20210807+bt-2021-11-10")

def load_vocab(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        vocab = file.read().splitlines()
    return vocab

def save_vocab_as_yaml(vocab, file_path):
    vocab_dict = {token: idx for idx, token in enumerate(vocab)}
    with open(file_path, 'w', encoding='utf-8') as file:
        dump(vocab_dict, file, allow_unicode=True, sort_keys=False)

# Load the vocab file
src = load_vocab(basePath / "opusTCv20210807+bt.spm32k-spm32k.src.vocab")
trg = load_vocab(basePath / "opusTCv20210807+bt.spm32k-spm32k.trg.vocab")

# Save the vocabulary to a YAML file
save_vocab_as_yaml(src, basePath / "opusTCv20210807+bt.spm32k-spm32k.src.vocab.yml")
save_vocab_as_yaml(trg, basePath / "opusTCv20210807+bt.spm32k-spm32k.trg.vocab.yml")