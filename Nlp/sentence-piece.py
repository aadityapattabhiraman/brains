#!/home/akugyo/Programs/Python/torch/bin/python

import sentencepiece
import os


# write a toy.txt file with some random text
with open("toy.txt", "w", encoding="utf-8") as f:
    f.write("SentencePiece is an unsupervised text tokenizer and detokenizer mainly for Neural Network-based text generation systems where the vocabulary size is predetermined prior to the neural model training. SentencePiece implements subword units (e.g., byte-pair-encoding (BPE) [Sennrich et al.]) and unigram language model [Kudo.]) with the extension of direct training from raw sentences. SentencePiece allows us to make a purely end-to-end system that does not depend on language-specific pre/postprocessing.")

options = dict(input="toy.txt",
               input_format="text",
               model_prefix="tok400",
               model_type="bpe",
               vocab_size=400,
               normalization_rule_name="identity",
               remove_extra_whitespaces=False,
               input_sentence_size=200000000,
               max_sentence_length=4192,
               seed_sentencepiece_size=1000000,
               shuffle_input_sentence=True,
               character_coverage=0.99995,
               byte_fallback=True,
               split_digits=True,
               split_by_unicode_script=True,
               split_by_whitespace=True,
               split_by_number=True,
               max_sentencepiece_length=16,
               add_dummy_prefix=True,
               allow_whitespace_only_pieces=True,
               unk_id=0,
               bos_id=1,
               eos_id=2,
               pad_id=-1,
               num_threads=os.cpu_count(),
)

sentencepiece.SentencePieceTrainer.train(**options)

sp = sentencepiece.SentencePieceProcessor()
sp.load("tok400.model")
vocab = [[sp.id_to_piece(idx), idx] for idx in range(sp.get_piece_size())]

ids = sp.encode("hello 안녕하세요")
print(ids)

print([sp.id_to_piece(idx) for idx in ids])
