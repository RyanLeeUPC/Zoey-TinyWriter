from zoey.bpe import BPETokenizer

TEXT = (
    "Once upon a time, there was a little girl named Lily. She loved to play. "
    "One day, Lily saw a big dog. The dog was happy! Lily and the dog played all day.\n"
) * 50


def test_learns_common_words():
    tok = BPETokenizer.train(TEXT, vocab_size=300, verbose=False)
    ids = tok.encode(" Lily")
    assert len(ids) == 1, "a frequent word should become a single token"


def test_round_trip_any_text():
    tok = BPETokenizer.train(TEXT, vocab_size=300, verbose=False)
    for s in [TEXT, "unseen wörds, emoji 🐶 and   spaces\n\n", "", "a_b-c 123"]:
        assert tok.decode(tok.encode(s)) == s


def test_merges_applied_in_learned_order():
    tok = BPETokenizer.train("ab ab ab abc abc", vocab_size=260, verbose=False)
    # "a"+"b" is the most common pair, so it must be merge #1 (id 256).
    assert tok.merges[0] == (ord("a"), ord("b"))
    assert tok.encode("ab") == [256]
