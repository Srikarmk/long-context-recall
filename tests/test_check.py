from lcrecall.check import classify, norm

WRONG = {"same_stem": ["Wexfield"], "same_prefix": ["Wickridge"], "same_suffix": ["Yelmore"]}


def test_norm_strips_case_accents_punctuation():
    assert norm("  Zélie THORNINGHAM.  ") == "zelie thorningham"
    assert norm("**Answer:** Ivelstow!") == "answer ivelstow"


def test_correct_forms():
    for text in ["Zelie Thorningham", "zelie thorningham.", "The collector Zelie Thorningham",
                 "Thorningham", "Answer: Zelie Thorningham", "**Zelie Thorningham**"]:
        assert classify(text, ["Thorningham"], WRONG)["outcome"] == "correct", text


def test_word_boundary_not_substring():
    assert classify("Thorninghamshire", ["Thorningham"], WRONG)["outcome"] == "other"


def test_wrong_entity_and_mixed():
    r = classify("Rosalind Yelmore", ["Thorningham"], WRONG)
    assert r["outcome"] == "wrong_entity" and r["picked"] == "same_suffix" and not r["correct"]
    r = classify("Zelie Thorningham or Tamsin Wexfield", ["Thorningham"], WRONG)
    assert r["outcome"] == "mixed" and not r["correct"]


def test_abstain_and_other():
    for text in ["not stated", "Not stated.", "The document does not say.", "Unknown", "I don't know"]:
        assert classify(text, ["Thorningham"], WRONG)["outcome"] == "abstain", text
    assert classify("Hubert Smith", ["Thorningham"], WRONG)["outcome"] == "other"
    assert classify("", ["Thorningham"], WRONG)["outcome"] == "other"


def test_absent_fact_right_reply_is_to_abstain():
    assert classify("not stated", None, WRONG)["correct"]
    r = classify("Tamsin Wexfield", None, WRONG)
    assert r["outcome"] == "wrong_entity" and not r["correct"]
    assert not classify("Hubert Smith", None, WRONG)["correct"]


def test_twohop_bridge_and_decoy():
    decoys = {"decoy1": ["Dornby"], "decoy2": ["Marrowby"]}
    assert classify("Ivelstow", ["Ivelstow"], decoys, ["Thorningham"])["outcome"] == "correct"
    assert classify("Zelie Thorningham, born in Ivelstow", ["Ivelstow"], decoys, ["Thorningham"])["correct"]
    assert classify("Zelie Thorningham", ["Ivelstow"], decoys, ["Thorningham"])["outcome"] == "bridge"
    r = classify("Dornby", ["Ivelstow"], decoys, ["Thorningham"])
    assert r["outcome"] == "wrong_entity" and r["picked"] == "decoy1"
