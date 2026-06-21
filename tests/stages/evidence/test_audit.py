from agastya.stages.evidence.audit import GENESIS_PREV_HASH, append, verify_chain


def test_first_entry_links_to_genesis():
    chain = append((), "evidence_created", {"content_hash": "abc"})
    assert chain[0].seq == 0
    assert chain[0].prev_hash == GENESIS_PREV_HASH


def test_append_links_each_entry_to_predecessor():
    chain = append((), "created", {"id": 1})
    chain = append(chain, "accessed", {"id": 1})
    assert chain[1].seq == 1
    assert chain[1].prev_hash == chain[0].entry_hash
    assert verify_chain(chain) is True


def test_append_is_pure():
    original = append((), "created", {"id": 1})
    append(original, "accessed", {"id": 1})
    assert len(original) == 1


def test_verify_detects_payload_tamper():
    from dataclasses import replace

    chain = append((), "created", {"id": 1})
    tampered = (replace(chain[0], payload_hash="deadbeef"),)
    assert verify_chain(tampered) is False


def test_verify_detects_reordering():
    chain = append((), "created", {"id": 1})
    chain = append(chain, "accessed", {"id": 1})
    reordered = (chain[1], chain[0])
    assert verify_chain(reordered) is False
