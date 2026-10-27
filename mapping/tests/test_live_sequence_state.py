from mapping.robot.go2_mapping_lio.go2_mapping_lio.sequence_state import KeyframeSequenceState


def test_keyframe_sequence_state_recovers_the_last_confirmed_sequence(tmp_path):
    state = KeyframeSequenceState(tmp_path / "live-preview.sequence")

    assert state.load() == 0
    state.store(2395)

    assert state.load() == 2395


def test_keyframe_sequence_state_treats_a_corrupt_or_missing_file_as_uninitialized(tmp_path):
    path = tmp_path / "live-preview.sequence"
    path.write_text("not-a-sequence", encoding="ascii")

    assert KeyframeSequenceState(path).load() == 0
