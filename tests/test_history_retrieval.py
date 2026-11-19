"""Independent tests that run without FAISS; FAISS integration has a separate gate."""
import numpy as np
import pytest
from src.history_retrieval import HistoryEncoder, fit_history_encoder, history_examples


def data():
    # Same-time items for user 1 may not train on one another.
    return np.asarray([
        [1, 1, 5, 10], [1, 2, 5, 11], [1, 3, 5, 11],
        [1, 4, 5, 12], [2, 2, 5, 10], [2, 4, 5, 11],
        [3, 5, 5, 50],
    ], dtype=np.int64)


def test_same_timestamp_never_enters_query_context():
    examples = history_examples(data(), np.arange(1, 7))
    tie = [e for e in examples if e.user == 1 and e.timestamp == 11]
    assert len(tie) == 2
    assert all(e.context == (1,) for e in tie)
    assert all({2, 3} <= e.excluded for e in tie)
    assert next(e for e in examples if e.user == 1 and e.timestamp == 12).context == (1, 2, 3)


def test_catalogue_and_strict_time_boundaries():
    examples = history_examples(data(), np.array([1, 2, 3, 4]))
    assert all(all(item in (1, 2, 3, 4) for item in e.context) for e in examples)
    assert not any(e.user == 3 for e in examples)
    assert not any(e.target == 5 for e in examples)


def test_inductive_train_query_generalizes_without_user_id():
    model, trace, samples = fit_history_encoder(data(), np.arange(1, 7), 7, 81, epochs=10, dim=8)
    assert samples >= 3 and len(trace) == 10
    assert all(np.isfinite(trace))
    # A new user with known positive history can query the same item geometry.
    known = model.encode([1, 2], np.arange(1, 7))
    assert known.shape == (8,) and np.isfinite(known).all()
    assert np.array_equal(known, model.encode([1, 2], np.arange(1, 7)))
    with pytest.raises(ValueError, match='no known positive history'):
        model.encode([99], np.arange(1, 7))


def test_rejects_training_with_no_history_and_empty_catalogue():
    with pytest.raises(ValueError, match='no strictly historical'):
        fit_history_encoder(data()[:1], np.arange(1, 7), 7, 1)
    with pytest.raises(ValueError, match='empty training catalog'):
        history_examples(data(), np.asarray([], dtype=int))


def test_cohort_diagnostic_covers_usable_history_not_just_trained_ids():
    from src.movielens_data import TemporalData
    train = np.asarray([[1, 1, 5, 1], [1, 2, 5, 2], [2, 1, 1, 2]], dtype=np.int64)
    rank = np.asarray([[2, 1, 5, 3], [3, 9, 5, 3]], dtype=np.int64)
    validation = np.asarray([[4, 1, 2, 4]], dtype=np.int64)
    test = np.asarray([[1, 1, 5, 5], [2, 2, 5, 5], [3, 2, 5, 5], [4, 2, 5, 5]], dtype=np.int64)
    example = TemporalData(train, rank, validation, test, 5, 10, np.array([1,2]), (2,4,5), 'fixture')
    result = example.query_cohort_audit('test')
    # User 1 is trained, but no *new* catalogue positive: repeated item 1
    assert result['counts']['no_evaluable_future_positive'] == 1
    assert result['counts']['history_only'] == 1
    assert result['counts']['history_items_not_in_training_catalog'] == 1
    assert result['counts']['no_positive_history'] == 1
    assert next(q for q in result['queries'] if q['user'] == 2)['known_positive'] == [1]


def test_history_encode_uses_prefix_order_and_last_32():
    encoder = HistoryEncoder(items=40, dim=4)
    allow = np.arange(1, 40)
    a = encoder.encode(list(range(1, 36)), allow)
    b = encoder.encode(list(range(4, 36)), allow)
    assert np.allclose(a, b)


def test_history_sequence_required_not_unordered_set():
    model = HistoryEncoder(7, dim=4)
    with pytest.raises(ValueError, match='timestamp ordered'):
        model.encode({1, 2}, np.arange(1, 7))
