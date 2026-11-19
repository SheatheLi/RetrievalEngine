"""Real saved v2 artifact plus self-consistent, synthetic-only exact-service fixture."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil

import lightgbm as lgb
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.exact_diagnostic import ExactDiagnostic
from src.exact_diagnostic_service import create_app
from src.movielens_data import load

SEED7 = Path(__file__).resolve().parents[1] / "models" / "seed7"


def test_authentic_legacy_model_verifies_without_faiss():
    model = ExactDiagnostic.load(SEED7)
    assert model.manifest["schema"] == "two-stage-recommender.v2"
    seen = {1, 2, 3, 4, 5, 6, 7}
    first = model.recommend(1, seen, 60)
    second = model.recommend(1, seen, 60)
    assert first == second
    assert first["backend"] == "numpy_exact_no_faiss"
    assert first["method"] == "exact_ranker" and not first["approximate"]
    ids = [x["item"] for x in first["items"]]
    assert len(ids) == 10 and len(set(ids)) == 10 and not (set(ids) & seen)
    assert set(ids) <= set(model.catalog.tolist())
    with pytest.raises(ValueError, match="history encoder unavailable"):
        model.recommend(1, seen, 60, history=[1])


def test_actual_saved_model_candidates_match_independent_numpy_oracle():
    model = ExactDiagnostic.load(SEED7)
    state = model.state
    seen = {1, 4, 18, 300, 4000}
    expected = [int(x) for x in model.catalog if x not in seen]
    expected.sort(key=lambda i: (-float(np.asarray(state['item_vectors'][i], dtype=np.float64)
                                              @ np.asarray(state['user_vectors'][1], dtype=np.float64)), i))
    # Float32 vectorized BLAS rounding can change only near-ties; check inclusion and score ordering
    ids = model._candidates(state['user_vectors'][1], state['item_vectors'], seen, 70)
    assert len(ids) == 70 and not set(ids) & seen
    scores = (state['item_vectors'][ids] @ state['user_vectors'][1]).astype(float)
    assert np.all(scores[:-1] >= scores[1:] - 1e-6)
    available_scores = state['item_vectors'][np.array(expected)] @ state['user_vectors'][1]
    assert min(scores) >= np.partition(available_scores, -70)[-70] - 1e-5


@pytest.mark.parametrize('component', ['retrieval.npz','ranker.txt','hnsw.index'])
def test_legacy_model_rejects_corrupt_component(tmp_path, component):
    folder = tmp_path / 'copy'
    shutil.copytree(SEED7, folder)
    with (folder / component).open('ab') as output:
        output.write(b'corrupt')
    with pytest.raises(ValueError, match='model/index version mismatch'):
        ExactDiagnostic.load(folder)


def test_legacy_model_requires_complete_manifest(tmp_path):
    folder = tmp_path / 'copy'
    shutil.copytree(SEED7, folder)
    path = folder / 'model.json'
    meta = json.loads(path.read_text())
    del meta['sha256']['hnsw.index']
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='incomplete artifact'):
        ExactDiagnostic.load(folder)


def make_synthetic_fixture(tmp_path):
    """Fresh data-dependent exact-only model; never claims MovieLens/FAISS training."""
    rng = np.random.default_rng(164)
    records = []
    for timestamp in range(1, 401):
        user = 8 if 270 <= timestamp <= 279 else 9 if 330 <= timestamp <= 334 else 1 + timestamp % 7
        item = 1 + (timestamp * 11 + timestamp // 9) % 35
        rating = 2 if user == 9 else 5 if timestamp % 3 else 2
        records.append((user, item, rating, timestamp))
    data_file = tmp_path / 'ratings.tsv'
    np.savetxt(data_file, np.array(records), fmt='%d', delimiter='\t')
    data = load(data_file)
    dimensions = 8
    item_vectors = rng.normal(0, .1, size=(data.items, dimensions)).astype('float32')
    item_vectors[0] = 0
    user_vectors = np.zeros((data.users, dimensions), dtype='float32')
    interactions = np.zeros((data.users, data.items), dtype='float32')
    for user, item, rating, _ in data.train:
        if rating >= 4:
            interactions[user, item] = 1
    for user in range(1, data.users):
        ids = np.flatnonzero(interactions[user])
        user_vectors[user] = item_vectors[ids].mean(axis=0) if len(ids) else 0
    counts = interactions.sum(axis=0)
    user_counts = interactions.sum(axis=1)
    similarity = interactions.T @ interactions
    np.fill_diagonal(similarity, 0)
    # A genuine train-prefix-derived ranker fixture, not random targets.
    # Labels mean observed synthetic train positives vs *unobserved proxies*.
    features, labels = [], []
    for user in range(1, data.users):
        positives = np.flatnonzero(interactions[user] > 0)
        unseen = np.asarray(sorted(set(data.catalog.tolist()) - set(positives.tolist())), dtype=int)
        if not len(positives) or not len(unseen):
            continue
        candidates = np.r_[positives[:8], unseen[:8]]
        matrix = np.column_stack((
            item_vectors[candidates] @ user_vectors[user],
            np.log1p(counts[candidates]),
            interactions[user] @ similarity[:, candidates],
            np.full(len(candidates), np.log1p(user_counts[user])),
            np.linalg.norm(item_vectors[candidates], axis=1))).astype('float32')
        features.append(matrix)
        labels.extend([1] * min(8, len(positives)) + [0] * min(8, len(unseen)))
    ranker = lgb.train({'objective':'regression','verbosity':-1,'num_threads':1,
                        'min_data_in_leaf':1,'min_data_in_bin':1},
                        lgb.Dataset(np.vstack(features),label=np.asarray(labels)),num_boost_round=8)
    model_folder = tmp_path / 'synthetic_exact_model'
    model_folder.mkdir()
    from src.history_retrieval import fit_history_encoder
    history, loss, examples = fit_history_encoder(data.train, data.catalog, data.items,
                                                 seed=17, epochs=2, dim=dimensions)
    np.savez_compressed(model_folder/'retrieval.npz',user_vectors=user_vectors,
        item_vectors=item_vectors,catalog=data.catalog,popularity=counts,
        user_counts=user_counts,interaction=interactions,similarity=similarity,
        history_item_vectors=history.items.weight.detach().cpu().numpy().copy(),
        history_projection=history.projection.weight.detach().cpu().numpy().copy())
    ranker.save_model(str(model_folder/'ranker.txt'))
    filenames = ('retrieval.npz','ranker.txt')
    hashes = {name:sha256((model_folder/name).read_bytes()).hexdigest() for name in filenames}
    (model_folder/'model.json').write_text(json.dumps({
       'schema':'two-stage-recommender.exact.v1','sha256':hashes,
       'input_sha256':data.input_sha256,'history_max_items':history.max_history,
       'purpose':'SYNTHETIC_TEST_FIXTURE_NOT_RECOMMENDATION_QUALITY'}))
    return model_folder, data_file, data


def test_exact_service_live_contract_and_strict_lineage(tmp_path):
    folder, data_file, data = make_synthetic_fixture(tmp_path)
    client = TestClient(create_app(folder, data_file))
    health = client.get('/health')
    assert health.status_code == 200 and health.json()['approximate'] is False
    u = int(data.train[data.train[:,2]>=4,0].min())
    response = client.get(f'/recommend/{u}?budget=12')
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['method'] == 'exact_ranker' and not result['approximate']
    seen = set(data.prefix('test')[data.prefix('test')[:,0] == u,1].tolist())
    ids = [x['item'] for x in result['items']]
    assert ids and not set(ids) & seen
    cold = client.get('/recommend/8?budget=12')
    assert cold.status_code == 200, cold.text
    assert cold.json()['representation'] == 'history_encoder'
    assert cold.json()['method'] == 'history_exact'
    assert not set(x['item'] for x in cold.json()['items']) & set(
        data.prefix('test')[data.prefix('test')[:,0] == 8,1].tolist())
    assert client.get('/recommend/9?budget=12').status_code == 422
    assert client.get('/recommend/999').status_code == 422
    assert client.get('/recommend/1?budget=0').status_code == 422
    assert client.get('/recommend/1?budget=10000').status_code == 422
    with data_file.open('ab') as output:
        output.write(b'\n')
    with pytest.raises(ValueError, match='serving data/model mismatch'):
        create_app(folder, data_file)


def test_exact_fixture_tamper_then_rehash_still_rejects_shape(tmp_path):
    folder, data_file, data = make_synthetic_fixture(tmp_path)
    with np.load(folder/'retrieval.npz', allow_pickle=False) as archive:
        arrays = {name:archive[name].copy() for name in archive.files}
    arrays['similarity'] = arrays['similarity'][:-1]
    np.savez_compressed(folder/'retrieval.npz', **arrays)
    manifest = json.loads((folder/'model.json').read_text())
    manifest['sha256']['retrieval.npz']=sha256((folder/'retrieval.npz').read_bytes()).hexdigest()
    (folder/'model.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='invalid model representation dimensions'):
        ExactDiagnostic.load(folder)
