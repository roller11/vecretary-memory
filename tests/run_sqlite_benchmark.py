# SPDX-License-Identifier: MPL-2.0
"""Opt-in synthetic exact-scan benchmark; never opens an existing store."""

import argparse
import json
import math
from pathlib import Path
import platform
import random
import sqlite3
import statistics
import tempfile
import time

from instinct.sqlite_records import EmbeddingIdentity, MAX_DIMENSION, MAX_RECORDS, SQLiteRecord
from instinct.sqlite_search import vector_distance
from instinct.sqlite_search_cache import MAX_CACHE_BYTES, MAX_CACHE_RECORDS
from instinct.sqlite_store import SQLiteRecordStore


def summarize(values):
    ordered = sorted(values)
    return {'samples': len(values), 'median_ms': statistics.median(values),
            'min_ms': ordered[0], 'max_ms': ordered[-1],
            'p95_ms': ordered[math.ceil(len(values) * 0.95) - 1] if len(values) >= 20 else None}


def timed(action):
    started = time.perf_counter()
    result = action()
    return result, (time.perf_counter() - started) * 1000


def checkpoint(output, results):
    output.seek(0)
    json.dump(results, output, indent=2, allow_nan=False)
    output.write('\n')
    output.truncate()
    output.flush()


def reference(records, query, *, filtered=False):
    eligible = [record for record in records.values()
                if not json.loads(record.metadata_json).get('recycled_at')
                and (not filtered or (json.loads(record.metadata_json)['importance_score'] >= 70
                                      and json.loads(record.metadata_json)['segment_type'] == 'context'))]
    ranked = sorted(((vector_distance(query, record.embedding, 'cosine'), record.record_id)
                     for record in eligible))[:10]
    return [(record_id, distance) for distance, record_id in ranked]


def verify(hits, expected):
    actual = [(hit.record.record_id, hit.distance) for hit in hits]
    if actual != expected:
        raise AssertionError('Exact reference ranking or distance mismatch')


def check_bounds(store):
    cache = store._search_cache
    if len(cache._entries) > MAX_CACHE_RECORDS or not 0 <= cache.retained_bytes <= MAX_CACHE_BYTES:
        raise AssertionError('Search cache exceeded its configured bound')
    return {'records': len(cache._entries), 'accounted_bytes': cache.retained_bytes}


def fixture(identity, count, seed):
    generator = random.Random(seed)
    records = {}
    for index in range(count):
        vector = [generator.uniform(-1, 1) for _ in range(identity.dimension)]
        if index == 0:
            vector = [0.0] * identity.dimension
        metadata = {'importance_score': 50 + index % 40,
                    'segment_type': 'context' if index % 2 else 'implementation_detail'}
        if index and index % 17 == 0:
            metadata['recycled_at'] = '2026-01-01'
        record = SQLiteRecord.from_dict({'id': f'synthetic-{index:05d}',
                                        'content': 'Synthetic benchmark fixture',
                                        'metadata': metadata, 'embedding': vector}, identity)
        records[record.record_id] = record
    queries = [[generator.uniform(-1, 1) for _ in range(identity.dimension)] for _ in range(4)]
    return records, queries


def benchmark(root, count, dimension, samples, seed):
    identity = EmbeddingIdentity('synthetic', 'benchmark-fixed', dimension, 'cosine')
    records, queries = fixture(identity, count, seed)
    expected = [reference(records, query) for query in queries]
    scenarios = {}
    started = time.perf_counter()
    with SQLiteRecordStore.create(root, identity) as store:
        items = list(records.values())
        for offset in range(0, count, 100):
            store.insert_records(items[offset:offset + 100])
        insertion_ms = (time.perf_counter() - started) * 1000
        hits, cold_ms = timed(lambda: store.search_vectors(queries[0]))
        verify(hits, expected[0])
        check_bounds(store)
        warm = []
        for index in range(samples):
            query_index = index % len(queries)
            hits, elapsed = timed(lambda: store.search_vectors(queries[query_index]))
            verify(hits, expected[query_index])
            check_bounds(store)
            warm.append(elapsed)
        scenarios['warm_varied'] = summarize(warm)
        filtered_expected = reference(records, queries[0], filtered=True)
        filtered = []
        for _ in range(samples):
            hits, elapsed = timed(lambda: store.search_vectors(queries[0], min_importance=70,
                                                             segment_types=['context']))
            verify(hits, filtered_expected)
            check_bounds(store)
            filtered.append(elapsed)
        scenarios['filtered'] = summarize(filtered)
        for name, update_count in [('top10_metadata', min(10, count)),
                                   ('corpus10pct_metadata', max(1, count // 10))]:
            targets = ([record_id for record_id, _ in expected[0]] if name == 'top10_metadata'
                       else list(records)[:update_count])
            mutation_times = []
            search_times = []
            for iteration in range(samples):
                def mutate():
                    for record_id in targets:
                        records[record_id] = store.patch_metadata(record_id, {'benchmark_revision': iteration})
                _, mutation_ms = timed(mutate)
                hits, elapsed = timed(lambda: store.search_vectors(queries[0]))
                verify(hits, expected[0])
                for hit in hits:
                    if hit.record != records[hit.record.record_id]:
                        raise AssertionError('Stale metadata returned after write')
                check_bounds(store)
                mutation_times.append(mutation_ms)
                search_times.append(elapsed)
            scenarios[name] = {'updates_per_sample': len(targets),
                               'mutation_batch': summarize(mutation_times),
                               'search': summarize(search_times)}
        record_id = next(iter(records))
        vector_times = []
        vector_search = []
        for iteration in range(samples):
            original = records[record_id]
            updated = SQLiteRecord.from_dict({'id': record_id, 'content': original.content,
                                             'metadata': json.loads(original.metadata_json),
                                             'embedding': queries[iteration % len(queries)]}, identity)
            _, elapsed = timed(lambda: store.upsert_records([updated]))
            records[record_id] = updated
            hits, scan_ms = timed(lambda: store.search_vectors(queries[0]))
            verify(hits, reference(records, queries[0]))
            check_bounds(store)
            vector_times.append(elapsed)
            vector_search.append(scan_ms)
        scenarios['vector_upsert'] = {'updates_per_sample': 1, 'mutation_batch': summarize(vector_times),
                                      'search': summarize(vector_search)}
        cache = check_bounds(store)
    with SQLiteRecordStore.open_read_only(root, identity) as observer:
        hits, reopened_ms = timed(lambda: observer.search_vectors(queries[0]))
        verify(hits, reference(records, queries[0]))
        check_bounds(observer)
    return {'records': count, 'dimensions': dimension, 'insert_ms': insertion_ms,
            'cold_search_ms': cold_ms, 'reopened_search_ms': reopened_ms,
            'cache': cache, 'scenarios': scenarios, 'exact_reference': 'passed'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-synthetic-benchmark', action='store_true', required=True)
    parser.add_argument('--sizes', type=int, nargs='+', default=[230, 512, 1000, 5000, 10000])
    parser.add_argument('--dimension', type=int, default=768)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args(argv)
    if (not 1 <= arguments.dimension <= MAX_DIMENSION or not 1 <= arguments.samples <= 100
            or any(not 1 <= size <= MAX_RECORDS for size in arguments.sizes)):
        parser.error('Invalid size, dimension or sample count')
    if arguments.output.exists():
        parser.error('Output must be a new file')
    results = {'status': 'running', 'python': platform.python_version(), 'platform': platform.system(),
               'machine': platform.machine(), 'sqlite': sqlite3.sqlite_version, 'seed': arguments.seed,
               'samples_per_scenario': arguments.samples, 'metric': 'cosine',
               'cache_limit_records': MAX_CACHE_RECORDS, 'cache_limit_bytes': MAX_CACHE_BYTES,
               'scope': 'Synthetic sequential record scans; no provider, MCP, dashboard or live data',
               'datasets': []}
    with arguments.output.open('x', encoding='ascii') as output:
        checkpoint(output, results)
        try:
            with tempfile.TemporaryDirectory(prefix='vecretary-synthetic-benchmark-') as temporary:
                for index, count in enumerate(arguments.sizes):
                    print(f'Benchmarking {count} synthetic records ({arguments.dimension} dimensions)', flush=True)
                    results['datasets'].append(benchmark(Path(temporary) / f'dataset-{index}', count,
                                                         arguments.dimension, arguments.samples, arguments.seed))
                    checkpoint(output, results)
            results['status'] = 'passed'
        except BaseException as error:
            results.update(status='failed', failure_type=type(error).__name__)
            raise
        finally:
            checkpoint(output, results)
    print('Synthetic benchmark and exact-reference checks passed', flush=True)


if __name__ == '__main__':
    main()
