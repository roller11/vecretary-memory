# SPDX-License-Identifier: MPL-2.0
"""Small correctness checks; deliberately no timing pass/fail thresholds."""

import json

import pytest

from tests.run_sqlite_benchmark import benchmark, main, summarize
from tests import run_sqlite_benchmark as helper


@pytest.mark.parametrize('count', [24, 513])
def test_small_benchmark_checks_search_mutations_and_reopen(tmp_path, count):
    result = benchmark(tmp_path / 'synthetic', count, 8, 2, 1729)
    assert result['exact_reference'] == 'passed'
    assert result['cache']['records'] == min(count, 512)
    assert result['scenarios']['corpus10pct_metadata']['updates_per_sample'] == max(1, count // 10)
    assert result['scenarios']['vector_upsert']['search']['samples'] == 2


def test_summary_only_reports_p95_with_twenty_samples():
    assert summarize([1, 2, 3])['p95_ms'] is None
    assert summarize(list(range(1, 21)))['p95_ms'] == 19


def test_main_requires_opt_in_and_preserves_existing_output(tmp_path):
    output = tmp_path / 'results.json'
    with pytest.raises(SystemExit):
        main(['--output', str(output)])
    assert not output.exists()
    arguments = ['--confirm-synthetic-benchmark', '--sizes', '12', '--dimension', '4',
                 '--samples', '1', '--output', str(output)]
    main(arguments)
    assert json.loads(output.read_text())['status'] == 'passed'
    content = output.read_bytes()
    with pytest.raises(SystemExit):
        main(arguments)
    assert output.read_bytes() == content


@pytest.mark.parametrize('option,value', [('--sizes', '0'), ('--sizes', '10001'),
                                        ('--dimension', '0'), ('--samples', '101')])
def test_invalid_bounds_create_no_output(tmp_path, option, value):
    output = tmp_path / 'results.json'
    with pytest.raises(SystemExit):
        main(['--confirm-synthetic-benchmark', '--output', str(output), option, value])
    assert not output.exists()


def test_failed_benchmark_preserves_sanitized_checkpoint(tmp_path, monkeypatch):
    def fail(*arguments):
        raise RuntimeError('Sensitive diagnostic not for public output')
    monkeypatch.setattr(helper, 'benchmark', fail)
    output = tmp_path / 'failed.json'
    with pytest.raises(RuntimeError):
        main(['--confirm-synthetic-benchmark', '--sizes', '12', '--output', str(output)])
    report = json.loads(output.read_text())
    assert report['status'] == 'failed'
    assert report['failure_type'] == 'RuntimeError'
    assert 'Sensitive' not in output.read_text()
