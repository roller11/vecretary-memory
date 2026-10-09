# SPDX-License-Identifier: MPL-2.0
import importlib.util
import json
import os
from pathlib import Path
import tomllib

import pytest


spec = importlib.util.spec_from_file_location(
    'export_public_draft', Path(__file__).resolve().parents[1] / 'tools/export_public_draft.py',
)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'source'
    for name in set(export.FIXED) | set(export.SQLITE_OVERRIDES) | export.BINARY_FILES:
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        if name in export.BINARY_FILES:
            file.write_bytes(b'\x89PNG\r\n\x1a\n\x00\xffsynthetic-art')
        else:
            file.write_text('Synthetic reviewed content', encoding='utf-8')
    (root / 'tests').mkdir(exist_ok=True)
    return root


def test_export_excludes_private_operational_files(source, tmp_path):
    (source / 'docs/PRIVATE_STORE.json').write_text('Do not copy')
    (source / '.env').write_text('Do not copy')
    (source / 'instinct/chroma.sqlite3').write_bytes(b'Do not copy')
    (source / 'tests/private_browser.mjs').write_text('Do not copy')
    target = tmp_path / 'draft'
    report = export.prepare(source, target)
    assert report['status'] == 'DRAFT_ONLY_NOT_CLEARED_FOR_PUBLICATION'
    assert (target / 'README.md').read_text() == 'Synthetic reviewed content'
    assert not (target / 'docs/PRIVATE_STORE.json').exists()
    assert not (target / '.env').exists()
    assert not (target / 'instinct/chroma.sqlite3').exists()
    assert (target / 'tests/dashboard_capability_recovery.mjs').is_file()
    assert not (target / 'tests/private_browser.mjs').exists()
    for name in ('LICENSE', 'CONTRIBUTING.md', 'SECURITY.md', 'THIRD_PARTY_NOTICES.md'):
        assert (target / name).is_file()


def test_project_source_notices_survive_export(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = root / 'DRAFT_EXPORT.json'
    profile = json.loads(manifest.read_text())['profile'] if manifest.is_file() else 'legacy'
    target = tmp_path / 'reviewed-notices'
    report = export.prepare(root, target, profile=profile)
    source_files = [record for record in report['files'] if record['path'].endswith('.py')
                    or record['path'] in ('pyproject.toml', 'MANIFEST.in', '.github/workflows/ci.yml')]
    assert source_files
    for record in source_files:
        name = record['path']
        source_text = (target / name).read_text(encoding='utf-8')
        assert source_text.splitlines()[0] == '# SPDX-License-Identifier: MPL-2.0', name
        assert source_text == (root / record['source']).read_text(encoding='utf-8'), name


def test_existing_destination_is_never_overwritten(source, tmp_path):
    with pytest.raises(ValueError, match='new'):
        export.prepare(source, source)
    assert (source / 'pyproject.toml').read_text() == 'Synthetic reviewed content'


def test_machine_path_blocks_export_before_destination_creation(source, tmp_path):
    (source / 'instinct/private.py').write_text('/home/' + 'private-person/store')
    target = tmp_path / 'draft'
    with pytest.raises(ValueError, match='review required'):
        export.prepare(source, target)
    assert not target.exists()


def test_reviewed_funding_survives_export(source, tmp_path):
    (source / '.github/FUNDING.yml').write_text('github: roller11\n')
    target = tmp_path / 'draft'
    export.prepare(source, target)
    assert (target / '.github/FUNDING.yml').read_text() == 'github: roller11\n'


def test_migration_guide_survives_export(source, tmp_path):
    guide = source / 'docs/SQLITE_MIGRATION.md'
    guide.write_text('Offline migration and rollback boundaries', encoding='utf-8')
    target = tmp_path / 'draft'
    report = export.prepare(source, target)

    assert (target / 'docs/SQLITE_MIGRATION.md').read_bytes() == guide.read_bytes()
    assert any(entry['path'] == 'docs/SQLITE_MIGRATION.md' for entry in report['files'])


def test_supplied_logo_assets_export_byte_for_byte_with_hashes(source, tmp_path):
    target = tmp_path / 'draft'
    report = export.prepare(source, target)

    for name in export.BINARY_FILES:
        raw = (target / name).read_bytes()
        assert raw.startswith(b'\x89PNG\r\n\x1a\n')
        entry = next(item for item in report['files'] if item['path'] == name)
        assert entry['bytes'] == len(raw)
        assert entry['sha256'] == export.hashlib.sha256(raw).hexdigest()


def test_sqlite_release_profile_packages_logo_assets_and_sdist_files(source, tmp_path):
    (source / 'docs/sqlite-release/pyproject.toml').write_text(
        '[tool.setuptools.package-data]\ninstinct = ["starter_memories.json", "assets/*.png"]\n',
        encoding='utf-8',
    )
    (source / 'MANIFEST.in').write_text(
        'recursive-include instinct/assets *.png\n', encoding='utf-8',
    )
    target = tmp_path / 'draft'
    export.prepare(source, target)

    config = tomllib.loads((target / 'pyproject.toml').read_text(encoding='utf-8'))
    assert 'assets/*.png' in config['tool']['setuptools']['package-data']['instinct']
    assert 'recursive-include instinct/assets *.png' in (target / 'MANIFEST.in').read_text(encoding='utf-8')


def test_unreviewed_binary_asset_is_not_exported(source, tmp_path):
    (source / 'instinct/assets').mkdir(parents=True, exist_ok=True)
    (source / 'instinct/assets/private.png').write_bytes(b'Private')
    target = tmp_path / 'draft'
    export.prepare(source, target)
    assert not (target / 'instinct/assets/private.png').exists()


@pytest.mark.skipif(os.name == 'nt', reason='Symlink creation requires privileges on Windows')
def test_symlinked_source_is_refused(source, tmp_path):
    outside = tmp_path / 'private.py'
    outside.write_text('Private content')
    (source / 'instinct/linked.py').symlink_to(outside)
    with pytest.raises(ValueError, match='Unsafe source'):
        export.prepare(source, tmp_path / 'draft')
