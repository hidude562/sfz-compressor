"""dctjoin.gm: the General MIDI table is complete and every program resolves to a built folder."""
import os

import pytest

from dctjoin.gm import GM_PROGRAMS, build_bank

LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'dctjoin_library')


def test_table_is_complete_and_unique():
    assert len(GM_PROGRAMS) == 128
    names = [p[0] for p in GM_PROGRAMS]
    assert len(set(names)) == 128
    assert GM_PROGRAMS[0][0] == 'Acoustic Grand Piano' and GM_PROGRAMS[127][0] == 'Gunshot'
    assert GM_PROGRAMS[40][0] == 'Violin' and GM_PROGRAMS[56][0] == 'Trumpet' and GM_PROGRAMS[73][0] == 'Flute'


@pytest.mark.skipif(not os.path.isdir(LIB), reason='needs the built library')
def test_every_program_points_at_a_built_folder():
    missing = sorted({p[1] for p in GM_PROGRAMS if not os.path.isdir(os.path.join(LIB, p[1]))})
    assert not missing, missing


@pytest.mark.skipif(not os.path.isdir(LIB), reason='needs the built library')
def test_bank_builds_into_a_temp_dir(tmp_path):
    res = build_bank(LIB, str(tmp_path))
    assert res['programs'] == 128 and not res['missing']
    files = sorted(os.listdir(tmp_path))
    assert '000 Acoustic Grand Piano.sfz' in files and '127 Gunshot.sfz' in files and 'README.md' in files
    txt = open(tmp_path / '000 Acoustic Grand Piano.sfz').read()
    assert 'default_path=../Grand Piano/' in txt and 'ampeg_hold=' in txt and 'ampeg_release_oncc72=2' in txt
    txt = open(tmp_path / '056 Trumpet.sfz').read()
    assert 'default_path=../Trumpet/' in txt and 'loop_start=' in txt
