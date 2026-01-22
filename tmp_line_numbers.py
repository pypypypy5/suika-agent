from pathlib import Path
lines = Path("envs/suika_wrapper.py").read_text(encoding="utf-8").splitlines()
for i, line in enumerate(lines, start=1):
    if 'self._sync_observation_space()' in line:
        print('sync call line', i)
    if 'def _sync_observation_space' in line:
        print('sync def line', i)
    if 'def _process_observation' in line:
        print('process def line', i)
    if 'def _format_score' in line:
        print('format score line', i)
