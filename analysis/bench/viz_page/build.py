# Builds index.html from template.html with viz_data.json embedded (the viewer does not reliably serve fetch() of side files).
import json, pathlib
here = pathlib.Path(__file__).parent
data = json.dumps(json.load(open(here / 'viz_data.json')), separators=(',', ':')).replace('</', '<\\/')
(here / 'index.html').write_text((here / 'template.html').read_text().replace('/*DATA*/', data))
print('index.html', (here / 'index.html').stat().st_size, 'bytes')
