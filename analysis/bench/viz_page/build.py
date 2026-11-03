# Builds index.html from template.html with viz_data.json and Chart.js 4.4.1 embedded: the viewer does not reliably serve
# fetch() of side files or CDN scripts, so the page has no external dependency except fonts.
import json, pathlib
here = pathlib.Path(__file__).parent
data = json.dumps(json.load(open(here / 'viz_data.json')), separators=(',', ':')).replace('</', '<\\/')
chart = (here / 'chart.umd.min.js').read_text().replace('</', '<\\/')
html = (here / 'template.html').read_text().replace('/*DATA*/', data).replace('/*CHARTJS*/', chart)
(here / 'index.html').write_text(html)
print('index.html', (here / 'index.html').stat().st_size, 'bytes')
