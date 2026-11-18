# Store document 'paper' for the live page: LaTeX sources, the vector PDF and a zip of all three (base64), with provenance.
import base64, io, json, zipfile, datetime, sys
commit = sys.argv[1] if len(sys.argv) > 1 else ''
ft, mt, pdf = open('efig_figure.tex').read(), open('efig_methods.tex').read(), open('efig_log2.pdf', 'rb').read()
buf = io.BytesIO()
with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
    z.writestr('efig_figure.tex', ft); z.writestr('efig_methods.tex', mt); z.writestr('efig_log2.pdf', pdf)
doc = dict(figure_tex=ft, methods_tex=mt, pdf_b64=base64.b64encode(pdf).decode(), zip_b64=base64.b64encode(buf.getvalue()).decode(),
           pdf_name='efig_log2.pdf', zip_name='efig_paper_files.zip', commit=commit,
           updated=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC'))
json.dump(doc, open('efig_docs/paper.json', 'w'))
print({k: len(v) for k, v in doc.items() if isinstance(v, str)})
