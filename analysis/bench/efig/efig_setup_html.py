# Render efig_setup.md into the page's setup section (between the paper-files section and the scripts).
import re, html
md = open('efig_setup.md').read().split('\n')
def inline(t):
    t = html.escape(t, quote=False)
    t = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'<a href="\2" target="_blank" rel="noopener">\1</a>', t)
    t = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', t)
    return re.sub(r'`([^`]+)`', r'<code>\1</code>', t)
out, lvl = [], 0
def close_to(d):
    global lvl
    while lvl > d: out.append('</li></ul>'); lvl -= 1
for line in md:
    if line.startswith('## '): close_to(0); out.append(f'<h2 id="setup-h">{inline(line[3:])}</h2>'); continue
    if line.startswith('### '): close_to(0); out.append(f'<h3>{inline(line[4:])}</h3>'); continue
    m = re.match(r'^( *)- (.*)$', line)
    if m:
        d = len(m.group(1)) // 2 + 1
        if d > lvl:
            while lvl < d: out.append('<ul><li>'); lvl += 1
            out.append(inline(m.group(2)))
        else: close_to(d); out.append('</li><li>' + inline(m.group(2)))
        continue
    if line.strip(): close_to(0); out.append(f'<p class="cap">{inline(line.strip())}</p>')
close_to(0)
sec = '  <section class="fig methods setup" aria-labelledby="setup-h">\n' + '\n'.join(out) + '\n  </section>\n'
p = 'efig_page.html'; s = open(p).read()
s, n = re.subn(r'  <section class="fig methods setup".*?</section>\n', lambda _: sec, s, flags=re.S)
assert n == 1
open(p, 'w').write(s); print('setup section rendered')
