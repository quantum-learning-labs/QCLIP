# Read-only: count every job on the IonQ account by backend (no submissions)
import os, json, urllib.request, collections
key = open(os.path.expanduser('~/.ionq_key')).read().strip()
cnt, n, nxt = collections.Counter(), 0, None
for _ in range(500):
    url = 'https://api.ionq.co/v0.4/jobs?limit=100' + (f'&next={nxt}' if nxt else '')
    d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={'Authorization': 'apiKey ' + key}), timeout=60))
    jobs = d.get('jobs', d if isinstance(d, list) else [])
    for j in jobs:
        cnt[(j.get('backend'), j.get('type'))] += 1; n += 1
    nxt = d.get('next') if isinstance(d, dict) else None
    if not nxt or not jobs: break
print('jobs', n); [print(k, v) for k, v in cnt.most_common()]
