"""Style metrics for the report draft. Usage: python style_check.py final_report.tex

Counts the tells listed in STYLE.md so the numbers are checkable rather than felt.
"""
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'final_report.tex'
t = open(path, encoding='utf-8').read()
body = t.split('begin{abstract}')[1].split('begin{thebibliography}')[0]

# Drop TODO placeholder blocks: they are scaffolding, not prose.
out, i = [], 0
while True:
    a = body.find('\\begin{TODO}', i)
    if a == -1:
        out.append(body[i:])
        break
    out.append(body[i:a])
    b = body.find('\\end{TODO}', a)
    i = len(body) if b == -1 else b + len('\\end{TODO}')
body = ''.join(out)

prose = re.sub(r'\\[a-zA-Z]+\*?', ' ', body)
words = len([x for x in prose.split() if any(c.isalpha() for c in x)])
brit = sorted(set(re.findall(r'\b\w+(?:ised|isation|ising)\b', t)))

BASELINE = {  # draft 1, before the style pass
    'em-dashes (---)': 62,
    'semicolons': 29,
    'emph{} italics': 64,
    '"rather than"': 12,
    'prose words (body)': 4638,
}
now = {
    'em-dashes (---)': body.count('---'),
    'semicolons': body.count(';'),
    'emph{} italics': body.count('emph{'),
    '"rather than"': body.lower().count('rather than'),
    'prose words (body)': words,
}

print('%-22s %8s %8s' % ('metric', 'draft 1', 'now'))
for k, v in BASELINE.items():
    print('%-22s %8d %8d' % (k, v, now[k]))
print('%-22s %8s %8s' % ('British spellings', 'many', brit or 'none'))
