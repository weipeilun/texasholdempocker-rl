"""Controller of the Splendor auto-research loop (see auto_research/README.md).

    python auto_research/loop.py launch R001            start the round's training (detached), write launch.json
    python auto_research/loop.py watch R001 [--enforce-budget] [--json]
                                                         summarise metrics.jsonl / train.log -> watch.json
    python auto_research/loop.py stop R001               SIGINT to the round's process group (it saves and exits)
    python auto_research/loop.py status                  state.json and the current round's watch summary
    python auto_research/loop.py render [--out PATH]     the notebook page (splendor-auto.html) from the ledger

Run from the repository root with the training interpreter. The ledger is plain files under auto_research/:
state.json, goal.md, priors.md, program.md, rounds/<R>/{spec.yaml, plan.md, launch.json, watch.json,
analysis.md, decision.json}. Every number shown on the page comes from these files and the experiments'
metrics.jsonl.
"""
import argparse
import datetime
import html
import json
import os
import re
import signal
import subprocess
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, 'auto_research')
TZ = datetime.timezone(datetime.timedelta(hours=8))


def now():
    return datetime.datetime.now(TZ).isoformat(timespec='seconds')


def round_dir(name):
    return os.path.join(LEDGER, 'rounds', name)


def load_json(path, default=None):
    try:
        with open(path, encoding='UTF-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path, value):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='UTF-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=1, default=float)
    os.replace(tmp, path)


def load_spec(name):
    with open(os.path.join(round_dir(name), 'spec.yaml'), encoding='UTF-8') as f:
        return yaml.safe_load(f)


def read_text(path):
    try:
        with open(path, encoding='UTF-8') as f:
            return f.read()
    except FileNotFoundError:
        return ''


def experiment_dir_of(spec):
    with open(os.path.join(ROOT, spec['config']), encoding='UTF-8') as f:
        config = yaml.safe_load(f)
    return os.path.join(ROOT, config.get('experiment_root', 'experiments'), config['experiment_name'])


def read_metrics(experiment_dir):
    records = list()
    try:
        with open(os.path.join(experiment_dir, 'metrics.jsonl'), encoding='UTF-8') as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        pass
    return records


def training_processes():
    out = subprocess.run(['pgrep', '-af', 'train_splendor.py'], capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if 'pgrep' not in line]


def process_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:   # a zombie is not running
        with open(f'/proc/{pid}/stat') as f:
            return f.read().split(')')[-1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


# ============================================================================ launch / stop
def cmd_launch(args):
    spec = load_spec(args.round)
    directory = round_dir(args.round)
    if os.path.exists(os.path.join(directory, 'launch.json')) and not args.force:
        raise SystemExit(f'{args.round} was launched already (launch.json); --force to launch again')
    running = training_processes()
    if running and not args.force:
        raise SystemExit('another training is running:\n' + '\n'.join(running))
    command = [sys.executable, 'train_splendor.py', '--mode', 'single', '--default-config', spec['config']]
    if spec.get('resume'):
        command = [sys.executable, 'train_splendor.py', '--resume', spec['resume']]
    stdout = open(os.path.join(directory, 'stdout.log'), 'ab')
    process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               start_new_session=True)
    git = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    launch = {'round': args.round, 'launched_at': now(), 'pid': process.pid, 'pgid': os.getpgid(process.pid),
              'command': ' '.join(command), 'cwd': ROOT, 'experiment_dir': os.path.relpath(experiment_dir_of(spec), ROOT),
              'git_head': git, 'git_dirty_files': len(dirty.splitlines()) if dirty else 0,
              'budget_hours': spec.get('budget_hours')}
    save_json(os.path.join(directory, 'launch.json'), launch)
    state = load_json(os.path.join(LEDGER, 'state.json'), {})
    state.update(status='running', phase='watch', current_round=args.round, updated=now())
    save_json(os.path.join(LEDGER, 'state.json'), state)
    print(json.dumps(launch, ensure_ascii=False, indent=1))


def cmd_stop(args):
    launch = load_json(os.path.join(round_dir(args.round), 'launch.json'))
    if not launch:
        raise SystemExit(f'{args.round} has no launch.json')
    if not process_alive(launch['pid']):
        print(f'{args.round}: process {launch["pid"]} is not running')
        return
    # SIGINT to the training process only: it stops its game pool itself and saves a checkpoint. A SIGINT to
    # the whole process group also hits the pool workers, and the main loop then reports their exit as an
    # error (R003). The group gets it only if the main process is still alive after half the timeout.
    os.kill(launch['pid'], signal.SIGINT)
    deadline = time.time() + args.timeout
    group_signalled = False
    while time.time() < deadline and process_alive(launch['pid']):
        if not group_signalled and time.time() > deadline - args.timeout / 2:
            os.killpg(launch['pgid'], signal.SIGINT)
            group_signalled = True
        time.sleep(2)
    print(f'{args.round}: {"stopped" if not process_alive(launch["pid"]) else "still running after SIGINT"}')


# ============================================================================ watch
def _last(records, n, key):
    values = [r[key] for r in records if r.get(key) is not None][-n:]
    return float(sum(values) / len(values)) if values else None


def summarize(spec, launch, records, log_tail, log_mtime=None):
    iterations = [r for r in records if r['kind'] == 'iteration']
    evals = [r for r in records if r['kind'] == 'eval']
    benches = [r for r in records if r['kind'] == 'bench']
    summary = {'updated': now(), 'iterations': len(iterations)}
    if launch:
        started = datetime.datetime.fromisoformat(launch['launched_at'])
        summary['running'] = process_alive(launch['pid'])
        # a finished run ends at its log's last write, not now
        end = datetime.datetime.now(TZ) if summary['running'] or log_mtime is None else datetime.datetime.fromtimestamp(log_mtime, TZ)
        summary['elapsed_hours'] = (end - started).total_seconds() / 3600.
    if iterations:
        last = iterations[-1]
        summary.update(train_step_num=last['train_step_num'], lr=last.get('lr'),
                       games_per_second=_last(iterations, 5, 'games_per_second'),
                       avg_turns=_last(iterations, 5, 'avg_turns'), replay_buffer=last.get('replay_buffer'),
                       policy_loss=_last(iterations, 5, 'policy_loss'), value_loss=_last(iterations, 5, 'value_loss'))
        reasons = dict()
        for r in iterations[-5:]:
            for k, v in r['end_reasons'].items():
                reasons[k] = reasons.get(k, 0) + v
        summary['end_reasons_last5'] = reasons
        held = [r['held_out'] for r in iterations if r.get('held_out')]
        if held:
            def avg(k, part):
                values = [h[k] for h in part]
                return float(sum(values) / len(values))
            first, recent = held[:5], held[-5:]
            summary['held_out'] = {k: {'first5': avg(k, first), 'last5': avg(k, recent)}
                                   for k in ('value_ce', 'value_mse', 'value_mse_constant', 'winner_acc', 'policy_ce',
                                             'target_entropy')}
        # TD(λ) value targets (R003): the same metrics against the training target
        held = [r['held_out_target'] for r in iterations if r.get('held_out_target')]
        if held:
            first, recent = held[:5], held[-5:]
            summary['held_out_target'] = {k: {'first5': avg(k, first), 'last5': avg(k, recent)}
                                          for k in ('value_ce', 'value_mse', 'value_mse_constant')}
    if evals:
        summary['eval'] = [{'round': e['round'], 'iteration': e['iteration'], 'win_rate': e['win_rate'],
                            'completion': e['completion'], 'promoted': e['promoted']} for e in evals]
        summary['promotions'] = sum(e['promoted'] for e in evals)
    if benches:
        by_opponent = dict()
        for b in benches:
            for opponent, m in b['opponents'].items():
                by_opponent.setdefault(opponent, list()).append(
                    {'round': b['round'], 'iteration': b['iteration'], 'train_step_num': b.get('train_step_num'),
                     'win_share': m['win_share'], 'win_rate': m['win_rate'], 'games': m['games'],
                     'mean_rounds_when_winning': m.get('mean_rounds_when_winning')})
        summary['bench'] = by_opponent
        summary['bench_last3'] = {o: float(sum(x['win_share'] for x in v[-3:]) / len(v[-3:])) for o, v in by_opponent.items()}
        summary['bench_best'] = {o: max(x['win_share'] for x in v) for o, v in by_opponent.items()}
    if 'Finish loop' in log_tail:
        summary['end'] = 'finished'
    elif 'training stopped by an error' in log_tail or 'Traceback' in log_tail:
        summary['end'] = 'error'
    elif 'interrupted' in log_tail:
        summary['end'] = 'interrupted'
    target = spec.get('target', {})
    if target and summary.get('bench_last3'):
        opponent, threshold = target.get('opponent'), target.get('win_share')
        summary['target_reached'] = bool(opponent in summary['bench_last3'] and summary['bench_last3'][opponent] >= threshold)
    return summary


def cmd_watch(args):
    spec = load_spec(args.round)
    directory = round_dir(args.round)
    launch = load_json(os.path.join(directory, 'launch.json'))
    experiment_dir = os.path.join(ROOT, launch['experiment_dir']) if launch else experiment_dir_of(spec)
    log_path = os.path.join(experiment_dir, 'train.log')
    log = read_text(log_path)
    summary = summarize(spec, launch, read_metrics(experiment_dir), log[-20000:],
                        os.path.getmtime(log_path) if os.path.exists(log_path) else None)
    errors = [line for line in log.splitlines()[-3000:] if ' ERROR ' in line or 'Traceback' in line]
    summary['recent_errors'] = errors[-5:]
    budget = spec.get('budget_hours')
    if args.enforce_budget and budget and summary.get('running') and summary.get('elapsed_hours', 0) > budget:
        os.killpg(launch['pgid'], signal.SIGINT)
        summary['budget_stop'] = now()
    save_json(os.path.join(directory, 'watch.json'), summary)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return
    print(f'{args.round}: running={summary.get("running")} end={summary.get("end")} elapsed={summary.get("elapsed_hours", 0):.2f}h '
          f'iterations={summary["iterations"]} steps={summary.get("train_step_num")} lr={summary.get("lr")} '
          f'games/s={summary.get("games_per_second")} turns={summary.get("avg_turns")}')
    if summary.get('held_out'):
        print('held-out', {k: (round(v['first5'], 4), round(v['last5'], 4)) for k, v in summary['held_out'].items()})
    for e in summary.get('eval', [])[-6:]:
        print('eval', e)
    for opponent, rows in summary.get('bench', {}).items():
        print('bench', opponent, [(r['iteration'], round(r['win_share'], 3)) for r in rows])
    for line in summary['recent_errors']:
        print('ERROR', line[:300])


def cmd_status(args):
    state = load_json(os.path.join(LEDGER, 'state.json'), {})
    print(json.dumps(state, ensure_ascii=False, indent=1))
    if state.get('current_round'):
        args.round, args.enforce_budget, args.json = state['current_round'], False, False
        cmd_watch(args)


# ============================================================================ render
def _join(parts):
    """join wrapped source lines: a space only between two ASCII characters (none between CJK text)"""
    out = ''
    for part in (p.strip() for p in parts):
        if out and part and out[-1].isascii() and part[0].isascii():
            out += ' '
        out += part
    return out


def md_to_html(text):
    """small Markdown subset: headings, paragraphs, lists, tables, fenced code, **bold**, `code`, links"""
    def inline(s):
        s = html.escape(s, quote=False)
        s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
        s = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', s)
        s = re.sub(r'\[([^\]]+)\]\((https?://[^)\s]+)\)', r'<a href="\2">\1</a>', s)
        return s
    out, lines, i = list(), text.splitlines(), 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('```'):
            j = i + 1
            while j < len(lines) and not lines[j].startswith('```'):
                j += 1
            out.append('<pre>' + html.escape('\n'.join(lines[i + 1:j])) + '</pre>')
            i = j + 1
            continue
        m = re.match(r'^(#{1,6})\s+(.*)', line)
        if m:
            level = min(6, len(m.group(1)) + 2)       # '#' of a round document -> h3
            out.append(f'<h{level}>{inline(m.group(2))}</h{level}>')
            i += 1
            continue
        if line.startswith('|') and i + 1 < len(lines) and re.match(r'^\|[\s:|-]+\|$', lines[i + 1].strip()):
            header = [c.strip() for c in line.strip().strip('|').split('|')]
            rows, j = list(), i + 2
            while j < len(lines) and lines[j].startswith('|'):
                rows.append([c.strip() for c in lines[j].strip().strip('|').split('|')])
                j += 1
            out.append('<table><thead><tr>' + ''.join(f'<th>{inline(c)}</th>' for c in header) + '</tr></thead><tbody>'
                       + ''.join('<tr>' + ''.join(f'<td>{inline(c)}</td>' for c in r) + '</tr>' for r in rows)
                       + '</tbody></table>')
            i = j
            continue
        if re.match(r'^\s*([-*]|\d+\.)\s+', line):
            ordered = bool(re.match(r'^\s*\d+\.', line))
            items, j = list(), i
            while j < len(lines) and (re.match(r'^\s*([-*]|\d+\.)\s+', lines[j]) or (lines[j].startswith('  ') and lines[j].strip())):
                if re.match(r'^\s*([-*]|\d+\.)\s+', lines[j]):
                    items.append(re.sub(r'^\s*([-*]|\d+\.)\s+', '', lines[j]))
                else:
                    items[-1] = _join([items[-1], lines[j]])
                j += 1
            tag = 'ol' if ordered else 'ul'
            out.append(f'<{tag}>' + ''.join(f'<li>{inline(x)}</li>' for x in items) + f'</{tag}>')
            i = j
            continue
        if not line.strip():
            i += 1
            continue
        para, j = [line], i + 1
        while j < len(lines) and lines[j].strip() and not re.match(r'^(#|```|\||\s*([-*]|\d+\.)\s)', lines[j]):
            para.append(lines[j])
            j += 1
        out.append('<p>' + inline(_join(para)) + '</p>')
        i = j
    return '\n'.join(out)


def svg_lines(series, title, y_label, y_min=0., y_max=1., refs=()):
    """series: [(name, colour var, [(x, y), ...])]; refs: [(y, label)] dashed reference lines"""
    width, height, left, right, top, bottom = 760, 260, 52, 150, 26, 36
    xs = [x for _, _, points in series for x, _ in points]
    if not xs:
        return ''
    x0, x1 = min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1

    def px(x):
        return left + (x - x0) / (x1 - x0) * (width - left - right)

    def py(y):
        return top + (1 - (y - y_min) / (y_max - y_min)) * (height - top - bottom)
    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{html.escape(title)}">',
             f'<text x="{left}" y="16" class="sv-tb">{html.escape(title)}</text>']
    for k in range(6):
        y = y_min + (y_max - y_min) * k / 5
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{py(y):.1f}" y2="{py(y):.1f}" stroke="var(--grid)"/>'
                     f'<text x="{left - 6}" y="{py(y) + 4:.1f}" text-anchor="end" class="sv-t3">{y:.2f}</text>')
    for y, label in refs:
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{py(y):.1f}" y2="{py(y):.1f}" class="sv-dash"/>'
                     f'<text x="{width - right + 4}" y="{py(y) + 4:.1f}" class="sv-t3">{html.escape(label)}</text>')
    for k, (name, colour, points) in enumerate(series):
        if not points:
            continue
        d = ' '.join(f'{"M" if j == 0 else "L"}{px(x):.1f},{py(y):.1f}' for j, (x, y) in enumerate(points))
        parts.append(f'<path d="{d}" fill="none" stroke="var({colour})" stroke-width="2"/>')
        parts += [f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="2.6" fill="var({colour})"><title>{html.escape(name)}: '
                  f'{x} → {y:.3f}</title></circle>' for x, y in points]
        parts.append(f'<text x="{width - right + 4}" y="{top + 14 + 16 * k}" class="sv-t2" fill="var({colour})">'
                     f'● {html.escape(name)}</text>')
    parts.append(f'<text x="{(left + width - right) / 2:.0f}" y="{height - 6}" text-anchor="middle" class="sv-t3">{html.escape(y_label)}</text>')
    parts.append('</svg>')
    return '<div class="svgfig">' + ''.join(parts) + '</div>'


CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface-1:#fcfcfb;--surface-2:#f3f3ef;--surface-3:#eceae4;--ink-1:#0b0b0b;
--ink-2:#52514e;--ink-3:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,0.10);--s1:#2a78d6;--s2:#eb6834;
--s3:#1baf7a;--emph:#1c5cab;--good:#0ca30c;--warn:#fab219;--crit:#d03b3b;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
--sans:system-ui,-apple-system,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei","Noto Sans CJK SC",sans-serif}
@media (prefers-color-scheme: dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--page:#0d0d0d;--surface-1:#1a1a19;
--surface-2:#222221;--surface-3:#2a2a28;--ink-1:#fff;--ink-2:#c3c2b7;--ink-3:#898781;--grid:#2c2c2a;--axis:#383835;
--border:rgba(255,255,255,0.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--emph:#7fb2f2}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface-1:#1a1a19;--surface-2:#222221;--surface-3:#2a2a28;--ink-1:#fff;
--ink-2:#c3c2b7;--ink-3:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,0.10);--s1:#3987e5;--s2:#d95926;
--s3:#199e70;--emph:#7fb2f2}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:64px}
body{margin:0;background:var(--page);color:var(--ink-1);font-family:var(--sans);font-size:15px;line-height:1.75;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:0 24px 96px}
header.top{position:sticky;top:0;z-index:20;background:color-mix(in srgb,var(--page) 88%,transparent);backdrop-filter:blur(10px);border-bottom:1px solid var(--border)}
.top-in{max-width:1080px;margin:0 auto;padding:10px 24px;display:flex;align-items:center;gap:16px}
.brand{font-weight:650;font-size:14px;white-space:nowrap}.brand a{color:inherit;text-decoration:none}.brand span{color:var(--ink-3);font-weight:450}
nav.toc{display:flex;gap:2px;flex:1;overflow-x:auto;scrollbar-width:none}nav.toc::-webkit-scrollbar{display:none}
nav.toc a{color:var(--ink-2);text-decoration:none;font-size:12.5px;padding:4px 8px;border-radius:6px;white-space:nowrap}
nav.toc a:hover{background:var(--surface-2);color:var(--ink-1)}
.tbtn{border:1px solid var(--border);background:var(--surface-1);color:var(--ink-2);border-radius:8px;padding:5px 10px;font:inherit;font-size:12px;cursor:pointer;white-space:nowrap}
h1{font-size:30px;line-height:1.3;margin:40px 0 8px;letter-spacing:-.01em;font-weight:680}
.sub{color:var(--ink-2);font-size:15px;margin:0 0 4px}.meta{color:var(--ink-3);font-size:12.5px;margin:0}
h2{font-size:20px;margin:56px 0 4px;font-weight:660;scroll-margin-top:64px}h3{font-size:15.5px;margin:28px 0 6px;font-weight:640}
h4{font-size:14px;margin:18px 0 4px;font-weight:640}h5{font-size:13.5px;margin:14px 0 4px;font-weight:640}
p{margin:10px 0}ul,ol{margin:10px 0;padding-left:22px}li{margin:5px 0}strong{font-weight:640}
code{font-family:var(--mono);font-size:.88em;background:var(--surface-2);padding:1px 5px;border-radius:4px;border:1px solid var(--border)}
a{color:var(--s1)}
.note{border-left:2px solid var(--s1);background:var(--surface-2);border-radius:0 10px 10px 0;padding:12px 16px;margin:16px 0;font-size:14px}
.note.warn{border-left-color:var(--warn)}.note.key{border-left-color:var(--s3)}
.svgfig{background:var(--surface-1);border:1px solid var(--border);border-radius:14px;padding:16px;margin:20px 0;overflow-x:auto}
.svgfig svg{display:block;min-width:700px;width:100%;height:auto}svg text{font-family:var(--sans)}
.sv-t2{font-size:11px}.sv-t3{fill:var(--ink-3);font-size:10.5px}.sv-tb{fill:var(--ink-1);font-size:12.5px;font-weight:640}
.sv-dash{stroke:var(--ink-3);stroke-width:1;stroke-dasharray:4 3;fill:none}
.fig{margin:18px 0 6px;background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:12px 14px;overflow-x:auto}
.map{width:100%;border-collapse:collapse;font-size:13.5px;margin:8px 0 18px}
.map th,.map td{border-bottom:1px solid var(--border);padding:8px 8px;text-align:left;vertical-align:top}
.map th{color:var(--ink-2);font-weight:600;font-size:12.5px}
.md table{border-collapse:collapse;font-size:13.5px;margin:8px 0 14px}.md th,.md td{border-bottom:1px solid var(--border);padding:6px 8px;text-align:left;vertical-align:top}
.md pre,pre{background:var(--surface-2);border:1px solid var(--border);border-radius:8px;padding:10px 12px;overflow-x:auto;font-size:12.5px}
.chip{display:inline-block;border-radius:999px;padding:1px 9px;font-size:12px;border:1px solid var(--border)}
.chip.supported{background:#e6f4ec;color:#1a6b3c}.chip.refuted{background:#fbe9e5;color:#8f2f1c}
.chip.inconclusive{background:#f3efe0;color:#6b5b1c}.chip.none{color:var(--ink-3)}.chip.running{background:#e7effa;color:#1c4f8f}
.round{border:1px solid var(--border);border-radius:12px;padding:14px 18px;margin:14px 0;background:var(--surface-1)}
.round summary{cursor:pointer;font-weight:600}
.sec{border-left:3px solid var(--axis);padding:6px 12px;margin:10px 0;background:var(--surface-2);border-radius:0 8px 8px 0}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin:14px 0}
.kv div{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:10px 12px}
.kv .l{font-size:12px;color:var(--ink-3)}.kv .v{font-size:20px;font-weight:650;overflow-wrap:anywhere}
"""


def _fmt(x, digits=3):
    if x is None:
        return '–'
    if isinstance(x, float):
        return f'{x:.{digits}f}'
    return html.escape(str(x))


def render_round(name):
    directory = round_dir(name)
    spec = load_spec(name)
    launch = load_json(os.path.join(directory, 'launch.json'))
    watch = load_json(os.path.join(directory, 'watch.json'), {})
    decision = load_json(os.path.join(directory, 'decision.json'))
    verdict = (decision or {}).get('verdict') or ('running' if launch else 'none')
    period = ''
    if launch:
        period = f'{launch["launched_at"]} → {(decision or {}).get("decided_at", "进行中")}'
    gpu_hours = watch.get('elapsed_hours')
    parts = [f'<details class="round" id="{name.lower()}" {"open" if verdict == "running" else ""}>',
             f'<summary>{name} · {html.escape(spec["name"])} · <span class="chip {verdict}">{verdict}</span> '
             f'<span class="meta">{html.escape(spec.get("kind", ""))} · {period}'
             + (f' · GPU {gpu_hours:.1f} h' if gpu_hours else '') + '</span></summary>',
             f'<div class="sec"><b>假设</b> {html.escape(spec["hypothesis"].strip())}'
             f'<br><b>判定问题</b> {html.escape(spec.get("question", "").strip())}'
             f'<br><b>实验路径</b> <code>{html.escape(launch["experiment_dir"] if launch else spec.get("experiment_dir", ""))}</code>'
             f' · 配置 <code>{html.escape(spec["config"])}</code></div>']
    plan = read_text(os.path.join(directory, 'plan.md'))
    if plan:
        parts.append('<h3>计划</h3><div class="md">' + md_to_html(plan) + '</div>')
    parts.append('<h3>数据</h3>' + render_round_data(spec, watch))
    analysis = read_text(os.path.join(directory, 'analysis.md'))
    if analysis:
        parts.append('<h3>分析与结论</h3><div class="md">' + md_to_html(analysis) + '</div>')
    if decision:
        parts.append('<h3>决策</h3><pre>' + html.escape(json.dumps(decision, ensure_ascii=False, indent=1)) + '</pre>')
    if launch:
        parts.append(f'<p class="meta">规格: <code>auto_research/rounds/{name}/spec.yaml</code> · 启动: '
                     f'<code>{html.escape(launch["command"])}</code> · 代码 {html.escape(launch.get("git_head", ""))}'
                     f'（未提交改动 {launch.get("git_dirty_files", 0)} 个文件）</p>')
    parts.append('</details>')
    return '\n'.join(parts), spec, launch, watch, decision, verdict


def render_round_data(spec, watch):
    if not watch:
        return '<p class="meta">尚未开始。</p>'
    rows = [('状态', f'{"运行中" if watch.get("running") else "已结束"}' + (f'（{watch["end"]}）' if watch.get('end') else '')),
            ('已用时间', f'{watch.get("elapsed_hours", 0):.2f} h'), ('iteration', watch.get('iterations')),
            ('训练步数', watch.get('train_step_num')), ('当前学习率', _fmt(watch.get('lr'), 6) if watch.get('lr') else '–'),
            ('自对弈 games/s（近 5 轮）', _fmt(watch.get('games_per_second'), 2)),
            ('平均回合（turn，近 5 轮）', _fmt(watch.get('avg_turns'), 1)),
            ('评估晋升次数', watch.get('promotions', 0))]
    out = ['<div class="fig"><table class="map"><tbody>' + ''.join(
        f'<tr><td>{html.escape(k)}</td><td>{v if isinstance(v, str) else _fmt(v)}</td></tr>' for k, v in rows) + '</tbody></table></div>']
    bench = watch.get('bench', {})
    if bench:
        colours = ['--s1', '--s2', '--s3']
        series = [(o, colours[k % 3], [(r['iteration'], r['win_share']) for r in v]) for k, (o, v) in enumerate(sorted(bench.items()))]
        target = spec.get('target', {})
        refs = [(0.25, '势均力敌 0.25')] + ([(target['win_share'], f'目标 {target["win_share"]}')] if target else [])
        out.append(svg_lines(series, '基准：模型 1 座 vs 3 个规则机器人的胜率份额', 'iteration', refs=refs))
    evals = watch.get('eval', [])
    if evals:
        out.append(svg_lines([('新模型 vs best', '--s1', [(e['iteration'], e['win_rate']) for e in evals])],
                             '评估：新模型（2 座）vs best（2 座）胜率', 'iteration', refs=[(0.5, '持平 0.5'), (0.55, '晋升线')]))
    held = watch.get('held_out')
    if held:
        out.append('<div class="fig"><table class="map"><thead><tr><th>held-out（新对局、训练前）</th><th>前 5 轮</th><th>近 5 轮</th></tr></thead><tbody>'
                   + ''.join(f'<tr><td>{k}</td><td>{v["first5"]:.4f}</td><td>{v["last5"]:.4f}</td></tr>' for k, v in held.items())
                   + ''.join(f'<tr><td>{k}（对 TD(λ) 目标）</td><td>{v["first5"]:.4f}</td><td>{v["last5"]:.4f}</td></tr>'
                             for k, v in (watch.get('held_out_target') or {}).items())
                   + '</tbody></table></div>')
    return '\n'.join(out)


def cmd_render(args):
    state = load_json(os.path.join(LEDGER, 'state.json'), {})
    names = sorted(d for d in os.listdir(os.path.join(LEDGER, 'rounds')) if re.match(r'^R\d+$', d))
    rendered = [render_round(n) for n in names]
    counts = {v: sum(1 for r in rendered if r[5] == v) for v in ('supported', 'refuted', 'inconclusive')}
    gpu = sum((r[3] or {}).get('elapsed_hours') or 0 for r in rendered)
    best = state.get('best_model') or {}
    table = ['<div class="fig"><table class="map"><thead><tr><th>轮</th><th>名称</th><th>类型</th><th>假设</th><th>判定</th>'
             '<th>关键指标</th><th>实验路径</th><th>GPU 时长</th></tr></thead><tbody>']
    for (block, spec, launch, watch, decision, verdict), name in zip(rendered, names):
        key = ', '.join(f'{o.replace("bot:", "")} {v:.3f}' for o, v in (watch or {}).get('bench_last3', {}).items())
        table.append(f'<tr><td><a href="#{name.lower()}">{name}</a></td><td>{html.escape(spec["name"])}</td>'
                     f'<td>{html.escape(spec.get("kind", ""))}</td><td>{html.escape(spec["hypothesis"].strip()[:220])}</td>'
                     f'<td><span class="chip {verdict}">{verdict}</span></td><td>{html.escape(key) or "–"}</td>'
                     f'<td><code>{html.escape(launch["experiment_dir"] if launch else "–")}</code></td>'
                     f'<td>{_fmt((watch or {}).get("elapsed_hours"), 1)}</td></tr>')
    table.append('</tbody></table></div>')
    decisions = ['<div class="fig"><table class="map"><thead><tr><th>轮</th><th>时间</th><th>判定</th><th>目标</th><th>理由</th><th>下一轮</th></tr></thead><tbody>']
    for (block, spec, launch, watch, decision, verdict), name in zip(rendered, names):
        if decision:
            decisions.append(f'<tr><td>{name}</td><td>{html.escape(decision.get("decided_at", ""))}</td><td>{verdict}</td>'
                             f'<td>{html.escape(decision.get("goal_status", ""))}</td><td>{html.escape(decision.get("reason", ""))}</td>'
                             f'<td>{html.escape((decision.get("next") or {}).get("name", ""))}</td></tr>')
    decisions.append('</tbody></table></div>')
    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>璀璨宝石 AlphaZero 自动实验</title>
<style>{CSS}</style>
</head>
<body>
<header class="top"><div class="top-in"><div class="brand"><a href="./index.html">笔记</a> <span>· 璀璨宝石自动实验</span></div><nav class="toc"><a href="#goal">目标</a><a href="#bench">基准</a><a href="#priors">先验</a><a href="#program">纲领</a><a href="#rounds">逐轮</a><a href="#decisions">决策</a></nav><button class="tbtn" id="themeBtn" type="button">切换主题</button></div></header>
<div class="wrap">
<h1>璀璨宝石 AlphaZero 自动实验</h1>
<p class="sub">对 Splendor（璀璨宝石，四人局）的 Gumbel AlphaZero 训练运行一个自主实验循环：每一轮由 agent 提出要验证的改动，启动训练、按间隔检查、跑完后分析数据、下判定、定下一轮。这一页由实验账本 <code>auto_research/</code> 渲染，每轮更新；每轮的配置、实验目录、指标文件都写在轮次记录里。</p>
<p class="meta">状态 {html.escape(state.get('status', ''))} · 阶段 {html.escape(state.get('phase', ''))} · 当前轮 {html.escape(state.get('current_round') or '–')} · 目标 {html.escape(state.get('goal_status', 'open'))} · 更新 {now()}</p>
{('<div class="note warn">' + html.escape(state['note']) + '</div>') if state.get('note') else ''}
<div class="kv">
  <div><div class="l">已开始轮次</div><div class="v">{sum(1 for r in rendered if r[2])}</div></div>
  <div><div class="l">支持 / 证伪 / 不确定</div><div class="v">{counts['supported']} / {counts['refuted']} / {counts['inconclusive']}</div></div>
  <div><div class="l">累计 GPU 时长</div><div class="v">{gpu:.1f} h</div><div class="l">单卡 RTX PRO 6000</div></div>
  <div><div class="l">当前最好模型</div><div class="v" style="font-size:14px">{html.escape(best.get('path', '–'))}</div><div class="l">{html.escape(best.get('note', ''))}</div></div>
</div>
<h2 id="goal">1. 终极目标</h2><div class="md">{md_to_html(read_text(os.path.join(LEDGER, 'goal.md')))}</div>
<h2 id="bench">1a. 强度基准（"一般水平玩家"的代理）</h2><div class="md">{md_to_html(read_text(os.path.join(LEDGER, 'benchmark.md')))}</div>
<h2 id="priors">1b. 设计先验</h2><div class="md">{md_to_html(read_text(os.path.join(LEDGER, 'priors.md')))}</div>
<h2 id="program">2. 研究纲领（agent 维护）</h2><div class="md">{md_to_html(read_text(os.path.join(LEDGER, 'program.md')))}</div>
<h2 id="rounds">3. 逐轮记录</h2>
{''.join(table)}
{''.join(r[0] for r in rendered)}
<h2 id="decisions">4. 决策记录</h2>
{''.join(decisions)}
<p class="meta">机制：<code>auto_research/README.md</code>；控制脚本 <code>auto_research/loop.py</code>（launch / watch / stop / render）。</p>
</div>
<script>(function(){{var r=document.documentElement,f=null;document.getElementById('themeBtn').addEventListener('click',function(){{var d=f?f==='dark':window.matchMedia('(prefers-color-scheme: dark)').matches;f=d?'light':'dark';r.setAttribute('data-theme',f);}});}})();</script>
</body>
</html>
"""
    out = args.out or os.path.join(LEDGER, 'splendor-auto.html')
    with open(out, 'w', encoding='UTF-8') as f:
        f.write(page)
    print(f'{out}: {len(page)} bytes, {len(names)} rounds')


def main():
    parser = argparse.ArgumentParser(description='Splendor auto-research loop controller')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('launch')
    p.add_argument('round')
    p.add_argument('--force', action='store_true')
    p.set_defaults(func=cmd_launch)
    p = sub.add_parser('watch')
    p.add_argument('round')
    p.add_argument('--enforce-budget', action='store_true')
    p.add_argument('--json', action='store_true')
    p.set_defaults(func=cmd_watch)
    p = sub.add_parser('stop')
    p.add_argument('round')
    p.add_argument('--timeout', type=float, default=120.)
    p.set_defaults(func=cmd_stop)
    p = sub.add_parser('status')
    p.set_defaults(func=cmd_status)
    p = sub.add_parser('render')
    p.add_argument('--out', default=None)
    p.set_defaults(func=cmd_render)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
