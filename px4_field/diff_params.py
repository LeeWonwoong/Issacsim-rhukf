#!/usr/bin/env python3
"""diff_params.py — QGC .params 파일 두 개를 비교한다.

기체 사이에서 "설정을 옮길 값"을 고를 때 쓴다. 그룹(접두사)별로 묶어
보여주므로 조종기(RC_/COM_RC_) 관련만 골라내기 쉽다.

사용:
    python3 diff_params.py A.params B.params
    python3 diff_params.py A.params B.params --only RC_ COM_RC MAN_
    python3 diff_params.py A.params B.params --only RC_ --apply-to-b out.params

  --only            해당 접두사로 시작하는 것만 본다
  --apply-to-b      A 의 값을 B 에 덮어쓴 새 파일을 만든다 (QGC 로 불러올 수 있음)
                    ※ B 에 없는 항목은 추가하지 않는다 (펌웨어가 다르면 위험)

.params 형식:  <vehicle_id> <component_id> <name> <value> <type>
"""
import argparse
import sys


def load(path):
    """이름 → (값문자열, 원본줄) 사전.  주석/빈 줄은 헤더로 따로 보관."""
    vals, header = {}, []
    with open(path) as f:
        for line in f:
            s = line.rstrip('\n')
            if not s.strip() or s.lstrip().startswith('#'):
                header.append(s)
                continue
            p = s.split()
            if len(p) < 4:
                continue
            vals[p[2]] = (p[3], s)
    return vals, header


def num(s):
    try:
        return float(s)
    except ValueError:
        return None


def same(a, b):
    """1.0 과 1.000000 을 같게 본다."""
    x, y = num(a), num(b)
    if x is None or y is None:
        return a == b
    return abs(x - y) <= 1e-9 * max(1.0, abs(x), abs(y))


def fmt(v):
    x = num(v)
    if x is None:
        return v
    return str(int(x)) if x == int(x) and abs(x) < 1e15 else f'{x:g}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a'); ap.add_argument('b')
    ap.add_argument('--only', nargs='*', default=None,
                    help='이 접두사로 시작하는 파라미터만 (예: RC_ COM_RC MAN_)')
    ap.add_argument('--apply-to-b', metavar='OUT',
                    help='A 의 값을 B 에 덮어쓴 파일을 만든다')
    args = ap.parse_args()

    A, _ = load(args.a)
    B, hb = load(args.b)
    for path, d in ((args.a, A), (args.b, B)):
        if not d:
            print(f'✗ {path} 에 파라미터가 없다 (0 바이트이거나 형식이 다름). '
                  f'QGC → Parameters → Tools → Save to file 로 다시 뽑을 것.')
            return 1

    def keep(n):
        return args.only is None or any(n.startswith(p) for p in args.only)

    common = sorted(n for n in A.keys() & B.keys() if keep(n))
    diff = [n for n in common if not same(A[n][0], B[n][0])]
    only_a = sorted(n for n in A.keys() - B.keys() if keep(n))
    only_b = sorted(n for n in B.keys() - A.keys() if keep(n))

    print(f'A = {args.a}   ({len(A)} 개)')
    print(f'B = {args.b}   ({len(B)} 개)')
    if args.only:
        print(f'필터: {" ".join(args.only)}*')
    print(f'\n공통 {len(common)} 개 중 값이 다른 것 {len(diff)} 개\n')

    if diff:
        print(f'  {"파라미터":<24} {"A":>16}  {"B":>16}')
        print('  ' + '-' * 60)
        for n in diff:
            print(f'  {n:<24} {fmt(A[n][0]):>16}  {fmt(B[n][0]):>16}')
    else:
        print('  (값이 다른 것 없음 — 두 기체가 이 범위에서는 동일)')

    for lbl, lst, src in (('A 에만 있음', only_a, args.a), ('B 에만 있음', only_b, args.b)):
        if lst:
            print(f'\n  {lbl} ({len(lst)} 개) — 펌웨어 버전 차이일 수 있다')
            for n in lst:
                print(f'    {n}')

    if args.apply_to_b:
        if not diff:
            print('\n덮어쓸 것이 없다.')
            return 0
        out = []
        for line in hb:
            out.append(line)
        for n in sorted(B):
            if n in diff:
                p = B[n][1].split()
                p[3] = A[n][0]                 # A 의 값으로 교체
                out.append('\t'.join(p))
            else:
                out.append(B[n][1])
        with open(args.apply_to_b, 'w') as f:
            f.write('\n'.join(out) + '\n')
        print(f'\n✓ {args.apply_to_b} 생성 — B 를 기준으로 {len(diff)} 개를 A 값으로 교체했다.')
        print('  QGC → Parameters → Tools → Load from file 로 불러온 뒤 재부팅.')
        print('  ⚠ 불러오기 전에 현재 파라미터를 반드시 백업할 것.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
