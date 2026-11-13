# e-means in Bend 2: a second implementation of faiss contrib/e_means
# (NumpyEmeans, spherical or Euclidean), to compare against the Rust one.
#
#   emeans <vectors.f32> <n> <d> <k> <nepochs> <seed> <spherical 0|1> <outdir> [gpu 0|1]
#
# Reads n x d little-endian f32 rows, writes <outdir>/labels.i32 (last-pass
# labels, -1 for rows the last pass skipped), <outdir>/centroids.f32 (k x d)
# and <outdir>/report.json. See .okf/design/bend-emeans.md.
#
# Parallelism: each batch's assignment is a fork tree over the batch rows
# (on the GPU with gpu=1), and each EMA update is a fork tree over the
# clusters. The lanes share the arrays through @unsafe `+` parameters, as
# Bend's own histogram bench and array_atomic_fadd test do; every lane
# writes only its own slots.
import Base

# Basics
# ------

def pickf(c: Bool, +a: F32, +b: F32) -> F32:
  match c:
    case True{}:
      a
    case False{}:
      b

def picku(c: Bool, +a: U32, +b: U32) -> U32:
  match c:
    case True{}:
      a
    case False{}:
      b

def f32.of_bits(u: U32) -> F32:
  match u:
    case U32{w}:
      F32{w}

def f32.abs_max(+m: F32, +v: F32) -> F32:
  +a = F32.abs(v)
  pickf((a > m : F32), a, m)

def nat(+u: U32) -> Nat:
  U32.to_nat(u)

def ninf() -> F32:
  (0.0 - 3.4028235e38 : F32)

# ceil(log2(m)): the depth of an array that holds m cells.
def depth.of(big: Bool, +m: U32) -> Nat:
  match big:
    case False{}:
      0n
    case True{}:
      1n+U32.log2((m - 1 : U32))

def depth(+m: U32) -> Nat:
  depth.of(U32.is_lt(1, m), m)

# Random numbers: a counter-based hash (lowbias32), so no state is threaded.
def mix(+a: U32) -> U32:
  +b = (a .^. (a >> 16n) : U32)
  +c = (b * 2146121005 : U32)
  +e = (c .^. (c >> 15n) : U32)
  +f = (e * 2221713035 : U32)
  (f .^. (f >> 16n) : U32)

def rnd(+key: U32, +i: U32) -> U32:
  mix((key .^. mix((i * 2654435769 + 1 : U32)) : U32))

# A uniform float in (0, 1).
def unif(+h: U32) -> F32:
  ((U32.to_f32((h >> 8n : U32)) + 0.5) * 5.9604645e-8 : F32)

# Array helpers
# -------------

def zero.f(n: Nat, +i: U32, a: Array<F32>) -> Array<F32>:
  match n:
    case 0n:
      a
    case 1n+q:
      zero.f(q, (i + 1 : U32), Array.set(F32, a, i, 0.0))

def fill.u(n: Nat, +i: U32, +v: U32, a: Array<U32>) -> Array<U32>:
  match n:
    case 0n:
      a
    case 1n+q:
      fill.u(q, (i + 1 : U32), v, Array.set(U32, a, i, v))

def iota(n: Nat, +i: U32, a: Array<U32>) -> Array<U32>:
  match n:
    case 0n:
      a
    case 1n+q:
      iota(q, (i + 1 : U32), Array.set(U32, a, i, i))

def swap.b(+i: U32, +j: U32, +vi: U32, r: Array<U32> & U32) -> Array<U32>:
  (a, +vj) = r
  Array.set(U32, Array.set(U32, a, i, vj), j, vi)

def swap.a(+i: U32, +j: U32, r: Array<U32> & U32) -> Array<U32>:
  (a, +vi) = r
  swap.b(i, j, vi, Array.get(U32, a, j))

def swap(a: Array<U32>, +i: U32, +j: U32) -> Array<U32>:
  swap.a(i, j, Array.get(U32, a, i))

# Fisher-Yates, from slot i down to slot 1.
def shuffle.go(n: Nat, +i: U32, +key: U32, a: Array<U32>) -> Array<U32>:
  match n:
    case 0n:
      a
    case 1n+q:
      +j = (rnd(key, i) % (i + 1) : U32)
      shuffle.go(q, (i - 1 : U32), key, swap(a, i, j))

def shuffle(+m: U32, +key: U32, a: Array<U32>) -> Array<U32>:
  shuffle.go(nat((m - 1 : U32)), (m - 1 : U32), key, a)

# Partial Fisher-Yates: slots 0 .. n - 1 get distinct random entries of m.
def choose(n: Nat, +i: U32, +m: U32, +key: U32, a: Array<U32>) -> Array<U32>:
  match n:
    case 0n:
      a
    case 1n+q:
      +j = (i + rnd(key, i) % (m - i) : U32)
      choose(q, (i + 1 : U32), m, key, swap(a, i, j))

# Sum of n F32 cells from i, sequential.
def sum.go(n: Nat, +i: U32, +acc: F32, r: Array<F32> & F32) -> Array<F32> & F32:
  match n:
    case 0n:
      (a, +v) = r
      (a, acc)
    case 1n+q:
      (a, +v) = r
      sum.go(q, (i + 1 : U32), (acc + v : F32), Array.get(F32, a, (i + 1 : U32)))

def sum.f(a: Array<F32>, +i: U32, +n: U32) -> Array<F32> & F32:
  sum.go(nat(n), i, 0.0, Array.get(F32, a, i))

# Loading
# -------

def fill.bytes(bs: List<&2, U32>, +i: U32, x: Array<F32>) -> Array<F32> & U32:
  match bs:
    case b0 <> b1 <> b2 <> b3 <> rest:
      +w = (b0 .|. (b1 << 8n) .|. (b2 << 16n) .|. (b3 << 24n) : U32)
      fill.bytes(rest, (i + 1 : U32), Array.set(F32, x, i, f32.of_bits(w)))
    case _:
      (x, i)

# The floats of one read, or none when it failed: the caller checks the count.
def load.take(+i: U32, x: Array<F32>, res: Result<&1, &1, U32 & String, List<&2, U32>>) -> Array<F32> & U32:
  match res:
    case Fail{e}:
      (x, i)
    case Done{bs}:
      fill.bytes(bs, i, x)

@unsafe
def load.go(n: Nat, f: File, +off: U32, +chunk: U32, +i: U32, x: Array<F32>) -> IO(Array<F32> & U32):
  match n:
    case 0n:
      do IO<Array<F32> & U32>:
        File.close(f)
        return (x, i)
    case 1n+p:
      IO.bind(File & Result<&1, &1, U32 & String, List<&2, U32>>, Array<F32> & U32,
        File.read_at(f, off, chunk), pair => load.next(p, off, chunk, i, x, pair))

@unsafe
def load.next(n: Nat, +off: U32, +chunk: U32, +i: U32, x: Array<F32>,
  pair: File & Result<&1, &1, U32 & String, List<&2, U32>>) -> IO(Array<F32> & U32):
  (f2, res) = pair
  load.cont(n, off, chunk, f2, load.take(i, x, res))

@unsafe
def load.cont(n: Nat, +off: U32, +chunk: U32, f: File, xj: Array<F32> & U32) -> IO(Array<F32> & U32):
  (x2, +j) = xj
  load.go(n, f, (off + chunk : U32), chunk, j, x2)

# Writing
# -------

def le.bytes(+w: U32, tail: List<&2, U32>) -> List<&2, U32>:
  (w .&. 255 : U32) <> ((w >> 8n) .&. 255 : U32) <> ((w >> 16n) .&. 255 : U32) <> (w >> 24n : U32) <> tail

# The bytes of cells i - n + 1 .. i of a, built from the back.
def bytes.u.go(n: Nat, +i: U32, acc: List<&2, U32>, r: Array<U32> & U32) -> Array<U32> & List<&2, U32>:
  match n:
    case 0n:
      (a, +v) = r
      (a, le.bytes(v, acc))
    case 1n+q:
      (a, +v) = r
      bytes.u.go(q, (i - 1 : U32), le.bytes(v, acc), Array.get(U32, a, (i - 1 : U32)))

def bytes.f.go(n: Nat, +i: U32, acc: List<&2, U32>, r: Array<F32> & F32) -> Array<F32> & List<&2, U32>:
  match n:
    case 0n:
      (a, +v) = r
      (a, le.bytes(F32.bits(v), acc))
    case 1n+q:
      (a, +v) = r
      bytes.f.go(q, (i - 1 : U32), le.bytes(F32.bits(v), acc), Array.get(F32, a, (i - 1 : U32)))

def wrote(pair: File & Result<&1, &1, U32 & String, Unit>) -> IO(File):
  (f, res) = pair
  do IO<File>:
    IO.pass(Unit, res)
    return f

# cnt cells from i, in chunks of at most 65536.
@unsafe
def write.u(f: File, +i: U32, +cnt: U32, a: Array<U32>) -> IO(Unit):
  write.u.if(U32.is_zero(cnt), f, i, cnt, a)

@unsafe
def write.u.if(done: Bool, f: File, +i: U32, +cnt: U32, a: Array<U32>) -> IO(Unit):
  match done:
    case True{}:
      File.close(f)
    case False{}:
      write.u.at(f, i, cnt, U32.min(cnt, 65536), a)

@unsafe
def write.u.at(f: File, +i: U32, +cnt: U32, +m: U32, a: Array<U32>) -> IO(Unit):
  write.u.put(f, (i + m : U32), (cnt - m : U32),
    bytes.u.go(nat((m - 1 : U32)), (i + m - 1 : U32), [], Array.get(U32, a, (i + m - 1 : U32))))

@unsafe
def write.u.put(f: File, +i: U32, +cnt: U32, r: Array<U32> & List<&2, U32>) -> IO(Unit):
  (a, bs) = r
  do IO<Unit>:
    p : File & Result<&1, &1, U32 & String, Unit> <- File.write_bytes(f, bs)
    f2 : File <- wrote(p)
    write.u(f2, i, cnt, a)

@unsafe
def write.f(f: File, +i: U32, +cnt: U32, a: Array<F32>) -> IO(Unit):
  write.f.if(U32.is_zero(cnt), f, i, cnt, a)

@unsafe
def write.f.if(done: Bool, f: File, +i: U32, +cnt: U32, a: Array<F32>) -> IO(Unit):
  match done:
    case True{}:
      File.close(f)
    case False{}:
      write.f.at(f, i, cnt, U32.min(cnt, 65536), a)

@unsafe
def write.f.at(f: File, +i: U32, +cnt: U32, +m: U32, a: Array<F32>) -> IO(Unit):
  write.f.put(f, (i + m : U32), (cnt - m : U32),
    bytes.f.go(nat((m - 1 : U32)), (i + m - 1 : U32), [], Array.get(F32, a, (i + m - 1 : U32))))

@unsafe
def write.f.put(f: File, +i: U32, +cnt: U32, r: Array<F32> & List<&2, U32>) -> IO(Unit):
  (a, bs) = r
  do IO<Unit>:
    p : File & Result<&1, &1, U32 & String, Unit> <- File.write_bytes(f, bs)
    f2 : File <- wrote(p)
    write.f(f2, i, cnt, a)

# Recipe (base.py: scaling laws, equations 1-3)
# ---------------------------------------------

# Round a positive value to the nearest power of two (_round_power_of_two).
def pow2.near(small: Bool, +v: F32) -> U32:
  match small:
    case True{}:
      1
    case False{}:
      +e = F32.to_u32(F32.floor(F32.log2(v)))
      +lo = U32.shln(1, nat(e))
      +hi = (lo * 2 : U32)
      picku((v / U32.to_f32(lo) < U32.to_f32(hi) / v : F32), lo, hi)

def pow2.round(+v: F32) -> U32:
  pow2.near((v <= 1.0 : F32), v)

# The batch cap: the throughput law rounds n / 2, the quality law floors it.
def batch.cap(quality: Bool, +n: U32) -> U32:
  match quality:
    case True{}:
      U32.shr(n)
    case False{}:
      pow2.round((U32.to_f32(n) / 2.0 : F32))

def batch.size(+quality: Bool, +n: U32, +k: U32, +t: F32) -> U32:
  +c = pickf(quality, 0.00439175, 0.035134)
  +raw = (c * F32.pow(U32.to_f32(k), 0.134749) * F32.pow(U32.to_f32(n), 0.876473) * F32.pow(t, 0.813344) : F32)
  U32.max(1, U32.min(U32.min(pow2.round(raw), batch.cap(quality, n)), n))

def alpha0(+n: U32, +k: U32, +t: F32, +b: U32) -> F32:
  +a = (0.8244 * F32.pow(U32.to_f32(k), (0.0 - 0.4617 : F32)) * F32.pow(U32.to_f32(b), 0.9003) * F32.pow(U32.to_f32(n), (0.0 - 0.4498 : F32)) * F32.pow(t, (0.0 - 0.1057 : F32)) : F32)
  F32.clamp(a, 0.0001, 1.0)

# Round half to even, as Python's round.
def round.even.at(+f: F32, +g: F32) -> U32:
  +u = F32.to_u32(f)
  +odd = U32.is_eq((u .&. 1 : U32), 1)
  picku((g > 0.5 : F32), (u + 1 : U32), picku((g < 0.5 : F32), u, picku(odd, (u + 1 : U32), u)))

def round.even(+v: F32) -> U32:
  +f = F32.floor(v)
  round.even.at(f, (v - f : F32))

def rows.full(n: Nat, +m: U32, tail: List<&2, U32>) -> List<&2, U32>:
  match n:
    case 0n:
      tail
    case 1n+q:
      m <> rows.full(q, m, tail)

def rows.rem(zero: Bool, +r: U32, tail: List<&2, U32>) -> List<&2, U32>:
  match zero:
    case True{}:
      tail
    case False{}:
      r <> tail

def rows.of(whole: Nat, +rem: U32, +n: U32) -> List<&2, U32>:
  match whole:
    case 0n:
      [rem]
    case 1n+q:
      rows.full(q, n, rows.rem(U32.is_zero(rem), rem, [n]))

# _epoch_rows: whole passes, a rounded fractional prefix, then the closing pass.
def epoch.rows(+n: U32, +t: F32) -> List<&2, U32>:
  +whole = F32.floor(t)
  rows.of(nat(F32.to_u32(whole)), round.even(((t - whole) * U32.to_f32(n) : F32)), n)

def ceil.div(+a: U32, +b: U32) -> U32:
  ((a + b - 1) / b : U32)

def steps.of(rs: List<&2, U32>, +b: U32) -> List<&2, U32>:
  match rs:
    case []:
      []
    case r <> t:
      ceil.div(r, b) <> steps.of(t, b)

def list.sum(xs: List<&2, U32>, +acc: U32) -> U32:
  match xs:
    case []:
      acc
    case x <> t:
      list.sum(t, (acc + x : U32))

def list.strs(xs: List<&2, U32>) -> List<&2, String>:
  match xs:
    case []:
      []
    case x <> t:
      U32.show(x) <> list.strs(t)

def list.show(xs: List<&2, U32>) -> String:
  "[" ++ String.join(list.strs(xs), ", ") ++ "]"

# Cosine decay from alpha0 to alpha0 / 10 (_scheduled_alpha).
def alpha.at(+a0: F32, +step: U32, +total: U32) -> F32:
  +fr = (U32.to_f32(U32.min(step, total)) / U32.to_f32(U32.max(1, total)) : F32)
  +cs = (0.5 * (1.0 + F32.cos((F32.pi() * fr : F32))) : F32)
  (a0 * (0.1 + 0.9 * cs) : F32)

# Assignment kernel
# -----------------

type D2 is Type:
  D2{x: Array<F32>, u: Array<F32>, s: F32}

def dot.acc(+acc: F32, +xv: F32, r: Array<F32> & F32) -> Array<F32> & F32:
  (u, +uv) = r
  (u, (acc + xv * uv : F32))

# n more products into acc (beside u); rx holds x[xi].
def dot(n: Nat, +xi: U32, +ci: U32, ua: Array<F32> & F32, rx: Array<F32> & F32) -> D2:
  match n:
    case 0n:
      (u, +acc) = ua
      (x, +xv) = rx
      D2{x, u, acc}
    case 1n+q:
      (u, +acc) = ua
      (x, +xv) = rx
      dot(q, (xi + 1 : U32), (ci + 1 : U32), dot.acc(acc, xv, Array.get(F32, u, ci)), Array.get(F32, x, (xi + 1 : U32)))

type Q is Type:
  Q{a: Array<F32>, v0: F32, v1: F32, v2: F32, v3: F32}

def q.d(+v0: F32, +v1: F32, +v2: F32, r: Array<F32> & F32) -> Q:
  (a, +v3) = r
  Q{a, v0, v1, v2, v3}

def q.c(+i: U32, +v0: F32, +v1: F32, r: Array<F32> & F32) -> Q:
  (a, +v2) = r
  q.d(v0, v1, v2, Array.get(F32, a, (i + 3 : U32)))

def q.b(+i: U32, +v0: F32, r: Array<F32> & F32) -> Q:
  (a, +v1) = r
  q.c(i, v0, v1, Array.get(F32, a, (i + 2 : U32)))

def q.a(+i: U32, r: Array<F32> & F32) -> Q:
  (a, +v0) = r
  q.b(i, v0, Array.get(F32, a, (i + 1 : U32)))

def get4(a: Array<F32>, +i: U32) -> Q:
  q.a(i, Array.get(F32, a, i))

# n blocks of four products in four running sums, for speed (four add
# chains) and for precision (a pairwise finish); qx and qu hold the next four.
def dot4(n: Nat, +xi: U32, +ci: U32, +s0: F32, +s1: F32, +s2: F32, +s3: F32, qx: Q, qu: Q) -> D2:
  match n:
    case 0n:
      Q{x, +x0, +x1, +x2, +x3} = qx
      Q{u, +u0, +u1, +u2, +u3} = qu
      D2{x, u, ((s0 + s1) + (s2 + s3) : F32)}
    case 1n+q:
      Q{x, +x0, +x1, +x2, +x3} = qx
      Q{u, +u0, +u1, +u2, +u3} = qu
      +xj = (xi + 4 : U32)
      +cj = (ci + 4 : U32)
      dot4(q, xj, cj, (s0 + x0 * u0 : F32), (s1 + x1 * u1 : F32), (s2 + x2 * u2 : F32), (s3 + x3 * u3 : F32), get4(x, xj), get4(u, cj))

def dot.tail(+n: U32, +xi: U32, +ci: U32, st: D2) -> D2:
  D2{x, u, +s} = st
  dot(nat(n), xi, ci, (u, s), Array.get(F32, x, xi))

# x[xi .. xi + d] . u[ci .. ci + d]
def dot.row(+d: U32, +xi: U32, +ci: U32, x: Array<F32>, u: Array<F32>) -> D2:
  +m = (d .&. 4294967292 : U32)
  dot.tail((d - m : U32), (xi + m : U32), (ci + m : U32),
    dot4(nat((d >> 2n : U32)), xi, ci, 0.0, 0.0, 0.0, 0.0, get4(x, xi), get4(u, ci)))

# The score of a centroid: x . u_j - h_j (h is 0 when spherical, |u_j|^2 / 2
# when Euclidean), so the nearest centroid has the highest score.
type Sc is Type:
  Sc{x: Array<F32>, u: Array<F32>, h: Array<F32>, s: F32}

def score.h(+s: F32, x: Array<F32>, u: Array<F32>, rh: Array<F32> & F32) -> Sc:
  (h, +hv) = rh
  Sc{x, u, h, (s - hv : F32)}

def score.k(+j: U32, h: Array<F32>, dd: D2) -> Sc:
  D2{x, u, +s} = dd
  score.h(s, x, u, Array.get(F32, h, j))

def score(+j: U32, +d: U32, +xi: U32, x: Array<F32>, u: Array<F32>, h: Array<F32>) -> Sc:
  score.k(j, h, dot.row(d, xi, (j * d : U32), x, u))

type Best is Type:
  Best{x: Array<F32>, u: Array<F32>, h: Array<F32>, s: F32, j: U32}

# Centroids j .. j + left against the row at xi; st holds the score of
# centroid j - 1. Ties keep the lower index.
def cen(left: Nat, +j: U32, +d: U32, +xi: U32, +bs: F32, +bj: U32, st: Sc) -> Best:
  match left:
    case 0n:
      Sc{x, u, h, +s} = st
      +better = (s > bs : F32)
      Best{x, u, h, pickf(better, s, bs), picku(better, (j - 1 : U32), bj)}
    case 1n+q:
      Sc{x, u, h, +s} = st
      +better = (s > bs : F32)
      cen(q, (j + 1 : U32), d, xi, pickf(better, s, bs), picku(better, (j - 1 : U32), bj), score(j, d, xi, x, u, h))

@unsafe
def best.join(a: Best, b: Best) -> Best:
  Best{x, u, h, +s, +j} = a
  Best{x2, u2, h2, +s2, +j2} = b
  +right = (s2 > s : F32)
  Best{Array.join(F32, x, x2), Array.join(F32, u, u2), Array.join(F32, h, h2), pickf(right, s2, s), picku(right, j2, j)}

# Centroids j0 .. j0 + cnt (cnt >= 1) against the row at xi, split cd more
# times: on the GPU a row alone leaves most lanes idle.
@unsafe
def ctree(+cd: Nat, +j0: U32, +cnt: U32, +d: U32, +xi: U32, +x: Array<F32>, +u: Array<F32>, +h: Array<F32>) -> Best:
  match cd:
    case 0n:
      cen(nat((cnt - 1 : U32)), (j0 + 1 : U32), d, xi, ninf(), j0, score(j0, d, xi, x, u, h))
    case 1n+q:
      +m = U32.shr(cnt)
      a b = ctree(q, j0, m, d, xi, x, u, h) ctree(q, (j0 + m : U32), (cnt - m : U32), d, xi, x, u, h)
      best.join(a, b)

# Assignment lanes share x (rows), xn (row norms), u, h, p (the epoch's
# order) and l (labels by row); o is the lane's objective.
type Ak is Type:
  Ak{x: Array<F32>, xn: Array<F32>, u: Array<F32>, h: Array<F32>, p: Array<U32>, l: Array<U32>, o: F32}

def row.obj.e(+s: F32, +o: F32, r: Array<F32> & F32) -> Array<F32> & F32:
  (xn, +nv) = r
  (xn, (o + (nv - 2.0 * s) : F32))

# The row's error: 2 - 2 cos when spherical, |x - c|^2 when Euclidean.
def row.obj(sph: Bool, +r: U32, +s: F32, +o: F32, xn: Array<F32>) -> Array<F32> & F32:
  match sph:
    case True{}:
      (xn, (o + (2.0 - 2.0 * s) : F32))
    case False{}:
      row.obj.e(s, o, Array.get(F32, xn, r))

def row.put(+r: U32, +j: U32, x: Array<F32>, u: Array<F32>, h: Array<F32>, p: Array<U32>, l: Array<U32>, xo: Array<F32> & F32) -> Ak:
  (xn, +o) = xo
  Ak{x, xn, u, h, p, Array.set(U32, l, r, j), o}

def row.fin(sph: Bool, +r: U32, +o: F32, xn: Array<F32>, p: Array<U32>, l: Array<U32>, b: Best) -> Ak:
  Best{x, u, h, +s, +j} = b
  row.put(r, j, x, u, h, p, l, row.obj(sph, r, s, o, xn))

@unsafe
def row.at(+cd: Nat, +sph: Bool, +k: U32, +d: U32, +o: F32, x: Array<F32>, xn: Array<F32>, u: Array<F32>, h: Array<F32>, l: Array<U32>, pr: Array<U32> & U32) -> Ak:
  (p, +r) = pr
  +xi = (r * d : U32)
  row.fin(sph, r, o, xn, p, l, ctree(cd, 0, k, d, xi, x, u, h))

@unsafe
def row.one(+cd: Nat, +sph: Bool, +t: U32, +k: U32, +d: U32, a: Ak) -> Ak:
  Ak{x, xn, u, h, p, l, +o} = a
  row.at(cd, sph, k, d, o, x, xn, u, h, l, Array.get(U32, p, t))

# Batch positions t .. t + left of the order.
@unsafe
def row.go(left: Nat, +cd: Nat, +sph: Bool, +t: U32, +k: U32, +d: U32, a: Ak) -> Ak:
  match left:
    case 0n:
      a
    case 1n+q:
      row.go(q, cd, sph, (t + 1 : U32), k, d, row.one(cd, sph, t, k, d, a))

@unsafe
def ak.join(a: Ak, b: Ak) -> Ak:
  Ak{x, xn, u, h, p, l, +o} = a
  Ak{x2, xn2, u2, h2, p2, l2, +o2} = b
  Ak{Array.join(F32, x, x2), Array.join(F32, xn, xn2), Array.join(F32, u, u2), Array.join(F32, h, h2),
    Array.join(U32, p, p2), Array.join(U32, l, l2), (o + o2 : F32)}

# Positions lo .. lo + cnt of the order, split dep more times; lanes sum
# their objectives pairwise on the way up.
@unsafe
def atree(+dep: Nat, +cd: Nat, +sph: Bool, +lo: U32, +cnt: U32, +k: U32, +d: U32, +x: Array<F32>, +xn: Array<F32>,
  +u: Array<F32>, +h: Array<F32>, +p: Array<U32>, +l: Array<U32>) -> Ak:
  match dep:
    case 0n:
      row.go(nat(cnt), cd, sph, lo, k, d, Ak{x, xn, u, h, p, l, 0.0})
    case 1n+q:
      +m = U32.shr(cnt)
      a b = atree(q, cd, sph, lo, m, k, d, x, xn, u, h, p, l) atree(q, cd, sph, (lo + m : U32), (cnt - m : U32), k, d, x, xn, u, h, p, l)
      ak.join(a, b)

# Readout (_readout, normalize_into)
# ----------------------------------

type CU is Type:
  CU{c: Array<F32>, u: Array<F32>, h: Array<F32>}

def amax.go(n: Nat, +i: U32, +m: F32, r: Array<F32> & F32) -> Array<F32> & F32:
  match n:
    case 0n:
      (a, +v) = r
      (a, m)
    case 1n+q:
      (a, +v) = r
      amax.go(q, (i + 1 : U32), f32.abs_max(m, v), Array.get(F32, a, (i + 1 : U32)))

def sq.go(n: Nat, +i: U32, +sc: F32, +acc: F32, r: Array<F32> & F32) -> Array<F32> & F32:
  match n:
    case 0n:
      (a, +v) = r
      (a, acc)
    case 1n+q:
      (a, +v) = r
      +w = (v / sc : F32)
      sq.go(q, (i + 1 : U32), sc, (acc + w * w : F32), Array.get(F32, a, (i + 1 : U32)))

# u[i ..] = c[i ..] / sc / nn
def unit.go(n: Nat, +i: U32, +sc: F32, +nn: F32, u: Array<F32>, r: Array<F32> & F32) -> Array<F32> & Array<F32>:
  match n:
    case 0n:
      (c, +v) = r
      (c, u)
    case 1n+q:
      (c, +v) = r
      unit.go(q, (i + 1 : U32), sc, nn, Array.set(F32, u, i, ((v / sc) / nn : F32)), Array.get(F32, c, (i + 1 : U32)))

def ro.unit(h: Array<F32>, cu: Array<F32> & Array<F32>) -> CU:
  (c, u) = cu
  CU{c, u, h}

def ro.norm(+i: U32, +d: U32, +sc: F32, u: Array<F32>, h: Array<F32>, rs: Array<F32> & F32) -> CU:
  (c, +ss) = rs
  +nn = F32.sqrt(ss)
  ro.unit(h, unit.go(nat(d), i, sc, pickf((nn > 0.0 : F32), nn, 1.0), u, Array.get(F32, c, i)))

def ro.scale(+i: U32, +d: U32, u: Array<F32>, h: Array<F32>, rm: Array<F32> & F32) -> CU:
  (c, +m) = rm
  +sc = pickf((m > 0.0 : F32), m, 1.0)
  ro.norm(i, d, sc, u, h, sq.go(nat(d), i, sc, 0.0, Array.get(F32, c, i)))

def copy.go(n: Nat, +i: U32, +acc: F32, u: Array<F32>, r: Array<F32> & F32) -> D2:
  match n:
    case 0n:
      (c, +v) = r
      D2{c, u, acc}
    case 1n+q:
      (c, +v) = r
      copy.go(q, (i + 1 : U32), (acc + v * v : F32), Array.set(F32, u, i, v), Array.get(F32, c, (i + 1 : U32)))

def ro.half(+j: U32, h: Array<F32>, dd: D2) -> CU:
  D2{c, u, +ss} = dd
  CU{c, u, Array.set(F32, h, j, (0.5 * ss : F32))}

# Centroid j's readout: its unit direction (zero stays zero) when spherical;
# when Euclidean, itself, with h_j = |c_j|^2 / 2.
def readout(sph: Bool, +j: U32, +d: U32, cu: CU) -> CU:
  match sph:
    case True{}:
      CU{c, u, h} = cu
      +i = (j * d : U32)
      ro.scale(i, d, u, h, amax.go(nat(d), i, 0.0, Array.get(F32, c, i)))
    case False{}:
      CU{c, u, h} = cu
      +i = (j * d : U32)
      ro.half(j, h, copy.go(nat(d), i, 0.0, u, Array.get(F32, c, i)))

def readout.all(n: Nat, +sph: Bool, +j: U32, +d: U32, cu: CU) -> CU:
  match n:
    case 0n:
      cu
    case 1n+q:
      readout.all(q, sph, (j + 1 : U32), d, readout(sph, j, d, cu))

# Batch buckets (host)
# --------------------
# The update needs each cluster's batch rows. bc counts them, bo is where
# they start in bq, in batch order (np.add.at adds rows in that order).

type Bk is Type:
  Bk{p: Array<U32>, l: Array<U32>, bc: Array<U32>, bo: Array<U32>, bq: Array<U32>}

def inc.at(r: Array<U32> & U32, +j: U32) -> Array<U32>:
  (a, +v) = r
  Array.set(U32, a, j, (v + 1 : U32))

def bump(a: Array<U32>, +j: U32) -> Array<U32>:
  inc.at(Array.get(U32, a, j), j)

def cnt.lab(+t: U32, p: Array<U32>, bc: Array<U32>, bo: Array<U32>, bq: Array<U32>, pr: Array<U32> & U32) -> Bk:
  (l, +j) = pr
  Bk{p, l, bump(bc, j), bo, bq}

def cnt.row.at(+t: U32, l: Array<U32>, bc: Array<U32>, bo: Array<U32>, bq: Array<U32>, pr: Array<U32> & U32) -> Bk:
  (p, +r) = pr
  cnt.lab(t, p, bc, bo, bq, Array.get(U32, l, r))

def cnt.row(+t: U32, b: Bk) -> Bk:
  Bk{p, l, bc, bo, bq} = b
  cnt.row.at(t, l, bc, bo, bq, Array.get(U32, p, t))

def cnt.go(n: Nat, +t: U32, b: Bk) -> Bk:
  match n:
    case 0n:
      b
    case 1n+q:
      cnt.go(q, (t + 1 : U32), cnt.row(t, b))

# bo[j] = bc[0] + .. + bc[j - 1]
def off.go(n: Nat, +j: U32, +acc: U32, bo: Array<U32>, rc: Array<U32> & U32) -> Array<U32> & Array<U32>:
  match n:
    case 0n:
      (bc, +v) = rc
      (bc, bo)
    case 1n+q:
      (bc, +v) = rc
      off.go(q, (j + 1 : U32), (acc + v : U32), Array.set(U32, bo, j, acc), Array.get(U32, bc, (j + 1 : U32)))

def put.at(+r: U32, +j: U32, bq: Array<U32>, ro: Array<U32> & U32) -> Array<U32> & Array<U32>:
  (bo, +o) = ro
  (Array.set(U32, bo, j, (o + 1 : U32)), Array.set(U32, bq, o, r))

def put.fin(p: Array<U32>, l: Array<U32>, bc: Array<U32>, oq: Array<U32> & Array<U32>) -> Bk:
  (bo, bq) = oq
  Bk{p, l, bc, bo, bq}

def put.lab(+r: U32, p: Array<U32>, bc: Array<U32>, bo: Array<U32>, bq: Array<U32>, lr: Array<U32> & U32) -> Bk:
  (l, +j) = lr
  put.fin(p, l, bc, put.at(r, j, bq, Array.get(U32, bo, j)))

def put.row.at(l: Array<U32>, bc: Array<U32>, bo: Array<U32>, bq: Array<U32>, pr: Array<U32> & U32) -> Bk:
  (p, +r) = pr
  put.lab(r, p, bc, bo, bq, Array.get(U32, l, r))

def put.row(+t: U32, b: Bk) -> Bk:
  Bk{p, l, bc, bo, bq} = b
  put.row.at(l, bc, bo, bq, Array.get(U32, p, t))

def put.go(n: Nat, +t: U32, b: Bk) -> Bk:
  match n:
    case 0n:
      b
    case 1n+q:
      put.go(q, (t + 1 : U32), put.row(t, b))

# Back from the ends to the starts: bo[j] -= bc[j].
def back.sub(+j: U32, +c: U32, bc: Array<U32>, ro: Array<U32> & U32) -> Array<U32> & Array<U32>:
  (bo, +o) = ro
  (bc, Array.set(U32, bo, j, (o - c : U32)))

def back.at(+j: U32, bo: Array<U32>, rc: Array<U32> & U32) -> Array<U32> & Array<U32>:
  (bc, +c) = rc
  back.sub(j, c, bc, Array.get(U32, bo, j))

def back.go(n: Nat, +j: U32, cb: Array<U32> & Array<U32>) -> Array<U32> & Array<U32>:
  match n:
    case 0n:
      cb
    case 1n+q:
      (bc, bo) = cb
      back.go(q, (j + 1 : U32), back.at(j, bo, Array.get(U32, bc, j)))

def bk.offs(+k: U32, p: Array<U32>, l: Array<U32>, bq: Array<U32>, cb: Array<U32> & Array<U32>) -> Bk:
  (bc, bo) = cb
  Bk{p, l, bc, bo, bq}

def bk.back(+k: U32, b: Bk) -> Bk:
  Bk{p, l, bc, bo, bq} = b
  bk.offs(k, p, l, bq, back.go(nat(k), 0, (bc, bo)))

def bk.place(+b0: U32, +bs: U32, +k: U32, b: Bk) -> Bk:
  Bk{p, l, bc, bo, bq} = b
  bk.back(k, put.go(nat(bs), b0, bk.offs(k, p, l, bq, off.go(nat(k), 0, 0, bo, Array.get(U32, bc, 0)))))

def bucket(+b0: U32, +bs: U32, +k: U32, b: Bk) -> Bk:
  Bk{p, l, bc, bo, bq} = b
  bk.place(b0, bs, k, cnt.go(nat(bs), b0, Bk{p, l, fill.u(nat(k), 0, 0, bc), bo, bq}))

# Update kernel (Algorithm 1, line 8, as _apply_statistics)
# ---------------------------------------------------------
# Lanes share x and the buckets; lane j writes only cluster j's slots of
# c, u, h, w (usage), s (batch sums), ps (last-pass sums) and pc (counts).

type Uk is Type:
  Uk{x: Array<F32>, bq: Array<U32>, bc: Array<U32>, bo: Array<U32>, c: Array<F32>, u: Array<F32>,
    h: Array<F32>, w: Array<F32>, s: Array<F32>, ps: Array<F32>, pc: Array<U32>}

def add.at(+i: U32, +v: F32, r: Array<F32> & F32) -> Array<F32>:
  (a, +o) = r
  Array.set(F32, a, i, (o + v : F32))

# s[si ..] += x[xi ..], n cells; rx holds x[xi].
def addrow(n: Nat, +xi: U32, +si: U32, s: Array<F32>, rx: Array<F32> & F32) -> Array<F32> & Array<F32>:
  match n:
    case 0n:
      (x, +v) = rx
      (x, s)
    case 1n+q:
      (x, +v) = rx
      addrow(q, (xi + 1 : U32), (si + 1 : U32), add.at(si, v, Array.get(F32, s, si)), Array.get(F32, x, (xi + 1 : U32)))

type XS is Type:
  XS{x: Array<F32>, bq: Array<U32>, s: Array<F32>}

def rows.add(+i: U32, +si: U32, +d: U32, bq: Array<U32>, xs: Array<F32> & Array<F32>) -> XS:
  (x, s) = xs
  XS{x, bq, s}

def rows.one(+r: U32, +i: U32, +si: U32, +d: U32, x: Array<F32>, s: Array<F32>, bq: Array<U32>) -> XS:
  rows.add(i, si, d, bq, addrow(nat(d), (r * d : U32), si, s, Array.get(F32, x, (r * d : U32))))

def rows.at(+i: U32, +si: U32, +d: U32, x: Array<F32>, s: Array<F32>, rq: Array<U32> & U32) -> XS:
  (bq, +r) = rq
  rows.one(r, i, si, d, x, s, bq)

# Add the rows bq[i .. i + n] into s[si ..].
def rows.go(n: Nat, +i: U32, +si: U32, +d: U32, xs: XS) -> XS:
  match n:
    case 0n:
      xs
    case 1n+q:
      XS{x, bq, s} = xs
      rows.go(q, (i + 1 : U32), si, d, rows.at(i, si, d, x, s, Array.get(U32, bq, i)))

# c[ci ..] moves to the batch mean: c + weight (mean - c), or the mean when
# the weight is exactly one.
def ema.val(one: Bool, +wt: F32, +mean: F32, +cv: F32) -> F32:
  match one:
    case True{}:
      mean
    case False{}:
      (cv + wt * (mean - cv) : F32)

def ema.at(+one: Bool, +ci: U32, +wt: F32, +mean: F32, r: Array<F32> & F32) -> Array<F32>:
  (c, +cv) = r
  Array.set(F32, c, ci, ema.val(one, wt, mean, cv))

def ema.go(n: Nat, +ci: U32, +cnt: F32, +wt: F32, +one: Bool, c: Array<F32>, rs: Array<F32> & F32) -> Array<F32> & Array<F32>:
  match n:
    case 0n:
      (s, +sv) = rs
      (s, c)
    case 1n+q:
      (s, +sv) = rs
      ema.go(q, (ci + 1 : U32), cnt, wt, one, ema.at(one, ci, wt, (sv / cnt : F32), Array.get(F32, c, ci)), Array.get(F32, s, (ci + 1 : U32)))

# ps[i ..] += s[i ..]
def pass.go(n: Nat, +i: U32, ps: Array<F32>, rs: Array<F32> & F32) -> Array<F32> & Array<F32>:
  match n:
    case 0n:
      (s, +v) = rs
      (s, ps)
    case 1n+q:
      (s, +v) = rs
      pass.go(q, (i + 1 : U32), add.at(i, v, Array.get(F32, ps, i)), Array.get(F32, s, (i + 1 : U32)))

# Step parameters of one update.
type Up is Data:
  Up{sph: Bool, d: U32, alpha: F32, decay: F32, rep: Bool, last: Bool}

# Cluster j: zero its sums, add its batch rows.
def u.sum3(+j: U32, +d: U32, bc: Array<U32>, bo: Array<U32>, c: Array<F32>, u: Array<F32>, h: Array<F32>,
  w: Array<F32>, ps: Array<F32>, pc: Array<U32>, xs: XS) -> Uk:
  XS{x, bq, s} = xs
  Uk{x, bq, bc, bo, c, u, h, w, s, ps, pc}

def u.sum2(+j: U32, +d: U32, +cnt: U32, x: Array<F32>, bq: Array<U32>, bc: Array<U32>, c: Array<F32>, u: Array<F32>,
  h: Array<F32>, w: Array<F32>, s: Array<F32>, ps: Array<F32>, pc: Array<U32>, ro: Array<U32> & U32) -> Uk:
  (bo, +off) = ro
  u.sum3(j, d, bc, bo, c, u, h, w, ps, pc, rows.go(nat(cnt), off, (j * d : U32), d, XS{x, bq, zero.f(nat(d), (j * d : U32), s)}))

def u.sum1(+j: U32, +d: U32, x: Array<F32>, bq: Array<U32>, bo: Array<U32>, c: Array<F32>, u: Array<F32>,
  h: Array<F32>, w: Array<F32>, s: Array<F32>, ps: Array<F32>, pc: Array<U32>, rc: Array<U32> & U32) -> Uk:
  (bc, +cnt) = rc
  u.sum2(j, d, cnt, x, bq, bc, c, u, h, w, s, ps, pc, Array.get(U32, bo, j))

def u.sum(+j: U32, +d: U32, k: Uk) -> Uk:
  Uk{x, bq, bc, bo, c, u, h, w, s, ps, pc} = k
  u.sum1(j, d, x, bq, bo, c, u, h, w, s, ps, pc, Array.get(U32, bc, j))

# Cluster j: its usage decays; when it got rows, its mass grows and it moves.
def u.ema4(x: Array<F32>, bq: Array<U32>, bc: Array<U32>, bo: Array<U32>, u: Array<F32>, h: Array<F32>,
  w: Array<F32>, ps: Array<F32>, pc: Array<U32>, sc: Array<F32> & Array<F32>) -> Uk:
  (s, c) = sc
  Uk{x, bq, bc, bo, c, u, h, w, s, ps, pc}

def u.move(got: Bool, +j: U32, +d: U32, +cntf: F32, +alpha: F32, +retained: F32, +decayed: F32, x: Array<F32>, bq: Array<U32>,
  bc: Array<U32>, bo: Array<U32>, c: Array<F32>, u: Array<F32>, h: Array<F32>, w: Array<F32>, s: Array<F32>,
  ps: Array<F32>, pc: Array<U32>) -> Uk:
  match got:
    case False{}:
      Uk{x, bq, bc, bo, c, u, h, Array.set(F32, w, j, decayed), s, ps, pc}
    case True{}:
      +mass = (retained + alpha * cntf : F32)
      +wt = (alpha * cntf / mass : F32)
      +i = (j * d : U32)
      u.ema4(x, bq, bc, bo, u, h, Array.set(F32, w, j, mass), ps, pc,
        ema.go(nat(d), i, cntf, wt, F32.is_eq(wt, 1.0), c, Array.get(F32, s, i)))

def u.ema2(+j: U32, +p: Up, +cnt: U32, x: Array<F32>, bq: Array<U32>, bc: Array<U32>, bo: Array<U32>, c: Array<F32>,
  u: Array<F32>, h: Array<F32>, s: Array<F32>, ps: Array<F32>, pc: Array<U32>, rw: Array<F32> & F32) -> Uk:
  Up{sph, d, alpha, decay, +rep, last} = p
  (w, +wj) = rw
  +decayed = pickf(rep, wj, (wj * decay : F32))
  +retained = pickf(rep, 0.0, decayed)
  u.move(U32.is_ne(cnt, 0), j, d, U32.to_f32(cnt), alpha, retained, decayed, x, bq, bc, bo, c, u, h, w, s, ps, pc)

def u.ema1(+j: U32, +p: Up, x: Array<F32>, bq: Array<U32>, bo: Array<U32>, c: Array<F32>, u: Array<F32>,
  h: Array<F32>, w: Array<F32>, s: Array<F32>, ps: Array<F32>, pc: Array<U32>, rc: Array<U32> & U32) -> Uk:
  (bc, +cnt) = rc
  u.ema2(j, p, cnt, x, bq, bc, bo, c, u, h, s, ps, pc, Array.get(F32, w, j))

def u.ema(+j: U32, +p: Up, k: Uk) -> Uk:
  Uk{x, bq, bc, bo, c, u, h, w, s, ps, pc} = k
  u.ema1(j, p, x, bq, bo, c, u, h, w, s, ps, pc, Array.get(U32, bc, j))

# Cluster j's pass statistics: its count, and on the last pass its sums.
def u.pass3(x: Array<F32>, bq: Array<U32>, bc: Array<U32>, bo: Array<U32>, c: Array<F32>, u: Array<F32>,
  h: Array<F32>, w: Array<F32>, pc: Array<U32>, sp: Array<F32> & Array<F32>) -> Uk:
  (s, ps) = sp
  Uk{x, bq, bc, bo, c, u, h, w, s, ps, pc}

def u.pass2(last: Bool, +j: U32, +d: U32, x: Array<F32>, bq: Array<U32>, bc: Array<U32>, bo: Array<U32>, c: Array<F32>,
  u: Array<F32>, h: Array<F32>, w: Array<F32>, s: Array<F32>, ps: Array<F32>, pc: Array<U32>) -> Uk:
  match last:
    case False{}:
      Uk{x, bq, bc, bo, c, u, h, w, s, ps, pc}
    case True{}:
      +i = (j * d : U32)
      u.pass3(x, bq, bc, bo, c, u, h, w, pc, pass.go(nat(d), i, ps, Array.get(F32, s, i)))

def addu.at(+i: U32, +v: U32, r: Array<U32> & U32) -> Array<U32>:
  (a, +o) = r
  Array.set(U32, a, i, (o + v : U32))

def u.pass1(+last: Bool, +j: U32, +d: U32, +cnt: U32, x: Array<F32>, bq: Array<U32>, bo: Array<U32>, c: Array<F32>,
  u: Array<F32>, h: Array<F32>, w: Array<F32>, s: Array<F32>, ps: Array<F32>, pc: Array<U32>, bc: Array<U32>) -> Uk:
  u.pass2(last, j, d, x, bq, bc, bo, c, u, h, w, s, ps, addu.at(j, cnt, Array.get(U32, pc, j)))

def u.pass0(+last: Bool, +j: U32, +d: U32, x: Array<F32>, bq: Array<U32>, bo: Array<U32>, c: Array<F32>,
  u: Array<F32>, h: Array<F32>, w: Array<F32>, s: Array<F32>, ps: Array<F32>, pc: Array<U32>, rc: Array<U32> & U32) -> Uk:
  (bc, +cnt) = rc
  u.pass1(last, j, d, cnt, x, bq, bo, c, u, h, w, s, ps, pc, bc)

def u.pass(+last: Bool, +j: U32, +d: U32, k: Uk) -> Uk:
  Uk{x, bq, bc, bo, c, u, h, w, s, ps, pc} = k
  u.pass0(last, j, d, x, bq, bo, c, u, h, w, s, ps, pc, Array.get(U32, bc, j))

def u.read1(x: Array<F32>, bq: Array<U32>, bc: Array<U32>, bo: Array<U32>, w: Array<F32>, s: Array<F32>,
  ps: Array<F32>, pc: Array<U32>, cu: CU) -> Uk:
  CU{c, u, h} = cu
  Uk{x, bq, bc, bo, c, u, h, w, s, ps, pc}

def u.read(+sph: Bool, +j: U32, +d: U32, k: Uk) -> Uk:
  Uk{x, bq, bc, bo, c, u, h, w, s, ps, pc} = k
  u.read1(x, bq, bc, bo, w, s, ps, pc, readout(sph, j, d, CU{c, u, h}))

def u.one(+j: U32, +p: Up, k: Uk) -> Uk:
  Up{+sph, +d, alpha, decay, rep, +last} = p
  u.read(sph, j, d, u.pass(last, j, d, u.ema(j, p, u.sum(j, d, k))))

def u.go(n: Nat, +j: U32, +p: Up, k: Uk) -> Uk:
  match n:
    case 0n:
      k
    case 1n+q:
      u.go(q, (j + 1 : U32), p, u.one(j, p, k))

@unsafe
def uk.join(a: Uk, b: Uk) -> Uk:
  Uk{x, bq, bc, bo, c, u, h, w, s, ps, pc} = a
  Uk{x2, bq2, bc2, bo2, c2, u2, h2, w2, s2, ps2, pc2} = b
  Uk{Array.join(F32, x, x2), Array.join(U32, bq, bq2), Array.join(U32, bc, bc2), Array.join(U32, bo, bo2),
    Array.join(F32, c, c2), Array.join(F32, u, u2), Array.join(F32, h, h2), Array.join(F32, w, w2),
    Array.join(F32, s, s2), Array.join(F32, ps, ps2), Array.join(U32, pc, pc2)}

# Clusters j0 .. j0 + cnt, split dep more times.
@unsafe
def utree(+dep: Nat, +j0: U32, +cnt: U32, +p: Up, +x: Array<F32>, +bq: Array<U32>, +bc: Array<U32>, +bo: Array<U32>,
  +c: Array<F32>, +u: Array<F32>, +h: Array<F32>, +w: Array<F32>, +s: Array<F32>, +ps: Array<F32>, +pc: Array<U32>) -> Uk:
  match dep:
    case 0n:
      u.go(nat(cnt), j0, p, Uk{x, bq, bc, bo, c, u, h, w, s, ps, pc})
    case 1n+q:
      +m = U32.shr(cnt)
      a b = utree(q, j0, m, p, x, bq, bc, bo, c, u, h, w, s, ps, pc) utree(q, (j0 + m : U32), (cnt - m : U32), p, x, bq, bc, bo, c, u, h, w, s, ps, pc)
      uk.join(a, b)

# Revival (Algorithm 1, line 9; _choose_splits, _apply_splits)
# ------------------------------------------------------------
# Every five updates, clusters whose usage is below tau * mean(usage) take
# half of a donor: donors are drawn without replacement in proportion to
# their usage (Gumbel top-k), and the pair splits by +-1e-4 per coordinate.

type Scan is Type:
  Scan{w: Array<F32>, key: Array<F32>, tg: Array<U32>, nu: U32, nd: U32}

# w[j ..]: unused ids to tg in order, donors' Gumbel keys to key (others -inf).
def rv.scan(n: Nat, +j: U32, +thr: F32, +rkey: U32, +nu: U32, +nd: U32, key: Array<F32>, tg: Array<U32>, rw: Array<F32> & F32) -> Scan:
  match n:
    case 0n:
      (w, +wj) = rw
      Scan{w, key, tg, nu, nd}
    case 1n+q:
      (w, +wj) = rw
      +unused = (wj < thr : F32)
      +donor = Bool.and(Bool.not(unused), (wj > 0.0 : F32))
      +g = (F32.log(wj) - F32.log((0.0 - F32.log(unif(rnd(rkey, j))) : F32)) : F32)
      rv.scan(q, (j + 1 : U32), thr, rkey, picku(unused, (nu + 1 : U32), nu), picku(donor, (nd + 1 : U32), nd),
        Array.set(F32, key, j, pickf(donor, g, ninf())), Array.set(U32, tg, nu, j), Array.get(F32, w, (j + 1 : U32)))

# The first index of the largest key.
def amx.go(n: Nat, +j: U32, +bv: F32, +bj: U32, r: Array<F32> & F32) -> Array<F32> & U32:
  match n:
    case 0n:
      (a, +v) = r
      (a, picku((v > bv : F32), j, bj))
    case 1n+q:
      (a, +v) = r
      +better = (v > bv : F32)
      amx.go(q, (j + 1 : U32), pickf(better, v, bv), picku(better, j, bj), Array.get(F32, a, (j + 1 : U32)))

def f32.up(+v: F32) -> F32:
  +b = F32.bits(v)
  f32.of_bits(picku(F32.is_eq(v, 0.0), 1, picku((v > 0.0 : F32), (b + 1 : U32), (b - 1 : U32))))

# Donor and child coordinates, c[si + i] * (1 -+ 1e-4) and c[ti + i] * (1 +- 1e-4),
# the sign alternating from + at even coordinates; same says all pairs are equal.
type Sp is Type:
  Sp{c: Array<F32>, same: Bool}

def sp.go(n: Nat, +i: U32, +si: U32, +ti: U32, +same: Bool, rc: Array<F32> & F32) -> Sp:
  match n:
    case 0n:
      (c, +v) = rc
      Sp{c, same}
    case 1n+q:
      (c, +v) = rc
      +off = pickf(U32.is_zero((i .&. 1 : U32)), 0.0001, (0.0 - 0.0001 : F32))
      +dv = (v * (1.0 - off) : F32)
      +cv = (v * (1.0 + off) : F32)
      sp.go(q, (i + 1 : U32), si, ti, Bool.and(same, F32.is_eq(dv, cv)),
        Array.get(F32, Array.set(F32, Array.set(F32, c, (si + i : U32), dv), (ti + i : U32), cv), (si + i + 1 : U32)))

def sp.nudge.at(+ti: U32, r: Array<F32> & F32) -> Array<F32>:
  (c, +v) = r
  Array.set(F32, c, ti, f32.up(v))

def sp.nudge.if(same: Bool, +ti: U32, c: Array<F32>) -> Array<F32>:
  match same:
    case True{}:
      sp.nudge.at(ti, Array.get(F32, c, ti))
    case False{}:
      c

# _separate_splits: a child equal to its donor moves one float up.
def sp.nudge(+ti: U32, sp: Sp) -> Array<F32>:
  Sp{c, same} = sp
  sp.nudge.if(same, ti, c)

type Rv is Type:
  Rv{c: Array<F32>, w: Array<F32>, key: Array<F32>, tg: Array<U32>}

def rv.usage(+src: U32, +tgt: U32, c: Array<F32>, key: Array<F32>, tg: Array<U32>, rw: Array<F32> & F32) -> Rv:
  (w, +ws) = rw
  +half = (ws / 2.0 : F32)
  Rv{c, Array.set(F32, Array.set(F32, w, src, half), tgt, half), key, tg}

def rv.apply(+src: U32, +d: U32, c: Array<F32>, w: Array<F32>, key: Array<F32>, rt: Array<U32> & U32) -> Rv:
  (tg, +tgt) = rt
  rv.usage(src, tgt, sp.nudge((tgt * d : U32), sp.go(nat(d), 0, (src * d : U32), (tgt * d : U32), True{}, Array.get(F32, c, (src * d : U32)))),
    Array.set(F32, key, src, ninf()), tg, Array.get(F32, w, src))

def rv.pick(+i: U32, +d: U32, c: Array<F32>, w: Array<F32>, tg: Array<U32>, rk: Array<F32> & U32) -> Rv:
  (key, +src) = rk
  rv.apply(src, d, c, w, key, Array.get(U32, tg, i))

# Split i: the donor with the largest remaining key, and the i-th unused.
def rv.one(+i: U32, +d: U32, +k: U32, rv: Rv) -> Rv:
  Rv{c, w, key, tg} = rv
  rv.pick(i, d, c, w, tg, amx.go(nat((k - 1 : U32)), 0, ninf(), 0, Array.get(F32, key, 0)))

def rv.split(n: Nat, +i: U32, +d: U32, +k: U32, rv: Rv) -> Rv:
  match n:
    case 0n:
      rv
    case 1n+q:
      rv.split(q, (i + 1 : U32), d, k, rv.one(i, d, k, rv))

def rv.scanned(+d: U32, +k: U32, c: Array<F32>, sc: Scan) -> Rv & U32:
  Scan{w, key, tg, +nu, +nd} = sc
  +ns = U32.min(nu, nd)
  (rv.split(nat(ns), 0, d, k, Rv{c, w, key, tg}), ns)

def rv.thr(+d: U32, +k: U32, +coef: F32, +rkey: U32, c: Array<F32>, key: Array<F32>, tg: Array<U32>, rs: Array<F32> & F32) -> Rv & U32:
  (w, +total) = rs
  +thr = ((total * coef) / U32.to_f32(k) : F32)
  rv.scanned(d, k, c, rv.scan(nat(k), 0, thr, rkey, 0, 0, key, tg, Array.get(F32, w, 0)))

# One revival check: the arrays, and how many clusters it revived.
def revive(+d: U32, +k: U32, +coef: F32, +rkey: U32, c: Array<F32>, w: Array<F32>, key: Array<F32>, tg: Array<U32>) -> Rv & U32:
  rv.thr(d, k, coef, rkey, c, key, tg, sum.f(w, 0, k))

# Training step
# -------------

type St is Type:
  St{x: Array<F32>, xn: Array<F32>, c: Array<F32>, u: Array<F32>, h: Array<F32>, w: Array<F32>,
    p: Array<U32>, l: Array<U32>, bc: Array<U32>, bo: Array<U32>, bq: Array<U32>,
    s: Array<F32>, ps: Array<F32>, pc: Array<U32>, key: Array<F32>, tg: Array<U32>}

# n, d, k, the batch, spherical, seed, alpha0, total steps, the revival
# threshold tau, the assignment and update fork depths, and the GPU flag.
type Cfg is Data:
  Cfg{n: U32, d: U32, k: U32, b: U32, sph: Bool, seed: U32, a0: F32, total: U32, coef: F32, ad: Nat, cd: Nat, kd: Nat, gpu: Bool}

type So is Type:
  So{st: St, obj: F32, ns: U32}

@unsafe
def assign(gpu: Bool, +ad: Nat, +cd: Nat, +sph: Bool, +b0: U32, +bs: U32, +k: U32, +d: U32, x: Array<F32>, xn: Array<F32>,
  u: Array<F32>, h: Array<F32>, p: Array<U32>, l: Array<U32>) -> Ak:
  match gpu:
    case True{}:
      atree!(ad, cd, sph, b0, bs, k, d, x, xn, u, h, p, l)
    case False{}:
      atree(ad, cd, sph, b0, bs, k, d, x, xn, u, h, p, l)

def step.f(+o: F32, +ns: U32, x: Array<F32>, xn: Array<F32>, w: Array<F32>, p: Array<U32>, l: Array<U32>,
  bc: Array<U32>, bo: Array<U32>, bq: Array<U32>, s: Array<F32>, ps: Array<F32>, pc: Array<U32>, key: Array<F32>,
  tg: Array<U32>, cu: CU) -> So:
  CU{c, u, h} = cu
  So{St{x, xn, c, u, h, w, p, l, bc, bo, bq, s, ps, pc, key, tg}, o, ns}

def step.e(any: Bool, +sph: Bool, +k: U32, +d: U32, +o: F32, +ns: U32, x: Array<F32>, xn: Array<F32>, c: Array<F32>,
  u: Array<F32>, h: Array<F32>, w: Array<F32>, p: Array<U32>, l: Array<U32>, bc: Array<U32>, bo: Array<U32>,
  bq: Array<U32>, s: Array<F32>, ps: Array<F32>, pc: Array<U32>, key: Array<F32>, tg: Array<U32>) -> So:
  match any:
    case False{}:
      So{St{x, xn, c, u, h, w, p, l, bc, bo, bq, s, ps, pc, key, tg}, o, ns}
    case True{}:
      step.f(o, ns, x, xn, w, p, l, bc, bo, bq, s, ps, pc, key, tg, readout.all(nat(k), sph, 0, d, CU{c, u, h}))

def step.d(+sph: Bool, +k: U32, +d: U32, +o: F32, x: Array<F32>, xn: Array<F32>, u: Array<F32>, h: Array<F32>,
  p: Array<U32>, l: Array<U32>, bc: Array<U32>, bo: Array<U32>, bq: Array<U32>, s: Array<F32>, ps: Array<F32>,
  pc: Array<U32>, rr: Rv & U32) -> So:
  (rv, +ns) = rr
  Rv{c, w, key, tg} = rv
  step.e(U32.is_ne(ns, 0), sph, k, d, o, ns, x, xn, c, u, h, w, p, l, bc, bo, bq, s, ps, pc, key, tg)

def step.r(due: Bool, +cf: Cfg, +stp: U32, +o: F32, xn: Array<F32>, p: Array<U32>, l: Array<U32>, key: Array<F32>,
  tg: Array<U32>, x: Array<F32>, bq: Array<U32>, bc: Array<U32>, bo: Array<U32>, c: Array<F32>, u: Array<F32>,
  h: Array<F32>, w: Array<F32>, s: Array<F32>, ps: Array<F32>, pc: Array<U32>) -> So:
  match due:
    case False{}:
      So{St{x, xn, c, u, h, w, p, l, bc, bo, bq, s, ps, pc, key, tg}, o, 0}
    case True{}:
      Cfg{n, +d, +k, b, +sph, +seed, a0, total, +coef, ad, cd, kd, gpu} = cf
      +rkey = mix(((seed .^. 2246822507) + stp * 3266489909 : U32))
      step.d(sph, k, d, o, x, xn, u, h, p, l, bc, bo, bq, s, ps, pc, revive(d, k, coef, rkey, c, w, key, tg))

def step.c(+cf: Cfg, +stp: U32, +o: F32, xn: Array<F32>, p: Array<U32>, l: Array<U32>, key: Array<F32>, tg: Array<U32>, uk: Uk) -> So:
  Uk{x, bq, bc, bo, c, u, h, w, s, ps, pc} = uk
  step.r(U32.is_zero(((stp + 1) % 5 : U32)), cf, stp, o, xn, p, l, key, tg, x, bq, bc, bo, c, u, h, w, s, ps, pc)

@unsafe
def step.b(+cf: Cfg, +stp: U32, +last: Bool, +alpha: F32, +o: F32, x: Array<F32>, xn: Array<F32>, c: Array<F32>,
  u: Array<F32>, h: Array<F32>, w: Array<F32>, s: Array<F32>, ps: Array<F32>, pc: Array<U32>, key: Array<F32>,
  tg: Array<U32>, bk: Bk) -> So:
  Cfg{n, +d, +k, b, +sph, seed, a0, total, coef, ad, cd, +kd, gpu} = cf
  Bk{p, l, bc, bo, bq} = bk
  +up = {Up{sph, d, alpha, (1.0 - alpha : F32), F32.is_eq(alpha, 1.0), last} : Up}
  step.c(cf, stp, o, xn, p, l, key, tg, utree(kd, 0, k, up, x, bq, bc, bo, c, u, h, w, s, ps, pc))

@unsafe
def step.a(+cf: Cfg, +stp: U32, +b0: U32, +bs: U32, +last: Bool, +alpha: F32, c: Array<F32>, w: Array<F32>,
  bc: Array<U32>, bo: Array<U32>, bq: Array<U32>, s: Array<F32>, ps: Array<F32>, pc: Array<U32>, key: Array<F32>,
  tg: Array<U32>, ak: Ak) -> So:
  Cfg{n, d, +k, b, sph, seed, a0, total, coef, ad, cd, kd, gpu} = cf
  Ak{x, xn, u, h, p, l, +o} = ak
  step.b(cf, stp, last, alpha, o, x, xn, c, u, h, w, s, ps, pc, key, tg, bucket(b0, bs, k, Bk{p, l, bc, bo, bq}))

# One update: assign the batch at b0 (bs rows of the order), apply its
# statistics, and every fifth update revive unused clusters.
@unsafe
def step(+cf: Cfg, +stp: U32, +b0: U32, +bs: U32, +last: Bool, st: St) -> So:
  Cfg{n, +d, +k, b, +sph, seed, +a0, +total, coef, +ad, +cd, kd, +gpu} = cf
  St{x, xn, c, u, h, w, p, l, bc, bo, bq, s, ps, pc, key, tg} = st
  step.a(cf, stp, b0, bs, last, alpha.at(a0, stp, total), c, w, bc, bo, bq, s, ps, pc, key, tg,
    assign(gpu, ad, cd, sph, b0, bs, k, d, x, xn, u, h, p, l))

# Epochs (IO)
# -----------

type Ep is Type:
  Ep{st: St, stp: U32, o: F32, ns: U32}

type Stat is Data:
  Stat{obj: F32, ms: Nat, imb: F32, ns: U32}

@unsafe
def steps(+cf: Cfg, +stp: U32, +b0: U32, +rows: U32, +last: Bool, +o: F32, +comp: F32, +ns: U32, st: St) -> IO(Ep):
  steps.if(U32.is_lt(b0, rows), cf, stp, b0, rows, last, o, comp, ns, st)

@unsafe
def steps.if(more: Bool, +cf: Cfg, +stp: U32, +b0: U32, +rows: U32, +last: Bool, +o: F32, +comp: F32, +ns: U32, st: St) -> IO(Ep):
  match more:
    case False{}:
      IO.pure(Ep, Ep{st, stp, o, ns})
    case True{}:
      Cfg{n, d, k, +b, sph, seed, a0, total, coef, ad, cd, kd, gpu} = cf
      do IO<Ep>:
        so : So <- IO.pure(So, step(cf, stp, b0, U32.min(b, (rows - b0 : U32)), last, st))
        steps.k(cf, stp, (b0 + b : U32), rows, last, o, comp, ns, so)

# The epoch objective is a compensated (Kahan) sum of the batch objectives.
@unsafe
def steps.k(+cf: Cfg, +stp: U32, +b0: U32, +rows: U32, +last: Bool, +o: F32, +comp: F32, +ns: U32, so: So) -> IO(Ep):
  So{st, +ob, +n1} = so
  +y = (ob - comp : F32)
  +t = (o + y : F32)
  steps(cf, (stp + 1 : U32), b0, rows, last, t, ((t - o) - y : F32), (ns + n1 : U32), st)

def start.last(last: Bool, +n: U32, +kd: U32, ps: Array<F32>, l: Array<U32>) -> Array<F32> & Array<U32>:
  match last:
    case False{}:
      (ps, l)
    case True{}:
      (zero.f(nat(kd), 0, ps), fill.u(nat(n), 0, 4294967295, l))

def start.fin(x: Array<F32>, xn: Array<F32>, c: Array<F32>, u: Array<F32>, h: Array<F32>, w: Array<F32>, p: Array<U32>,
  bc: Array<U32>, bo: Array<U32>, bq: Array<U32>, s: Array<F32>, pc: Array<U32>, key: Array<F32>, tg: Array<U32>,
  pl: Array<F32> & Array<U32>) -> St:
  (ps, l) = pl
  St{x, xn, c, u, h, w, p, l, bc, bo, bq, s, ps, pc, key, tg}

# A new pass: a fresh order (seeded as _epoch_seed), zero counts, and on the
# last pass zero sums and no labels yet.
def start(+cf: Cfg, +e: U32, +last: Bool, st: St) -> St:
  Cfg{+n, +d, +k, b, sph, +seed, a0, total, coef, ad, cd, kd, gpu} = cf
  St{x, xn, c, u, h, w, p, l, bc, bo, bq, s, ps, pc, key, tg} = st
  start.fin(x, xn, c, u, h, w, shuffle(n, mix((seed * 1000003 + e : U32)), p), bc, bo, bq, s,
    fill.u(nat(k), 0, 0, pc), key, tg, start.last(last, n, (k * d : U32), ps, l))

def sq.u.go(n: Nat, +j: U32, +acc: F32, r: Array<U32> & U32) -> Array<U32> & F32:
  match n:
    case 0n:
      (a, +v) = r
      (a, acc)
    case 1n+q:
      (a, +v) = r
      +f = U32.to_f32(v)
      sq.u.go(q, (j + 1 : U32), (acc + f * f : F32), Array.get(U32, a, (j + 1 : U32)))

type Run is Type:
  Run{st: St, stats: List<&2, Stat>, stp: U32, ns: U32}

def list.empty(xs: List<&2, U32>) -> Bool:
  match xs:
    case []:
      True{}
    case _:
      False{}

@unsafe
def epochs(+cf: Cfg, +e: U32, rs: List<&2, U32>, stats: List<&2, Stat>, +t0: Nat, +stp: U32, +ns: U32, st: St) -> IO(Run):
  match rs:
    case []:
      IO.pure(Run, Run{st, stats, stp, ns})
    case +r <> +rest:
      +last = list.empty(rest)
      do IO<Run>:
        ep : Ep <- steps(cf, stp, 0, r, last, 0.0, 0.0, 0, start(cf, e, last, st))
        now : Nat <- IO.now()
        epochs.k(cf, e, r, rest, stats, t0, now, ns, ep)

@unsafe
def epochs.k(+cf: Cfg, +e: U32, +rows: U32, rest: List<&2, U32>, stats: List<&2, Stat>, +t0: Nat, +now: Nat, +ns0: U32, ep: Ep) -> IO(Run):
  Cfg{n, d, +k, b, sph, seed, a0, total, coef, ad, cd, kd, gpu} = cf
  Ep{st, +stp, +o, +ns} = ep
  St{x, xn, c, u, h, w, p, l, bc, bo, bq, s, ps, pc, key, tg} = st
  epochs.s(cf, e, rows, rest, stats, t0, now, ns0, stp, o, ns, x, xn, c, u, h, w, p, l, bc, bo, bq, s, ps, key, tg,
    sq.u.go(nat(k), 0, 0.0, Array.get(U32, pc, 0)))

@unsafe
def epochs.s(+cf: Cfg, +e: U32, +rows: U32, rest: List<&2, U32>, stats: List<&2, Stat>, +t0: Nat, +now: Nat, +ns0: U32,
  +stp: U32, +o: F32, +ns: U32, x: Array<F32>, xn: Array<F32>, c: Array<F32>, u: Array<F32>, h: Array<F32>,
  w: Array<F32>, p: Array<U32>, l: Array<U32>, bc: Array<U32>, bo: Array<U32>, bq: Array<U32>, s: Array<F32>,
  ps: Array<F32>, key: Array<F32>, tg: Array<U32>, rq: Array<U32> & F32) -> IO(Run):
  Cfg{n, d, +k, b, sph, seed, a0, total, coef, ad, cd, kd, gpu} = cf
  (pc, +sq) = rq
  +rf = U32.to_f32(rows)
  +imb = (U32.to_f32(k) * sq / (rf * rf) : F32)
  epochs(cf, (e + 1 : U32), rest, Stat{o, Nat.sub(now, t0), imb, ns} <> stats, t0, stp, (ns0 + ns : U32),
    St{x, xn, c, u, h, w, p, l, bc, bo, bq, s, ps, pc, key, tg})

# Closing mean (Algorithm 1, line 12; finalize, _spherical_closing)
# -----------------------------------------------------------------
# A centroid the last pass assigned rows to becomes their mean (normalized
# when spherical); the others keep their position. The result goes to u.

def div.go(n: Nat, +i: U32, +cnt: F32, s: Array<F32>, r: Array<F32> & F32) -> Array<F32> & Array<F32>:
  match n:
    case 0n:
      (ps, +v) = r
      (ps, s)
    case 1n+q:
      (ps, +v) = r
      div.go(q, (i + 1 : U32), cnt, Array.set(F32, s, i, (v / cnt : F32)), Array.get(F32, ps, (i + 1 : U32)))

def cp.go(n: Nat, +i: U32, s: Array<F32>, r: Array<F32> & F32) -> Array<F32> & Array<F32>:
  match n:
    case 0n:
      (c, +v) = r
      (c, s)
    case 1n+q:
      (c, +v) = r
      cp.go(q, (i + 1 : U32), Array.set(F32, s, i, v), Array.get(F32, c, (i + 1 : U32)))

type Fz is Type:
  Fz{c: Array<F32>, u: Array<F32>, h: Array<F32>, s: Array<F32>, ps: Array<F32>, pc: Array<U32>}

def fz.read(c: Array<F32>, ps: Array<F32>, pc: Array<U32>, cu: CU) -> Fz:
  CU{s, u, h} = cu
  Fz{c, u, h, s, ps, pc}

def fz.c(+sph: Bool, +j: U32, +d: U32, ps: Array<F32>, pc: Array<U32>, u: Array<F32>, h: Array<F32>, cs: Array<F32> & Array<F32>) -> Fz:
  (c, s) = cs
  fz.read(c, ps, pc, readout(sph, j, d, CU{s, u, h}))

def fz.p(+sph: Bool, +j: U32, +d: U32, c: Array<F32>, pc: Array<U32>, u: Array<F32>, h: Array<F32>, ps2: Array<F32> & Array<F32>) -> Fz:
  (ps, s) = ps2
  fz.read(c, ps, pc, readout(sph, j, d, CU{s, u, h}))

def fz.pick(got: Bool, +sph: Bool, +j: U32, +d: U32, +cnt: U32, c: Array<F32>, u: Array<F32>, h: Array<F32>, s: Array<F32>,
  ps: Array<F32>, pc: Array<U32>) -> Fz:
  match got:
    case True{}:
      +i = (j * d : U32)
      fz.p(sph, j, d, c, pc, u, h, div.go(nat(d), i, U32.to_f32(cnt), s, Array.get(F32, ps, i)))
    case False{}:
      +i = (j * d : U32)
      fz.c(sph, j, d, ps, pc, u, h, cp.go(nat(d), i, s, Array.get(F32, c, i)))

def fz.at(+sph: Bool, +j: U32, +d: U32, c: Array<F32>, u: Array<F32>, h: Array<F32>, s: Array<F32>, ps: Array<F32>, rc: Array<U32> & U32) -> Fz:
  (pc, +cnt) = rc
  fz.pick(U32.is_ne(cnt, 0), sph, j, d, cnt, c, u, h, s, ps, pc)

def fz.one(+sph: Bool, +j: U32, +d: U32, f: Fz) -> Fz:
  Fz{c, u, h, s, ps, pc} = f
  fz.at(sph, j, d, c, u, h, s, ps, Array.get(U32, pc, j))

def fz.go(n: Nat, +sph: Bool, +j: U32, +d: U32, f: Fz) -> Fz:
  match n:
    case 0n:
      f
    case 1n+q:
      fz.go(q, sph, (j + 1 : U32), d, fz.one(sph, j, d, f))

# Initial centroids (train): k distinct random rows, normalized when spherical
# ----------------------------------------------------------------------------

# dst[di ..] = src[si ..], n cells; r holds src[si].
def cp2.go(n: Nat, +di: U32, +si: U32, dst: Array<F32>, r: Array<F32> & F32) -> Array<F32> & Array<F32>:
  match n:
    case 0n:
      (src, +v) = r
      (src, dst)
    case 1n+q:
      (src, +v) = r
      cp2.go(q, (di + 1 : U32), (si + 1 : U32), Array.set(F32, dst, di, v), Array.get(F32, src, (si + 1 : U32)))

type In is Type:
  In{x: Array<F32>, c: Array<F32>, p: Array<U32>}

def init.fin(p: Array<U32>, xc: Array<F32> & Array<F32>) -> In:
  (x, c) = xc
  In{x, c, p}

def init.at(+j: U32, +d: U32, x: Array<F32>, c: Array<F32>, pr: Array<U32> & U32) -> In:
  (p, +r) = pr
  init.fin(p, cp2.go(nat(d), (j * d : U32), (r * d : U32), c, Array.get(F32, x, (r * d : U32))))

def init.one(+j: U32, +d: U32, i: In) -> In:
  In{x, c, p} = i
  init.at(j, d, x, c, Array.get(U32, p, j))

def init.go(n: Nat, +j: U32, +d: U32, i: In) -> In:
  match n:
    case 0n:
      i
    case 1n+q:
      init.go(q, (j + 1 : U32), d, init.one(j, d, i))

# Squared row norms, for the Euclidean objective.
def nrm.put(+r: U32, xn: Array<F32>, rs: Array<F32> & F32) -> Array<F32> & Array<F32>:
  (x, +v) = rs
  (x, Array.set(F32, xn, r, v))

def nrm.go(n: Nat, +r: U32, +d: U32, xx: Array<F32> & Array<F32>) -> Array<F32> & Array<F32>:
  match n:
    case 0n:
      xx
    case 1n+q:
      (x, xn) = xx
      nrm.go(q, (r + 1 : U32), d, nrm.put(r, xn, sq.go(nat(d), (r * d : U32), 1.0, 0.0, Array.get(F32, x, (r * d : U32)))))

# Report
# ------

def json.bool(b: Bool) -> String:
  match b:
    case True{}:
      "true"
    case False{}:
      "false"

def json.f(x: F32) -> String:
  F32.show(x)

def stats.rev(xs: List<&2, Stat>, acc: List<&2, Stat>) -> List<&2, Stat>:
  match xs:
    case []:
      acc
    case x <> t:
      stats.rev(t, x <> acc)

def stats.obj(xs: List<&2, Stat>) -> List<&2, String>:
  match xs:
    case []:
      []
    case Stat{o, ms, imb, ns} <> t:
      json.f(o) <> stats.obj(t)

def stats.ms(xs: List<&2, Stat>) -> List<&2, String>:
  match xs:
    case []:
      []
    case Stat{o, ms, imb, ns} <> t:
      Nat.show(ms) <> stats.ms(t)

def stats.imb(xs: List<&2, Stat>) -> List<&2, String>:
  match xs:
    case []:
      []
    case Stat{o, ms, imb, ns} <> t:
      json.f(imb) <> stats.imb(t)

def stats.ns(xs: List<&2, Stat>) -> List<&2, String>:
  match xs:
    case []:
      []
    case Stat{o, ms, imb, ns} <> t:
      U32.show(ns) <> stats.ns(t)

def arr(xs: List<&2, String>) -> String:
  "[" ++ String.join(xs, ", ") ++ "]"

def backend.str(gpu: Bool) -> String:
  match gpu:
    case True{}:
      "metal-gpu"
    case False{}:
      "c-cpu"

def law.str(quality: Bool) -> String:
  match quality:
    case True{}:
      "quality"
    case False{}:
      "throughput"

type Meta is Data:
  Meta{tstr: String, rows: List<&2, U32>, steps: List<&2, U32>, spp: U32, quality: Bool, load: Nat, init: Nat, train: Nat, fin: Nat}

def report(+cf: Cfg, +m: Meta, +stats: List<&2, Stat>, +nsplit: U32) -> String:
  Cfg{n, d, k, b, sph, seed, a0, total, coef, ad, cd, kd, gpu} = cf
  Meta{tstr, rows, steps, spp, quality, load, init, train, fin} = m
  "{\n" ++
  "  \"impl\": \"bend\",\n" ++
  "  \"backend\": \"" ++ backend.str(gpu) ++ "\",\n" ++
  "  \"n\": " ++ U32.show(n) ++ ", \"d\": " ++ U32.show(d) ++ ", \"k\": " ++ U32.show(k) ++
  ", \"nepochs\": " ++ tstr ++ ", \"seed\": " ++ U32.show(seed) ++ ", \"spherical\": " ++ json.bool(sph) ++ ",\n" ++
  "  \"recipe\": {\"batch_law\": \"" ++ law.str(quality) ++ "\", \"batch_size\": " ++ U32.show(b) ++
  ", \"alpha0\": " ++ json.f(a0) ++ ", \"alpha0_bits\": " ++ U32.show(F32.bits(a0)) ++
  ", \"steps_per_pass\": " ++ U32.show(spp) ++ ", \"total_steps\": " ++ U32.show(total) ++
  ", \"epoch_rows\": " ++ list.show(rows) ++ ", \"epoch_steps\": " ++ list.show(steps) ++
  ", \"unused_threshold\": " ++ json.f(coef) ++ "},\n" ++
  "  \"obj\": " ++ arr(stats.obj(stats)) ++ ",\n" ++
  "  \"time_ms\": " ++ arr(stats.ms(stats)) ++ ",\n" ++
  "  \"imbalance\": " ++ arr(stats.imb(stats)) ++ ",\n" ++
  "  \"nsplit\": " ++ arr(stats.ns(stats)) ++ ", \"n_splits\": " ++ U32.show(nsplit) ++ ",\n" ++
  "  \"load_ms\": " ++ Nat.show(load) ++ ", \"init_ms\": " ++ Nat.show(init) ++
  ", \"train_ms\": " ++ Nat.show(train) ++ ", \"finalize_ms\": " ++ Nat.show(fin) ++ "\n}\n"

# Main
# ----

type Args is Data:
  Args{path: String, n: U32, d: U32, k: U32, tstr: String, t: F32, seed: U32, sph: Bool, out: String, gpu: Bool}

def u32(s: String) -> U32:
  Maybe.default(&2, U32, U32.read(s), 0)

def f32(s: String) -> F32:
  Maybe.default(&2, F32, F32.read(s), 0.0)

def gpu.arg(rest: List<String>) -> Bool:
  match rest:
    case []:
      False{}
    case h <> t:
      U32.is_eq(u32(h), 1)

# On the GPU, rows times centroid blocks fill about 2^14 lanes.
def gpu.cd(gpu: Bool, +ad: Nat, +k: U32) -> Nat:
  match gpu:
    case False{}:
      0n
    case True{}:
      Nat.min(U32.log2(k), Nat.sub(14n, ad))

def cfg.of(+a: Args) -> Cfg:
  Args{path, +n, +d, +k, tstr, +t, seed, sph, out, +gpu} = a
  +quality = U32.is_le(d, 16)
  +b = batch.size(quality, n, k, t)
  +ad = depth(picku(gpu, b, U32.min(b, 256)))
  Cfg{n, d, k, b, sph, seed, alpha0(n, k, t, b), list.sum(steps.of(epoch.rows(n, t), b), 0),
    pickf(U32.is_le(d, 2), 0.0, 0.1), ad, gpu.cd(gpu, ad, k), depth(U32.min(k, 256)), gpu}

def meta.of(+a: Args, +cf: Cfg, +load: Nat, +init: Nat, +train: Nat, +fin: Nat) -> Meta:
  Args{path, +n, +d, k, tstr, +t, seed, sph, out, gpu} = a
  Cfg{n2, d2, k2, +b, sph2, seed2, a0, total, coef, ad, cd, kd, gpu2} = cf
  +rows = epoch.rows(n, t)
  Meta{tstr, rows, steps.of(rows, b), ceil.div(n, b), U32.is_le(d, 16), load, init, train, fin}

def st.make(+n: U32, +d: U32, +k: U32, +b: U32, x: Array<F32>, xn: Array<F32>, p: Array<U32>, c: Array<F32>, u: Array<F32>, h: Array<F32>) -> St:
  +kd = depth((k * d : U32))
  +kk = depth(k)
  St{x, xn, c, u, h, Array.new(F32, kk, 1.5258789e-05), p, Array.new(U32, depth(n), 4294967295),
    Array.new(U32, kk, 0), Array.new(U32, kk, 0), Array.new(U32, depth(b), 0),
    Array.new(F32, kd, 0.0), Array.new(F32, kd, 0.0), Array.new(U32, kk, 0), Array.new(F32, kk, 0.0), Array.new(U32, kk, 0)}

def st.sph(+n: U32, +d: U32, +k: U32, +b: U32, x: Array<F32>, p: Array<U32>, h: Array<F32>, uc: Array<F32> & Array<F32>) -> St:
  (u, c) = uc
  st.make(n, d, k, b, x, Array.new(F32, 0n, 0.0), p, c, u, h)

def st.euc(+n: U32, +d: U32, +k: U32, +b: U32, p: Array<U32>, c: Array<F32>, u: Array<F32>, h: Array<F32>, xx: Array<F32> & Array<F32>) -> St:
  (x, xn) = xx
  st.make(n, d, k, b, x, xn, p, c, u, h)

# The starting state: arrays sized, centroids drawn from the rows.
def st.fin(sph: Bool, +n: U32, +d: U32, +k: U32, +b: U32, x: Array<F32>, p: Array<U32>, cu: CU) -> St:
  match sph:
    case True{}:
      CU{c, u, h} = cu
      st.sph(n, d, k, b, x, p, h, cp.go(nat((k * d : U32)), 0, c, Array.get(F32, u, 0)))
    case False{}:
      CU{c, u, h} = cu
      st.euc(n, d, k, b, p, c, u, h, nrm.go(nat(n), 0, d, (x, Array.new(F32, depth(n), 0.0))))

def st.drawn(+sph: Bool, +n: U32, +d: U32, +k: U32, +b: U32, i: In) -> St:
  In{x, c, p} = i
  st.fin(sph, n, d, k, b, x, p, readout.all(nat(k), sph, 0, d, CU{c, Array.new(F32, depth((k * d : U32)), 0.0), Array.new(F32, depth(k), 0.0)}))

def st.init(+cf: Cfg, x: Array<F32>) -> St:
  Cfg{+n, +d, +k, +b, +sph, +seed, a0, total, coef, ad, cd, kd, gpu} = cf
  p0 = choose(nat(k), 0, n, mix((seed .^. 625341585 : U32)), iota(nat(n), 0, Array.new(U32, depth(n), 0)))
  st.drawn(sph, n, d, k, b, init.go(nat(k), 0, d, In{x, Array.new(F32, depth((k * d : U32)), 0.0), p0}))

def fin.state(+cf: Cfg, st: St) -> Fz:
  Cfg{n, +d, +k, b, +sph, seed, a0, total, coef, ad, cd, kd, gpu} = cf
  St{x, xn, c, u, h, w, p, l, bc, bo, bq, s, ps, pc, key, tg} = st
  fz.go(nat(k), sph, 0, d, Fz{c, u, h, s, ps, pc})

type Out is Type:
  Out{l: Array<U32>, u: Array<F32>}

def out.fz(l: Array<U32>, f: Fz) -> Out:
  Fz{c, u, h, s, ps, pc} = f
  Out{l, u}

def out.of(+cf: Cfg, st: St) -> Out:
  St{x, xn, c, u, h, w, p, l, bc, bo, bq, s, ps, pc, key, tg} = st
  out.fz(l, fin.state(cf, St{x, xn, c, u, h, w, p, Array.new(U32, 0n, 0), bc, bo, bq, s, ps, pc, key, tg}))

@unsafe
def main.write(+a: Args, +cf: Cfg, +m: Meta, +stats: List<&2, Stat>, +ns: U32, o: Out) -> IO(Unit):
  Args{path, +n, +d, +k, tstr, t, seed, sph, +out, gpu} = a
  Out{l, u} = o
  do IO<Unit>:
    fl : File <- IO.try(File, File.open(out ++ "/labels.i32", "w"))
    write.u(fl, 0, n, l)
    fc : File <- IO.try(File, File.open(out ++ "/centroids.f32", "w"))
    write.f(fc, 0, (k * d : U32), u)
    fr : File <- IO.try(File, File.open(out ++ "/report.json", "w"))
    pr : File & Result<&1, &1, U32 & String, Unit> <- File.write(fr, report(cf, m, stats, ns))
    fr2 : File <- wrote(pr)
    File.close(fr2)

def last.obj(xs: List<&2, Stat>) -> F32:
  match xs:
    case []:
      0.0
    case Stat{o, ms, imb, ns} <> []:
      o
    case x <> t:
      last.obj(t)

@unsafe
def main.done(+a: Args, +cf: Cfg, +t0: Nat, +t1: Nat, +t2: Nat, +t3: Nat, r: Run) -> IO(Unit):
  Run{st, stats, stp, +ns} = r
  +stats2 = stats.rev(stats, [])
  do IO<Unit>:
    +t4 : Nat <- IO.now()
    o : Out <- IO.pure(Out, out.of(cf, st))
    +t5 : Nat <- IO.now()
    main.write(a, cf, meta.of(a, cf, Nat.sub(t1, t0), Nat.sub(t2, t1), Nat.sub(t3, t2), Nat.sub(t5, t4)), stats2, ns, o)
    IO.print("bend e-means: final pass objective " ++ json.f(last.obj(stats2)) ++ ", train " ++ Nat.show(Nat.sub(t3, t2)) ++ " ms, " ++ U32.show(ns) ++ " splits")

@unsafe
def main.train(+a: Args, +cf: Cfg, +t0: Nat, +t1: Nat, x: Array<F32>) -> IO(Unit):
  Args{path, +n, d, k, tstr, +t, seed, sph, out, gpu} = a
  Cfg{n2, d2, k2, +b, sph2, seed2, a0, total, coef, ad, cd, kd, gpu2} = cf
  do IO<Unit>:
    st : St <- IO.pure(St, st.init(cf, x))
    +t2 : Nat <- IO.now()
    r : Run <- epochs(cf, 0, epoch.rows(n, t), [], t2, 0, 0, st)
    +t3 : Nat <- IO.now()
    main.done(a, cf, t0, t1, t2, t3, r)

@unsafe
def main.loaded.if(ok: Bool, +a: Args, +t0: Nat, +t1: Nat, x: Array<F32>) -> IO(Unit):
  match ok:
    case False{}:
      IO.die(Unit, 3, "short read: the file holds fewer than n * d floats")
    case True{}:
      main.train(a, cfg.of(a), t0, t1, x)

@unsafe
def main.loaded(+a: Args, +t0: Nat, +t1: Nat, xc: Array<F32> & U32) -> IO(Unit):
  Args{path, +n, +d, k, tstr, t, seed, sph, out, gpu} = a
  (x, +cnt) = xc
  main.loaded.if(U32.is_eq(cnt, (n * d : U32)), a, t0, t1, x)

@unsafe
def main.load(+a: Args, f: File) -> IO(Unit):
  Args{path, +n, +d, k, tstr, t, seed, sph, out, gpu} = a
  +bytes = (n * d * 4 : U32)
  do IO<Unit>:
    +t0 : Nat <- IO.now()
    xc : Array<F32> & U32 <- load.go(nat(ceil.div(bytes, 1048576)), f, 0, 1048576, 0, Array.new(F32, depth((n * d : U32)), 0.0))
    +t1 : Nat <- IO.now()
    main.loaded(a, t0, t1, xc)

@unsafe
def main.sized.if(ok: Bool, +a: Args, f: File) -> IO(Unit):
  match ok:
    case False{}:
      IO.die(Unit, 3, "the file size is not n * d * 4 bytes")
    case True{}:
      main.load(a, f)

@unsafe
def main.sized(+a: Args, pr: File & Result<&1, &1, U32 & String, U32>) -> IO(Unit):
  Args{path, +n, +d, k, tstr, t, seed, sph, out, gpu} = a
  (f, res) = pr
  main.sized.res(a, (n * d * 4 : U32), f, res)

@unsafe
def main.sized.res(+a: Args, +want: U32, f: File, res: Result<&1, &1, U32 & String, U32>) -> IO(Unit):
  match res:
    case Fail{e}:
      IO.die(Unit, 3, "cannot size the input file")
    case Done{+sz}:
      main.sized.if(U32.is_eq(sz, want), a, f)

@unsafe
def main.open(ok: Bool, +a: Args) -> IO(Unit):
  match ok:
    case False{}:
      IO.die(Unit, 2, "bad arguments: need n, d, k >= 1, k <= n, nepochs > 0, n * d < 2^30")
    case True{}:
      Args{+path, n, d, k, tstr, t, seed, sph, out, gpu} = a
      do IO<Unit>:
        f : File <- IO.try(File, File.open(path, "r"))
        pr : File & Result<&1, &1, U32 & String, U32> <- File.size(f)
        main.sized(a, pr)

def args.ok(+a: Args) -> Bool:
  Args{path, +n, +d, +k, tstr, +t, seed, sph, out, gpu} = a
  Bool.and(Bool.and(U32.is_ne(n, 0), U32.is_ne(d, 0)),
    Bool.and(Bool.and(U32.is_ne(k, 0), U32.is_le(k, n)),
      Bool.and((t > 0.0 : F32), (U32.to_f32(n) * U32.to_f32(d) < 1073741824.0 : F32))))

@unsafe
def main.args(args: List<String>) -> IO(Unit):
  match args:
    case _ <> path <> sn <> sd <> sk <> +st <> ss <> sp <> out <> rest:
      +a = {Args{path, u32(sn), u32(sd), u32(sk), st, f32(st), u32(ss), U32.is_eq(u32(sp), 1), out, gpu.arg(rest)} : Args}
      main.open(args.ok(a), a)
    case _:
      IO.die(Unit, 2, "usage: emeans <vectors.f32> <n> <d> <k> <nepochs> <seed> <spherical 0|1> <outdir> [gpu 0|1]")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    main.args(args)
