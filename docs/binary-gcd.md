# The binary GCD implementation explained

This library previously contained two constant-time GCD engines: `safegcd`
(Bernstein–Yang, [gcd.cr.yp.to](https://gcd.cr.yp.to/papers.html#safegcd)) and
`bingcd` (Pornin's optimized binary GCD,
[eprint 2020/972](https://eprint.iacr.org/2020/972)).
Both are now replaced by a single engine under
[`src/modular/gcd/`](../src/modular/gcd.rs), backing `Uint`/`BoxedUint`
`gcd`/`xgcd`/`invert_mod` and the Jacobi/Kronecker symbol, in constant time and
variable time alike.

At its core this is still Pornin's algorithm — the same elementary step, the
same idea of batching many steps into one wide-matrix update. What has changed
is *how* the batching is done. This document explains each change, why it was
made, and proves the termination bound the batching relies on.

Throughout, `W` is the machine word size (`Word::BITS`, 64 on most targets),
`N` is the input bit width, `S` is the number of elementary steps folded into
one batch, and a **batch** is one call that folds those `S` steps into a single
matrix, applied once at full width.

---

## 1. Split form instead of Pornin's glued form

Pornin approximates each operand with a single register-width word (§2 of the
paper, Algorithm 2). For a `2k`-bit register he packs the exact low `k − 1`
bits together with a `k + 1`-bit approximation of the operand's top,
contiguously, into one value:

```
ā = (a mod 2^(k−1)) + 2^(k−1)·⌊a / 2^(n−k−1)⌋
```

and `b̄` likewise, where `n = max(len(a), len(b))`. The inner loop then runs its
subtract-and-halve steps on `ā`/`b̄` as ordinary single-word integers: a borrow
out of the low `k − 1` bits is *meant* to ripple up into the high
approximation, because together they represent one packed number.

The cost is that the two halves share one register's worth of bits. Growing the
`k + 1`-bit approximation to buy precision shrinks the batch length `k − 1`,
and vice versa. Widening the whole representation to a double word for each
operand would relieve the pressure, but at a significant per-step cost.

This implementation instead carries each operand as **two independent machine
words, each with its own full register width**:

- a **low word** (`al`/`bl` below) — the exact low `W` bits, driving the
  parity decision every step
- a **high word** (`ah`/`bh`) — a comparison value derived from the top of the
  operand, driving the swap decision every step

See `partial_xgcd` in `bingcd.rs`: the `hi` and `lo` chains run the same
subtract-and-shift shape side by side, gated by the same `apply_sub`/
`apply_swap` decisions, but as two separate `W`-bit values with no arithmetic
relationship to each other. Unlike `ā`, `ah` is not "the top bits of a single
packed number that also holds `al`". That buys two things.

**More steps per batch.** A `W`-bit glued register is a `2k`-bit register with
`k = W/2`, so Pornin's own bound caps it at `k − 1 = 31` steps. Here the low
bits and the high approximation no longer compete for one register's budget, so
a batch can run `GCD_BATCH_SIZE = 58` steps at `W = 64` — roughly double. The
price is that the high word is no longer tied to the operand's true value by
the packed representation's own exactness, so it can drift out of sync over
those extra steps. Bounding that drift is what §2 is about, and the batch
length is set by that bound rather than by the register: the split form runs as
many steps as its own accumulated error can still be certified against, and no
more.

**Independent chains issue in parallel.** Because `ah -= bh` and `al -= bl`
operate on unrelated registers, they carry no data dependency on each other and
can issue together on a wide core. A glued register's single subtraction has no
such freedom: by construction its low bits' borrow must reach its high bits, so
the whole thing runs as one dependent chain.

The swap decision costs nothing extra either. The step subtracts `ah - bh`
unconditionally and uses the borrow as the swap signal, the same trick
`divstep` uses on the sign of its own counter.

## 2. The ambiguity band

Running the high word for many steps without renormalizing it against the true
operand means a comparison can occasionally be wrong: when the two compact
words are nearly equal, accumulated representation error can flip which one
looks larger.

For GCD this doesn't matter. A wrong comparison wastes a step but cannot
corrupt the result — §4's termination proof holds whether or not the
comparisons were correct. For the Jacobi symbol it matters a great deal: a
wrong swap silently flips the accumulated quadratic-reciprocity sign, and
nothing downstream can distinguish a legitimate flip from a spurious one. Every
step the Jacobi path executes has to be certifiably correct.

So the Jacobi variant doesn't push through a batch unconditionally. It tracks
an **ambiguity band** — the width of the interval within which a comparison
cannot be trusted — and halts the batch the moment a comparison falls inside
it, rather than repairing after the fact. In the code this is the `HALTING`
branch of `partial_xgcd`:

```rust,ignore
let above_threshold = if HALTING {
    word::choice_from_nz(hi_diff >> SPLIT_THRESHOLD_BITS).or(exact.or(a_odd.not()))
} else {
    Choice::TRUE
};
```

Below the threshold `unhalted` goes false, and every following step in the batch
becomes a no-op rather than risk a step whose direction can't be certified.

Two disjuncts widen that test, both covering cases where nothing needs
certifying. `exact` is set when the extraction has returned the operands
themselves rather than an approximation of them — when both fit in the compact
word, so it carries no representation error and there is no band to clear.
`a_odd.not()` covers a halving step, whose direction comes from the exact low
word rather than from a comparison at all.

`SPLIT_THRESHOLD_BITS` is `bitlen(T)`, where `T` is the drift bound of §4's
Lemma 2 — enumerated exhaustively rather than guessed. Because drift
accumulates per step, `T` depends on the batch length, so the constant and the
batch size determine each other and are solved together in §4:

| | `W = 64` | `W = 32` | `W = 16` | `W = 8` |
|---|---|---|---|---|
| `T` | 23 | 12 | 7 | 5 |
| `SPLIT_THRESHOLD_BITS` | 5 | 4 | 3 | 3 |
| `GCD_BATCH_SIZE` | 58 | 27 | 12 | 4 |

Both paths read the same scheduled extraction over deferred-sign operands
(§3), so one drift bound serves both and the table has no per-path column:
`T(S) = ⌈(3S + 25)/9⌉`, and `GCD_BATCH_SIZE = W − SPLIT_THRESHOLD_BITS − 1` at
every word size.

The batch length is one step below the longest the drift bound would certify.
That step is the margin Pornin's parameters carry for free — his inner loop
runs `k − 1` iterations against a `k + 1`-bit approximation — and §4.6 shows
what it buys.

The Jacobi path additionally consults `T` at run time, as the halt test above;
the GCD path consults it only through the batch length.

## 3. Avoiding O(N) costs per round

Compared to `safegcd`, Pornin's algorithm carries several costs that scale with
the operand's full width, which is why it falls behind at larger sizes despite
needing fewer reduction steps. This implementation removes or reduces three of
them, making it competitive at up to hundreds of limbs.

### The tracked window

`GcdPair::len` in `pair.rs` is a limb-granularity bound: nothing at or above
that limb index is read or written during the round, in either loop
(`a.leading_mut(len)` and `b.leading_mut(len)` clip every
operand access to it). It steps down one whole limb at a time, whenever the
consumed step count crosses a precomputed threshold (`window_shrink_at`),
which bounds a round's memory footprint and access pattern to less than the
full `N` bits.

It doesn't engage immediately. The first threshold isn't crossed until the
remaining budget `k_remain` has fallen to roughly `N`, and since the total
budget starts at `2N − 1`, that is the halfway point of the run. The window
sits at full width through the first half and narrows toward zero across the
second, averaging about `3/4` of full-width work — a **~25% reduction** in the
limbs that wide operations ever touch. §4's corollary derives this and shows
the schedule depends only on public quantities.

### The extraction window

This is the position of the high-word values fed into the transition matrix.
Pornin computes it by scanning both inputs for their aligned top words. The
standard constant-time way to do that is to walk every limb from the top once
per round, maintaining a "have we passed the true top yet" flag: one zero-check
and two conditional swaps per limb, every round (see `GcdPair::extract_compact_pair`). At
small sizes the overhead is minor; at larger `N` it is not.

For tracked window sizes above `SMALL_THRESHOLD_LIMBS`, this implementation
reads a fixed-size **3-limb trip** instead, via `GcdPair::scheduled_extract_pair`,
at a position the round tracks directly. What is tracked is the *bottom* of the
extraction: `extract_pos` holds `E − W`, starting at `N − W` and descending a
fixed `S/2` bits every round, so `E_i = N − (S/2)·i` after `i` batches for the
reference position `E` that §4 reasons about.

`S` is odd at some word sizes, so `extract_pos` is held in **half-bits**: the
round subtracts `S` from `2·extract_pos` and takes the bit index as `bit2 >> 1`.
That keeps the schedule exact — no rounding drift accumulating across the
run — with one subtraction and a predicated borrow, and no division. No scan
is involved either way: every limb index the trip touches is public and
schedule-derived rather than data-derived.

Handing over `E − W` rather than `E` is what makes the trip wide enough at both
ends, and the offset is `W` rather than `S` for a reason the termination proof
needs. Write the limb alignment as `E_lo = E − W − r`, `r ∈ [0, W)`. The trip
spans `3W` bits from there, so its top sits at `E + 2W − r`, clearing `E + S` by

```
    2W − r − S  ≥  W − S + 1
```

— 7, 6, 5 and 5 at the four word sizes, worst at `r = W − 1`. That margin is
what Theorem 1 needs: an operand can overflow the trip only if it exceeds
`E + S`.

At the bottom the offset does more than reach far enough. `E_lo ≤ E − W` means
the window's base sits a full word below the reference position, so an operand
that fails to fill the window is thereby known to lie well below `E` — which is
exactly the headroom §4.6's Case (iii) spends. An offset of `S` would reach far
enough at both ends but would leave that case a few bits short at each word
size; the extra `W − S` bits of descent cost nothing else, since the trip stays
three limbs and the schedule still moves by `S/2` per round.

Within the trip the position is effectively data-chosen, even though the trip
itself is not. The shared normalizing shift is taken from the larger of the
pair, so whenever the true top bit lies inside the trip the compact words are
exactly the top `W` bits of the larger operand — the same choice Pornin's
per-round scan makes, arrived at without the scan.

Where an operand's true top bit lies above the trip entirely, a single
overflow flag — a plain OR-reduction, not a per-limb zero-check-and-swap —
records the fact. When either flag is set, the extraction discards both compact
words and substitutes the flags themselves, each widened to a full-word mask:

```rust,ignore
Limb::select(a_hi, Limb::choice_to_mask(a_over), any_over).0,
Limb::select(b_hi, Limb::choice_to_mask(b_over), any_over).0,
```

The flagged operand becomes `MAX` and its partner becomes exactly zero, so the
comparison the batch makes is "nonzero beats zero" — which stays true for every
step of the batch, however far the marker has been halved, unlike a comparison
between two magnitudes. §4.2 proves the substitution sound.

The flag is the shift's own out-of-range case. Where the normalizing shift `t`
satisfies `κ = 3W − t`, an operand overflowing the trip is one whose `κ` exceeds
`3W` — a shift the trip is too narrow to express, `t < 0`. A wider extraction
would report it as an ordinary shift and Case (ii) would absorb it; three limbs
cannot, so it is signalled separately. The substitution is what a full window
degenerates to at that boundary: `MAX` against zero is the limit of a compact
pair whose separation has run past what the window can represent.

Neither the normalization nor the operand's real bits matter on this branch.
The marker carries dominance only; the reduction the batch banks comes from the
exact low words.

Only once the tracked window has itself narrowed to `SMALL_THRESHOLD_LIMBS`
(cheap to scan regardless) does the implementation drop the scheduled trip and
switch to Stage 2's exact, data-derived extraction.

### Deferred sign correction

Forcing both operands back to positive each round is another full-width pass.
Instead, the operand is tracked through an
`ExtendedIntRef`: an ordinary `UintRef` plus
one extra sign/overflow limb, genuinely negative in two's complement across
rounds, and collapsed to a true magnitude only where something downstream needs
one — the switch to Stage 2, or the final result. The correction folds into
work already being done rather than adding a pass of its own.

While Stage 1 runs, a negative operand's window is read by complementing the
extracted words in place rather than negating the wide value: the high word
bitwise (`!raw`), the low word by wrapping negation. The bitwise form leaves the
high word one short of the true magnitude where a borrow would have propagated,
which costs a ulp of precision and nothing else — it can waste a round, never
corrupt the result (§4). Nothing has to scan the operand's sign to know which
case applies.

The read yields a magnitude on both words — the high word by bitwise
complement, the low word by wrapping negation, which is what gives the Jacobi
path exact low bits of `|a|`. So the compact recurrence, and the matrix it
accumulates, both live in magnitude space, and the wide application enters the
deferred signs into that matrix's coefficients: one conditional negation per
column, and the wide update then reproduces the magnitude-space step exactly.
That is the sign correction, folded into the matrix application rather than run
as a pass of its own, and §4.1 records what the proof needs from it as (MAG).
Without it the wide update would *add* magnitudes wherever the two signs
differ, which costs no correctness and all of the reduction bound.

Both paths use this. The complement read is what makes the drift enumeration
start from a two-ulp interval rather than one (§4.5), and that cost is priced
into the constants of §2 for GCD and Jacobi alike. Where the Jacobi symbol
needs exact low bits of the magnitude rather than an off-by-one window — its
per-step factors read `a mod 4` — it takes them from the low limb alone, as
`(!a_lo).wrapping_add(1)` under the sign mask: the borrow of a two's-complement
negation propagates upward only, so the lowest limb is exact without touching
the rest of the operand.

That trick is scoped to Stage 1. The transition into Stage 2 performs the one
genuine full-width negation `b` ever needs; from there
`GcdPair::gcd_small_with_budget` extracts with the exact, unsigned
`GcdPair::extract_compact_pair` and re-corrects both operands to non-negative every round via
`wrapping_apply_unsigned_shift` — cheap, because that window is capped at
`SMALL_THRESHOLD_LIMBS` (8) limbs rather than `N`. Past that point neither the
deferred sign nor the complement trick is needed.

## 4. The termination guarantee

The batched algorithm must halt within a fixed, public budget of elementary
steps — Pornin's own `2N − 1` (`iterations()` in `bingcd.rs`). The
implementation spends that budget from a counter `k_remain`, decremented by `S`
per batch, rather than running a fixed number of batches: Stage 1 batches run
while the tracked window is above `SMALL_THRESHOLD_LIMBS`, Stage 2 continues on
the narrowed window until `k_remain` reaches zero, and the last two words are
reduced exactly. What has to be shown is that the budget suffices, whichever
mix of stages spends it.

Everything in this section hangs off a single invariant. Section 4.1 states it;
4.2 and 4.3 draw the two consequences the implementation needs — that the
extraction is sound, and that the algorithm terminates — both of which assume
the invariant holds. Sections 4.4 to 4.6 discharge that assumption, 4.7 derives
the shrinking window of §3 as a by-product, and 4.8 records the alternatives
that were tried and why each fails.

### 4.1 Setup and the invariant

Everything in this section is stated on **magnitudes**. The deferred sign of §3
means an operand can be negative between batches, and the extraction reads it
by bitwise complement, so the compact words the batch compares are magnitudes;
`bitlen` below is the bit length of `|a|`, and a comparison is between `|a|`
and `|b|`.

The wide update agrees with that, and it is worth saying why explicitly, since
it is the one property of §4 that lives outside §4:

> **(MAG)** Up to and including the first banded step of §4.6, every executed
> step replaces the larger *magnitude* by `(larger − smaller)/2`, never by
> `(larger + smaller)/2`.

*Why it holds.* The matrix `M` a batch accumulates is a magnitude-space matrix,
`(A', B')ᵀ = M·(A, B)ᵀ`, while the wide operands are `x = diag(s_a, s_b)·(A, B)ᵀ`
with `s ∈ {±1}` the deferred signs. Since `diag(s)² = I`,

```
    M·diag(s) · x  =  M·(A, B)ᵀ  =  (A', B')ᵀ
```

so folding the signs into `M`'s columns — one conditional negation each — makes
the wide update compute the magnitude-space result exactly, and the batch
proceeds as though begun on the non-negative pair `(A₀, B₀)`. A correct step
replaces the larger of a non-negative pair by their halved difference and leaves
both non-negative, so everything up to the first banded step is non-negative and
`|a − b| = |A − B|` there.

*Why it is scoped.* Past that step the claim is dropped, and must be: a
divergence leaves one operand negative, after which the wide subtraction adds
magnitudes. §4.6 asks for no magnitude-subtraction bound beyond that point —
only the non-regrowth argument, which runs on the compact words.

Without the fold the wide update would compute `(A + B)/2` where it meant
`(A − B)/2` whenever the two signs differ. That would not be a correctness bug —
`gcd(A + B, B) = gcd(A, B)`, so the answer survives — but the magnitude *grows*,
and with it `Φ`. Lemma 1 and every reduction bound in §4.6 would fail silently,
on inputs that still produce the right gcd. The fold is §3's "the correction
folds into work already being done rather than adding a pass of its own": the
correction is not skipped, it is moved into an operation the batch was
performing anyway.

The error interval `(MAG)` inherits is not inherited at all. Lemma 2's `[−1, 1)`
per operand is a property of the *read* — a truncation, plus the one-ulp
shortfall of the bitwise complement on a negative operand — so it is
re-established from scratch at every batch by the extraction, whatever the
previous batch left behind. No lemma is needed to carry it across a boundary.

For operands `a`, `b` at the start of a batch, write

```
    L = max(bitlen a, bitlen b),   m = min(bitlen a, bitlen b),   Φ = L + m
```

where `E` is the reference position of §3's schedule — `S` above the tracked
`extract_pos`, which the round decrements by `S/2` in `gcd_odd_with_budget`,
tracked in half-bits so that an odd `S` stays exact (§3) — so
`E_i = N − (S/2)·i`, and `E_i` may land on a half-bit. Primes mark
post-batch quantities throughout, so `E' = E − S/2` and `ΔΦ = Φ − Φ'`.

Absolute bit positions are not the natural coordinates here. Everything the
proof needs is a *distance below the schedule*, so write

```
    λ = E − L,        μ = E − m
```

and note that `λ ≤ μ`. The two quantities the argument is stated in are then
both sums or differences of these:

```
    h = 2E − Φ = λ + μ            headroom under the invariant
    G = L − m  = μ − λ            the operands' separation
```

The invariant carried across batches is:

> **(INV)** `Φ ≤ 2E`, equivalently `h ≥ 0`, equivalently `λ + μ ≥ 0`.

It holds initially, since `E₀ = N ≥ L ≥ m` gives `Φ₀ ≤ 2N = 2E₀`. Theorems 1
and 2 need only the weaker corollary `m ≤ E`, that is `μ ≥ 0`, immediate from
`m ≤ Φ/2 ≤ E`. `h` is not slack to be discarded: §4.6 spends it.

The window's geometry reduces the same way. The extraction reads `3W` bits
from a base `E_lo = E − W − r`, with `r ∈ [0, W)` the limb alignment (§3), so
its fill, its normalizing shift, and the scale of the compact words are all
functions of `λ` and `r`:

```
    κ = L − E_lo = W + r − λ,     t = 3W − κ = 2W + λ − r,     σ = E − W − λ
```

`κ` is deliberately unclamped: it is negative when both operands lie below the
window, and exceeds `3W` — the width the trip covers — when they overflow it,
which is the flag of §3 and the `t < 0` of §4.6's case line. The clamped
`min(W, max(0, κ))` counts usable bits, but the arguments below need the
unclamped form. What matters is that the whole of §4.6's case analysis is a
statement about where `λ` falls relative to `r`.

### 4.2 Theorem 1 — soundness of the extraction

*Under (INV), at most one operand can raise the overflow flag of §3. The
substitution that branch performs therefore yields one `MAX` and one zero, and
the comparison it produces is correct at every step of the batch.*

*Proof.* A flag fires exactly when an operand exceeds the top of the trip —
equivalently, when its `κ` exceeds `3W` and the shift of §3 would have to be
negative. The trip's top sits at or above `E + S`, and by (INV) the smaller
operand satisfies `m ≤ E < E + S`, so its flag cannot fire; at most one flag is
raised, and the substitution yields exactly one `MAX` and one zero rather than
two of either.

The partner is then zero, and no step of the batch can make it nonzero: a step
replaces the larger operand and leaves the other untouched. The marker survives
the batch, since `MAX >> S ≠ 0` for `S ≤ W − 1`. The ordering the batch reads
is therefore fixed for its whole duration, and it agrees with the truth because
a flagged operand exceeds `E` by more than `S`, so its true bit length still
exceeds `E ≥ m` after all `S` halvings. ∎

The certainty here rests on the zero, not on any gap between magnitudes — one
side stays exactly zero however far the marker has been halved. That is what
allows the substitution to discard the flagged operand's real bits, and with
them the leading-zero scan this branch would otherwise need (§3): the marker
carries dominance, not magnitude.

The gap itself is nonetheless wide enough for the whole batch, which is why
§2's halt test needs no extra disjunct even though it measures only the gap.
`MAX >> k` drops below the threshold `2^SPLIT_THRESHOLD_BITS` only once
`k > W − SPLIT_THRESHOLD_BITS`, while the last comparison of a batch uses
`k = S − 1`. The condition is therefore `S ≤ W − SPLIT_THRESHOLD_BITS + 1`, and
`S = W − SPLIT_THRESHOLD_BITS − 1` clears it by two:

```
    W = 64:  MAX >> 57 = 127  vs threshold 32
    W = 32:  MAX >> 26 =  63  vs threshold 16
    W = 16:  MAX >> 11 =  31  vs threshold  8
    W =  8:  MAX >>  3 =  31  vs threshold  8
```

The two bits are not spare capacity to be reclaimed. One is the step §4.5 gives
up so that a divergence banks more than the batch owes; the other is the
inequality's own margin. What the coupling still forbids is raising the
threshold constant by more than two without shortening the batch in step —
which §4.5's admissible range would otherwise permit — since that would let the
marker fall inside the band before the batch ends, and the batch would truncate
on a comparison that was never uncertain.

### 4.3 Theorem 2 — termination

*Under (INV), the algorithm has reached `m = 0` by the time the step budget is
exhausted.*

*Proof.* The schedule and the counter are the same quantity. Both start at
their full value and both fall by `S` per batch — `E` by `S/2` in half-bits —
so throughout the run

```
    E  =  (k_remain + 1) / 2
```

which is `N` at `k_remain = 2N − 1` and `1/2` at `k_remain = 0`. The last batch
may overshoot, taking `k_remain` below zero and `E` with it; either way
`E ≤ 1/2` at exhaustion, and (INV) gives `Φ ≤ 2E ≤ 1`. Since `L ≥ m ≥ 0`, that
forces `m = 0`: one operand has become zero and the other is the gcd. Nothing
in this
depends on how the budget was split between stages, only that each batch
decrements the counter by the steps it was charged for. Every elementary step
replaces one
operand by `u ± v` and halves it, and the halving is always of an even value —
the parity comes from the exact low word, so a step halves `a` only when `a` is
even, and subtracts only when both are odd, in which case the difference is
even. Under that parity condition every such step preserves `gcd(a, b)`,
whether or not the comparison that chose it was correct
— so reaching `m = 0` by any route leaves the surviving operand equal to the
gcd. ∎

### 4.4 Two lemmas

Two facts about a batch, both independent of whether its comparisons were
correct. The first bounds what a batch can undo, the second when it can be
wrong at all.

**Lemma 1 (monotonicity).** *`L` is non-increasing across a batch, whether or
not its comparisons were correct.*

*Proof.* A step replaces one operand `u` by `(u ± v)/2` and leaves the other
unchanged. Since `|u ± v| ≤ |u| + |v| ≤ 2^{L+1}`, the replacement has bit
length at most `L`. ∎

The lemma is deliberately sign-agnostic — it holds for `u + v` as readily as for
`u − v` — and that is also the limit of what it gives. It says a batch cannot
make things worse; it says nothing about a batch making them better. The
banked reduction of §4.6 is a different and stronger claim, that the replaced
operand drops to *half the difference* of the two, and it holds only under
(MAG): with the signs folded the step subtracts magnitudes, and without them it
would add them, at which point Lemma 1 would still be true and the banking bound
false. The two are used for different purposes throughout §4.6 and should not be
read as the same fact at different strengths.

**Lemma 2 (divergence threshold).** *A step's comparison can be wrong only if
the two compact words differ by less than `T(S)`, the width of the
representation-error interval after `S` steps.*

*Proof.* Write `P = 2^σ` for the weight of the compact words' least significant
bit, and `ρ = (true slot value) − (compact word)·P` for the representation
error. A comparison is wrong when `|x| ≥ |y|` holds for the compact words while
`|A| < |B|` holds for the true operands, which requires
`(|x| − |y|)·P < ρ_b − ρ_a`. The threshold is the width of the `ρ` interval,
which has two sources.

The first is the **initial truncation**, worth two ulps. Both paths read the
same scheduled trip over deferred-sign operands (§3): the shift is capped so no
bits below the window base are fabricated, which holds the truncation itself to
one ulp, and a negative operand is read by bitwise complement, which is `|x| − 1`
and costs the second. The interval is `[−1, 1)` per operand, and it is the same
interval on both paths — there is no longer a path-specific precondition
here.

The second is **accumulation**. Decomposing `ρ_j = P·F_j + (u₀ρ₀ᵃ + v₀ρ₀ᵇ)/2^j`,
the second term is at most one ulp, since each row of the batch matrix
satisfies `|u| + |v| ≤ 2^j`. The first term `F` — the accumulated floor error
of the compact recurrence — obeys the same recurrence as the compact words
themselves and depends only on the compact start state and the parity sequence,
never on the operands. It can therefore be bounded by exhaustive enumeration of
a fixed-width recurrence rather than a search over operand space. ∎

### 4.5 The drift bound

Write `D` for the difference of the two operands' representation errors, in
ulps of the compact word, and `Δ(S)` for the largest `|D|` reachable within a
batch of `S` steps. `T(S) = ⌈Δ(S)⌉` is then Lemma 2's threshold.

**Where the constant comes from.** `Δ(S)` is a maximum over an enumeration of
the error recurrence: a two-dimensional affine system with three
branch maps and a half-ulp injection per step, carried forward `S` steps as a
convex hull in exact rationals. The model over-approximates a concrete
execution — it lets every branch run at every step — so `Δ(S)` is a safe upper
bound rather than the implementation's exact reachable maximum, and the hull is
exact for the model, so the closed form below is exact rather than fitted.
Appendix A gives the maps, the abstraction argument, and the certificate for
the four constants.

**The horizon is the batch length.** The accumulated floor error grows with the
number of steps enumerated, so a bound read off at a shorter horizon than the
batch actually runs is not a bound. At horizon `S` the enumeration solves in
closed form:

```
    Δ(S)  =  (3S + 25)/9  +  (−1)^S / (9·2^(S−1)),        T(S) = ⌈Δ(S)⌉
```

exact, with no unquantified remainder — the tail alternates in sign rather than
approaching from one side, so a bound that dropped it would be wrong for half
the batch lengths. It never disturbs the ceiling: `(3S + 25)/9` is never an
integer, since `3S ≡ 2 (mod 9)` has no solution, so it sits at least `1/9` from
one, while the tail is under `1/9` for every `S ≥ 2`. Hence
`T(S) = ⌈(3S + 25)/9⌉`.

The one-ulp form is `S/3 + 10/9`; the extra `5/3` is the complement read, and
the next paragraph is why it is `5/3` and not `1`.

The linear growth is structural rather than incidental. The step map's
characteristic roots are `1/2` and `−1`, and the `−1` eigenvector is `D`
itself, so the half-ulp injected by each shift-out lands on a neutral mode and
resonates instead of damping. The extremal trajectory is a run of swap-subtract
steps with the dropped bit alternating; the slope `1/3` is `(2/3) × (1/2)` —
half the injections surviving the sign flip, projected onto `D`. Only the
constant depends on the initial interval, and hence on the extraction: `10/9`
from an exact truncation, `25/9` from the complement read this engine uses.

**A larger initial interval would compound, not add.** Anything that widens the
starting interval is amplified by the same dynamics that amplify the injected
error, so for an interval of `q` ulps the band is neither `q + Δ(S)` nor
`q·Δ(S)`:

| `q` | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| band at `S = 27` | 10.11 | 11.78 | 15.44 | 23.11 |
| band at `S = 58` | 20.44 | 22.11 | 25.78 | 33.44 |

(`q = 1` is the exact-truncation figure `S/3 + 10/9`; `Δ(S)` above is the
`q = 2` entry.)

The operative row is `q = 2`, the complement read of §3. What the table shows
is how little margin is left beyond it: a window the operands don't fill,
normalized without a cap on the shift, would cost `q = 2^k` for a shortfall of
`k` bits, and at `W = 32` a one-bit shortfall on top of the signed read —
`q = 4`, band 15.44 against an effective threshold of 15 — is already the end
of it. The scheduled extraction guards against exactly that by capping its
normalizing shift at `min(clz(a_trip | b_trip), 2W)`, which holds `σ` at or
above the window base so that no bits below the base are fabricated. The window
may then be short, but short is not the same as amplified: `κ < W` costs
banking, which §4.6 pays for out of headroom, and costs no drift at all.

Solving `S + bitlen(T(S)) ≤ W − 1` for the largest self-consistent `S` gives the
batch length and threshold tabulated in §2 — `S = 58, T = 23` at `W = 64`,
`S = 27, T = 12` at `W = 32` — each landing on equality with the bound at all
four sizes, as they must if `S` is maximal subject to it. The `− 1` is the
deliberate step given up: solving against `W` instead would give `S = 59` and
`S = 28`, one longer, and §4.6 shows what that step is worth. Unifying the two
paths onto the signed read is free at `W = 64`, `32` and `16`, where the
power-of-two rounding of the threshold already had room for the extra `5/3`;
only `W = 8` pays, losing a step and a threshold bit.

The same inequality has a second reading, which is why one constant serves both
purposes. Rearranged as `S ≤ W − bitlen(2^SPLIT_THRESHOLD_BITS + Δ)`, it says
the batch is strictly shorter than what a single divergent step can pay for at
a full window; §4.6 uses it in that form to close the termination proof, and
shows separately that a partial window pays for itself out of headroom. The
cost of the strictness is one step per batch, which the `⌈(2N − 1)/S⌉` batches'
worth of budget mostly absorbs: unchanged at `N = 256` (9 batches) and `N =
512` (18), one batch more
at 384, 1024 and 2048, three more at 4096. Solved as `S + bitlen(T) ≤ W`, it
says the threshold and the batch length together fit the word, which is what
makes the halt test sound. The two differ by exactly one step at all four word
sizes, and banking is the binding one: a batch length whose reduction a single
divergence covers certifies its own comparisons with a bit in hand.

**What the shared constant costs the GCD path.** The `2^SPLIT_THRESHOLD_BITS`
term above is the halt threshold, and only the Jacobi path has one. Without a
halt, a wrong comparison is far more constrained. In the orientation where the
compact says `ah ≥ bh` while `A < B`, `(A − B)/2^σ = (ah − bh) − D < 0` forces
`D > ah − bh ≥ 0`; in the mirror orientation the same algebra gives
`D ≤ ah − bh < 0`. Either way the compact separation and the error have opposite
effects rather than adding, and `|A − B|/2^σ = |D| − |ah − bh| ≤ Δ`. A GCD-only
build could
therefore take `β = bitlen(Δ)` — 5 at `W = 64`, 4 at `W = 32` — and run
`S = 59` and `S = 28`, one step longer, with neither arm of the inequality
above applying. Unifying the two paths costs the GCD path exactly that one
step; it is not split, because two batch lengths would mean two of every
constant below.

The shift form of the test in §2 makes the effective threshold
`2^SPLIT_THRESHOLD_BITS − 1`, i.e. 31 and 15 against admissible ranges
`[23, 105]` and `[12, 52]`. The rounding costs no margin, and the room it
leaves above `T` is what absorbed the move to the signed read.

### 4.6 Theorem 3 — the invariant is preserved

§4.1's coordinates do the work that a window-position lemma would otherwise
have to: `κ = W + r − λ` is a definition, not a bound, so every bit by which the
operands fall short of filling the window is a bit by which `L` sits below `E`.

**Theorem 3.** *(INV) is preserved by every batch.*

The proof runs: obligation and constants, then the case line, then the three
cases, of which only Case (ii) subdivides. Two things follow the cases and
belong to the proof — why a halted batch is charged in full, and why the banked
reduction cannot be given back. The choices the constants embody, and the
alternatives to them, are §4.8.

*Proof.* We must show `Φ' ≤ 2E'`, that is

```
    ΔΦ  ≥  S − h,        h = 2E − Φ ≥ 0
```

so what a batch owes is `S` less whatever headroom the induction hypothesis
already leaves it. Write

```
    β = bitlen(2^SPLIT_THRESHOLD_BITS + Δ),        so  S = W − β
```

which is §4.5's inequality at equality, holding at all four word sizes.

Three constants are in play here and they are easy to conflate, so the chain
that relates them is worth writing once. Let `q = SPLIT_THRESHOLD_BITS`. Then
`T(S) = ⌈Δ(S)⌉` is the comparison-error threshold of Lemma 2, `Δ(S)` bounds the
error difference `D`, and `2^q − 1` is the implementation's integer test — the
largest compact separation the halt lets through. With
`(a − b)/2^σ = (ah − bh) − D` from the definitions of the two errors,

```
    |a − b| / 2^σ  ≤  |ah − bh| + |D|  ≤  (2^q − 1) + Δ(S)  <  2^β,
                                              β = bitlen(2^q + Δ)
```

The two middle terms are different quantities, which is why they add: the first
is how far apart the *compact words* are, the second the error in *that
difference*. The first is at most `2^q − 1` because the step that trips the halt
is itself executed — `unhalted` goes false for the steps *after* it — so the
separation at a banded step is one below the threshold at most. The relation
between the three is `Δ ≤ T < 2^q`, from `q = bitlen(T)`; the certificate table
in §4.5 gives the exact values. A
divergent step read off a window carrying `j` significant bits banks
`C(j) = j − β + 1`, so `C(W) = S + 1`: at a full window a divergence banks one
bit more than the batch owes.

The number of significant bits `j` is not a quantity the proof has to
reconstruct: the extraction's normalizing shift reports it. Unclamped, the
larger operand reaches the top of the window and `σ = L − W`, giving `j = W`;
clamped, §3's cap holds `σ = E_lo` and leaves `W − κ` leading zeros rather than
fabricating bits below the base, giving `j = κ`. Since `κ = W + r − λ`, both
that boundary and the flag are thresholds on `λ`, and the entire case analysis
is one number line:

```
    λ < r − 2W        overflow, t < 0             Theorem 1
    λ ≤ r             window full, j = W          Case (ii)
    r < λ < S/2       window short, j = κ         Case (iii)
    λ ≥ S/2           L ≤ E' already              Case (i)
```

These cover: when `r ≥ S/2` the third interval is empty and the second and
fourth overlap.

The boundaries are worth checking explicitly, since three of them are places
where two descriptions of the same configuration have to agree.

- **`λ = r`, i.e. `κ = W`.** The trip's value has bit length `W`, so
  `clz = 2W` and the cap fires — the shift is clamped, `σ = E_lo`. But
  `κ = W` also means `L = E_lo + W`, so `σ = L − W` as well: the clamped and
  unclamped formulas coincide here, and `j = min(W, κ) = W` either way. The
  boundary belongs to Case (ii) and nothing turns on which side it is assigned.
- **`λ = r − 2W`, i.e. `κ = 3W`.** The larger operand's top bit is at offset
  `3W − 1` from `E_lo`, which is the highest position the trip covers, so it is
  *not* an overflow: the flag fires on `κ > 3W` strictly, and `κ = 3W` is the
  `t = 0` end of Case (ii).
- **`λ = S/2`.** Case (i) is stated with equality because `L ≤ E − S/2 = E'` is
  what it needs, and Case (iii) is strict, so the two do not both claim it.
- **`c = β` in Case (ii).** (†) needs `m₀ ≥ σ + β`, which is `c ≥ β`
  inclusive; the descent argument of the other branch needs `c ≤ β − 1`. The
  two are complementary with no gap and no overlap.

Three cases follow.

**Case (i): `λ ≥ S/2`.** Then `L ≤ E − S/2 = E'` directly, and since
`m' ≤ L' ≤ L` by Lemma 1,

```
    Φ' = L' + m' ≤ 2L' ≤ 2L ≤ 2E'.
```

The batch need achieve nothing at all. This covers `κ ≤ 0`, where the operands
lie entirely below the window, both compact words are zero, and no comparison
carries any information — `κ ≤ 0` forces `λ ≥ W + r`, so monotonicity alone
carries it.

The remaining `λ` split on the **band**, not on whether a comparison went
wrong. Call a step *banded* if it is a subtract step — `a` odd, so a comparison
is made — whose compact words satisfy `|ah − bh| < 2^SPLIT_THRESHOLD_BITS`, and
which is not one of the exact cases of §2. The band is what the halt test
measures, and it is also what a wrong comparison requires: whichever way round
the compact comparison is, being wrong forces `|ah − bh| ≤ |D| ≤ Δ`, and
`Δ < 2^SPLIT_THRESHOLD_BITS`. So

> every step before the first banded step has a correct comparison,

on either path and whether or not any comparison is ever wrong. That is the
statement the two branches below run on, and it is why the halting flag never
enters the argument: the halt fires exactly at the first banded step, so the two
paths agree about which step matters and differ only in what they do after
it.

**No banded step.** Then every comparison in the batch is correct, and on the
halting path no step is skipped either, so all `S` execute. Each replaces the
larger operand by `(larger − smaller)/2`, strictly reducing its bit length while
the other is untouched, so `Φ` falls by at least one per step and
`ΔΦ ≥ S ≥ S − h`.

**A banded step.** Otherwise let `d` be the first, and write `ℓ` for the larger
operand's bit length there. Its comparison may be correct or wrong — banded
means only that the compact separation is small, and on the Jacobi path a
banded step with a perfectly correct comparison still truncates the batch. The
argument does not care which: the step is executed on both paths — on the
halting path `unhalted` goes false for the steps *after* it — and by (MAG) it
replaces the larger magnitude by `|A − B|/2` either way. Bounding that needs
only the separation, so with the drift on top the true operands satisfy

```
    |a − b|  <  (2^SPLIT_THRESHOLD_BITS + Δ)·2^σ  <  2^(σ + β)
```

The step leaves `|a|` below half of that, which with `σ = L − j` is at most
`L − C(j)` bits, so `Φ_{d+1} ≤ (L − C(j)) + ℓ` and, since `Φ = L + m₀` at the
start,

```
    ΔΦ  ≥  (L + m₀) − (L − C(j)) − ℓ  =  C(j) + (m₀ − ℓ)
```

Only `ℓ` is left to bound, and the two cases bound it differently — the whole
weight of the divergence analysis is in which of them applies.

Two facts are available for it. Every step before `d` was correct, and a
correct step replaces the larger and leaves the smaller untouched, so the
minimum is non-increasing and `bitlen(min_d) ≤ m₀`. And the bound above puts
the larger within `2^(σ + β)` of it. Together,

```
    ℓ  ≤  max(m₀, σ + β) + 1                                        (†)
```

Note what (†) does *not* assume: nothing about where the operands sit relative
to the window at step `d`. The window's fill is a property of the batch's
start, and by step `d` both operands may have descended well below it.


**Case (ii): `t < 2W`, the window full.** Here `j = W` and `σ + β = L − S`, so
(†) reads `ℓ ≤ max(m₀, L − S) + 1`. Which term wins is a property of the
extraction rather than of the operands' history: the smaller operand's compact
word carries `c = m₀ − σ = W − G` significant bits, where `G = L − m₀` is the
operands' initial separation, and `m₀ ≥ L − S` is exactly `c ≥ β`. So the case
divides on whether the smaller operand reaches `β` bits inside the window —
`β` being the same constant the batch length is built from.

**`c ≥ β`.** Then `ℓ ≤ m₀ + 1`. The extra bit is the straddle case: the two
operands are close enough for drift to reverse them but sit either side of a
power of two, so the modulus the batch carries forward is one longer than the
minimum it started with. It costs one bit,

```
    ΔΦ  ≥  C(W) − 1  =  S
```

and one divergent step still covers the batch's entire quota, with no appeal to
`h` at all. This is the case that fixes `S` one below the maximum: `h = 0` is
admissible under (INV) and batch 0 realizes it, so there is nothing else for
the straddle bit to come out of.

**`c < β`.** Then (†) gives only `ℓ ≤ L − S + 1` and the bound above does not
close — but a smaller operand that fails to reach `β` bits inside the window is
one that started more than `S` bits below the larger, since `c = W − G`, and
that is enough to fix the whole batch's behaviour.

The compact larger starts at
`2^(W−1)` and each step replaces it by `⌊(ah − bh)/2⌋`, so writing
`u = ah + bh` gives `u_{k+1} ≥ u_k/2 − 1` and hence

```
    ah_k  ≥  2^(W−1−k) − bh − 2
```

At the batch's last step, `k = S − 1 = W − β − 1`, that is `2^β − bh − 2`, and
`bh ≤ 2^(β−1) − 1`, so `ah ≥ bh` throughout. **The compact comparison therefore
never swaps**, and the smaller operand is untouched for the whole batch: its bit
length stays `m₀`. Checked at all four word sizes, the minimum of `ah − bh` over
the batch is 2, attained at the final step with `c = β − 1`.

A divergence is now easy to account for. It means the compact comparison
reports `a ≥ b` — no swap — while truly `a < b`, so the true larger is `b`, the
operand the step leaves alone. Hence `ℓ = m₀` exactly, and

```
    ΔΦ  ≥  C(W) + m₀ − ℓ  =  C(W)  =  S + 1
```

The wide-gap branch banks a bit *more* than the batch owes, with no appeal to
`h` and no bound on bit-length adjacency. What made the narrow branch need the
straddle bit — that the survivor can be one longer than the batch's starting
minimum — cannot arise here, because the survivor is the starting minimum.

**Case (iii): `r < λ < S/2`, the window short.** Here (†) is not needed at all:
`ℓ ≤ L` by Lemma 1 suffices, since the window's shortfall has already been paid
for. With `κ = W + r − λ` and `S = W − β`, the step banks
`C(κ) = κ − β + 1 = S + 1 + r − λ`, and `m₀ − L = λ − μ`, so

```
    ΔΦ  ≥  C(κ) + m₀ − L  =  S + 1 + r − μ
```

against an obligation of `S − h = S − λ − μ`. The `μ` cancels and the whole
case reduces to

```
    λ + r + 1  ≥  0
```

which holds with `2r + 1` to spare, since `λ > r ≥ 0`. The case closes with no
claim about the operands' relative bit lengths, and therefore with no
dependence on how far they have descended by step `d`. ∎

Two things the cases leave open. The first is why a batch that stops early is
charged as though it ran to the end.

One theorem covers both paths, and the band is why. The bound is taken at step
`d + 1`; the paths differ only in what follows it. On the halting path nothing
does — `unhalted` is false, every remaining step is a no-op, and a no-op cannot
change `Φ` — so the bound at `d + 1` is the bound at the end of the batch. That
is what justifies charging `S/2` to a batch that executed `d + 1 < S` steps:
what pays for a batch is the banded step's cancellation, not a step count.

For the batch that runs on, what remains is to show the banked reduction cannot
be given back. It cannot, for two reasons that compose.

The compact words are running an ordinary binary GCD on themselves —
self-consistent and non-negative, because each step subtracts whichever of the
two the *compact* comparison judged smaller — so by Lemma 1 applied to them,
neither the larger nor the smaller ever grows: `|x_h,k| ≤ |x_h,d+1|` for every
`k ≥ d + 1`. And Lemma 2's error accounting holds at every step of the batch,
with the scale `σ` fixed for its whole duration and `|D_k| ≤ Δ(S)` at every
`k ≤ S`, so

```
    |X_k|  ≤  (|x_h,k| + Δ)·2^σ  ≤  (|x_h,d+1| + Δ)·2^σ        for all k ≥ d + 1
```

Taking bit lengths, `bitlen(X_k) ≤ σ + bitlen(|x_h,d+1| + Δ)` — the `Δ` inside
the `bitlen`, which is what keeps a fixed additive error from buying a bit at a
power-of-two boundary. The barrier is that bound, and it is set once at step
`d + 1` and never moves.

With the base case `Φ₀ ≤ 2N`, this establishes (INV) for every batch, which is
what Theorems 1 and 2 rest on.

### 4.7 Corollary — the operating window shrinks

(INV) also bounds how many limbs the wide extraction and matrix application
need to touch, which is what lets both shrink as the computation proceeds.

**Corollary.** *At every batch, `L ≤ min(N, 2E − m)`.*

*Proof.* `L ≤ N` is Lemma 1 applied from the start, where `L₀ ≤ N` is the input
width. `L ≤ 2E − m` is `Φ = L + m ≤ 2E`, i.e. (INV). ∎

The two terms dominate in turn. While `2E ≥ N` — the first half of the run —
(INV) says nothing useful and monotonicity alone carries `L ≤ N`. Once
`2E < N`, (INV) binds instead, and since `E` descends exactly `S/2` bits per
batch, the required window falls by exactly `S` bits per batch until `m` reaches
zero.

Integrating over the run, and writing `B` for the `⌈(2N − 1)/S⌉` batches the
budget affords: the first half occupies `B/2` batches at full width `N`, the
second falls linearly from `N` to `0` (mean `N/2`), for total wide work
`(B/2)·N + (B/2)·(N/2) = (3/4)·B·N`. That is the 25% reduction quoted in §3.
`window_limbs` narrows on exactly this schedule and never on the operands'
actual values, so the narrowing itself stays public and data-independent.

### 4.8 Why the constants are what they are

Three of §4.6's choices look arbitrary until the alternative is tried. Each of
these paragraphs records an alternative that fails, and where it fails, so that
a later edit does not quietly reintroduce it.

**Why the extraction offset is `W` and not `S`.** Case (iii)'s cancellation is
what fixes it. The offset is what puts `κ = W + r − λ`: an unfilled window is
one with `λ > r`, so the shortfall in banking is charged against a `λ` at least
as large, and the case pays for itself. At an offset of `S` the relation would
read `κ = S + r − λ`, the shortfall would outrun `λ` by `W − S` bits, and
Case (iii) would fail for `κ` just below `W` — a band that is non-empty at
every word size, and reachable at any batch where `E − S` lands near a limb
boundary.

**Why `S` is one below the maximum.** Case (ii) is the only case with no
headroom of its own to spend — `h = 0` is admissible under (INV), and batch 0
realizes it, since `Φ₀ = 2N` — so it has to close on the banking bound alone.
At `S = W − β + 1`, the longest batch the drift bound certifies, `C(W)` would
equal `S` exactly and the straddle bit would have nothing to come out of. One
step shorter and `C(W) = S + 1` covers it unconditionally.

**Why not widen (INV) instead.** The tight configuration scales with the slack,
so `Φ ≤ 2E + ε` reproduces the same one-bit shortfall for every `ε`. The step
has to come out of `S`.


**Relation to Pornin's proof.** Four points of contact, one of which does not
transfer.

*The margin.* His inner loop runs `k − 1` steps against a `k + 1`-bit
approximation whose error, being glued, does not accumulate, so his banking
bound exceeds his batch length by a bit. That margin is incidental there — his
batch length is capped by the `k − 1` exact low bits, not by banking — and
deliberate here, because the split form's purpose is to push `S` up until
banking is what binds. `S = W − q − 1` restores his margin at the point his
argument uses it.

*The case split.* Appendix A.3 does not ask whether the operands share a bit
length at a divergence; it splits on how far they have already shrunk. Case
(ii)'s two branches are the same move.

*The barrier.* The bound above is A.3's with one substitution: his level comes
from the glued form's error bound `2^{n−k−1}`, ours from `Δ`, and the barrier
moves with the constant. He reaches it by sign persistence — in the glued form
a divergence leaves the operands with opposite signs, that state persists
because halving preserves a sign and subtracting a negative from a positive
leaves a positive, and the barrier follows. Under (MAG) this engine never enters
that state, so the route is both unavailable and unnecessary: the compact words
are independent registers and their monotonicity gives the barrier directly.
The conclusions agree; the routes do not.

*The hidden iteration.* He reads the closing conditional negation as `S + 1`
iterations' worth of work for an `S`-bit guarantee. That iteration is consumed
by the guarantee rather than added to it — `Φ` is on magnitudes, and
`a = |a|; b = |b|` changes signs and no bit lengths — which is why the budget
divides `2N − 1` by `S` and not `S + 1`. The first version of his note made the
more aggressive claim; its errata records the counterexample.

## 5. The Jacobi symbol

The Jacobi/Kronecker symbol runs the same loop as GCD, over the same scheduled
extraction, with the same constants, the same deferred sign, and the same
absence of any correction pass. Two things differ, and both are local to the
batch boundary.

**It halts.** The symbol path uses the truncating (`HALTING = true`) form of §2,
since every executed step's swap has to be certifiably correct to keep the
running reciprocity sign trustworthy. GCD needs no such thing: a wrong
comparison there costs a sign, which the sign folding of §3 absorbs at the next
application. A halted batch's last executed step is §4.6's banded step, and it
is the only one whose comparison can have been wrong — so it is the only step
whose symbol bookkeeping can be wrong, and only if it did diverge.

**It corrects the symbol, not the state.** The batch keeps
whatever the mis-swapped step left, and the running symbol absorbs the
difference in one bit:

```
neg = a' < 0
jac ^= neg & ((b >> 1) & 1)
```

`neg` says the last comparison was wrong. Writing `X > Y` for the two true
operands at that step, a wrong guess retains the **larger**, so `b` holds `X`
while the correct continuation is `(A, Y)` with `A = |a'|` and `Y = X − 2A`.
Both `X` and `Y` are odd and positive; `A` may be even, or zero. Two things are
therefore wrong at once:

1. the step applied the factors for the modulus it kept rather than the one it
   should have — a fix of `eps ^ (2|X) ^ (2|Y)`, with `eps` the reciprocity bit
   for `X` and `Y`;
2. it hands on `(A, X)` rather than `(A, Y)`. The symbol has period `4A` in the
   denominator and the half shift `2A` costs bit 1 of `A`:
   `(A | n − 2kA) = (−1)^(k·bit1(A))·(A | n)`.

Bit 1, not bit 0. Reciprocity alone flips only when `X` and `Y` differ mod 4,
which is the odd case and gives `bit1(A)` correctly there, but misses
`A ≡ 2 mod 4`, where the flip arrives through the `(2|·)` factor instead — two
unrelated mechanisms landing on the same bit. Möller (arXiv:1907.07795) gives
the general rules for adding multiples of the numerator to the denominator.

The two corrections collapse. With `a0 = bit0(A)`, `a1 = bit1(A)`,
`x = bit1(X)`, and `bit1(Y) = x ^ a0`:

```
    eps             =  x & !a0
    (2|X) ^ (2|Y)   =  a1 ^ (a0 & x)
    (1)             =  a1 ^ x          both branches of a0 agree
    (1) ^ (2)       =  x
```

Flip the symbol iff the modulus the batch kept is `3 mod 4`. `A` drops out
entirely, which is why only its sign is read — no magnitude, no wide
subtraction, no pass over the operands. The per-step factors do need exact low
bits of `|a|`, which §3 supplies from the low limb alone.

This is a correctness argument and nothing more. The batch also keeps the
longer of the two operands, and what that costs in reduction is Theorem 3's
divergence branch — the `e` term when the window is full, the `κ` cases when it
is not.

## 6. The variable-time implementation

The constant-time paths above are constrained to a schedule that depends only
on public bit widths, precisely so their memory access pattern and control flow
reveal nothing about the operands. The variable-time implementation in
`vartime.rs` is under no such constraint, and
takes advantage of it in ways that would be unsound — or simply pointless — in
the constant-time engine.

- **The window is re-measured, not scheduled.** `GcdPairVartime` tracks `a`'s
  actual bit length and shrinks its tracked width (`len`) from the operands'
  real data every round, instead of following the `min(N, 2E)` schedule of
  §4.7. There is no secret to protect, so no reason to discard the tighter,
  data-derived bound.
- **Trailing zeros are stripped outright.** Every round starts with
  `strip_trailing_zeros`, removing however many factors of two `a` actually has
  in one step (`Word::trailing_zeros`), rather than spending one elementary GCD
  step per zero bit the way the constant-time loop must — it cannot let its
  iteration count depend on the operand. `partial_xgcd_vartime` does the
  equivalent inside a batch, jumping by `a.trailing_zeros()` instead of looping
  bit by bit.
- **Termination is checked directly**, on `a_bits == 0`, rather than by
  counting down a fixed worst-case `B`-batch budget. Variable-time code has no
  reason to keep running once the answer is in hand.

The split-form batching, the deferred sign, and (for the Jacobi symbol) the
halt test and one-bit symbol correction are otherwise the same building blocks
as the constant-time engine. What changes is only how aggressively the schedule driving them is
allowed to react to the actual operands.

---

## Appendix A. The drift enumeration

The material §4.5 relies on, separated because it is verification rather than
argument: the model whose maximum `Δ(S)` is, why that model over-approximates a
concrete batch safely, and the exact values at the four supported word sizes.
Notation is §4.5's — `D` the difference of the two representation errors, `σ`
the common scale, `q = SPLIT_THRESHOLD_BITS`.

**The enumeration, concretely.** Track the two errors as a pair
`(α, β_err)`, each the compact word minus the true slot value at the common
scale, and let `D = α − β_err`. Lemma 2 puts each in the half-open `[−1, 1)`;
the enumeration uses the **closed** `[−1, 1]`, which contains it, so what it
computes is an upper bound for the real reachable set and a maximum on a
compact one. Each step applies one of three maps, with `γ ∈ {0, ½}` the bit the
arithmetic shift drops:

```
    halve      a even          α' = α/2 − γ
    subtract   a odd, a ≥ b    α' = (α − β_err)/2 − γ
    swapsub    a odd, a < b    α' = (β_err − α)/2 − γ,   β_err' = α
```

Enumerate by carrying the convex hull of the reachable set forward `S` steps in
exact rationals, and read `Δ(S) = max|D|` off the vertices.


**The model over-approximates; the hull is exact for the model.** Two claims,
worth keeping apart.

`R_concrete ⊆ R_abstract`. A concrete execution does not choose freely among
the three maps — parity fixes halve against subtract, the compact comparison
fixes which subtract ordering runs — whereas the enumeration allows all three
at every step, and widens Lemma 2's half-open `[−1, 1)` to the closed
`[−1, 1]`. Both widenings are sound because neither constraint restricts the
adversary: the parity sequence is a function of unconstrained operands, and
both subtract orderings are reachable wherever a comparison can be wrong, which
is precisely where `D` is large. So `Δ(S)` is a **safe upper bound** on the
concrete drift, not the implementation's exact reachable maximum.

For the model itself the hull loses nothing. Every map is affine, and the
dropped bit contributes a Minkowski sum with `[−½, 0]` — the convex hull of the
two values `γ` actually takes, so an extreme point always uses an endpoint,
never an interior `γ`. Since `conv(f(X)) = f(conv(X))` for affine `f`, hulling
after each step yields exactly `conv(R_abstract,t)`, whose extreme points belong
to the set hulled. `Δ(S)` is therefore attained within the model, which is what
makes the closed form below exact rather than merely a fitted bound. Nothing
depends on sampling.

Two properties of the enumeration matter for the constant it produces.


**The four constants, certified.** Everything §4.6 needs is a rational
comparison at four values of `S`:

| `W` | `S` | `Δ(S)` exact | `T = ⌈Δ⌉` | `q` | `2^q` | `2^q + Δ` | `β` |
|---|---|---|---|---|---|---|---|
| 64 | 58 | `3186546936343924281 / 2^57` | 23 | 5 | 32 | 54.111 | 6 |
| 32 | 27 | `790393287 / 2^26` | 12 | 4 | 16 | 27.778 | 5 |
| 16 | 12 | `13881 / 2^11` | 7 | 3 | 8 | 14.778 | 4 |
| 8 | 4 | `33 / 2^3` | 5 | 3 | 8 | 12.125 | 4 |

with `q = SPLIT_THRESHOLD_BITS = bitlen(T)`. Reading off: `Δ < 2^q` at every
size, with the tightest margin at `W = 64` (22.111 against 32); and
`2^q + Δ ≤ 2^β` with `β` as tabulated, the tightest again at `W = 64` (54.111
against 64). Both are exact rational facts about the four numerators above.
