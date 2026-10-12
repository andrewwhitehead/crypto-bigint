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
is *how* the batching is done:

- **§1:** each operand is approximated by two independent words, an exact low
  word and an approximate high word, instead of one packed register. This allows
  longer batches.
- **§2:** the Jacobi symbol path stops a batch right after the first comparison
  it cannot certify.
- **§3:** the per-round work that scales with the operand width is cut down. The
  tracked window shrinks, the high words are read at a scheduled position
  instead of found by a scan, and sign correction is folded into the matrix
  application.

§4 sketches why the batched algorithm always finishes within its step budget,
and §5 and §6 cover the Jacobi symbol and the variable-time variant. The full
proof is in Appendix A, and the bound on how far the high words drift in
Appendix B.

Throughout, `W` is the machine word size (`Word::BITS`, 64 on most targets),
`N` is the input bit width, `S` is the number of elementary steps folded into
one batch, and a **batch** is one call that folds those `S` steps into a single
matrix, applied once at full width.

A constant-time reduction runs in up to three stages, skipping any the operand
width makes unnecessary:

- **Stage 1** applies batches while the tracked window (§3) is wider than
  `SMALL_THRESHOLD_LIMBS` (8 limbs). Each batch reads the operands' high words at
  a position set by the step schedule, without scanning the operands.
- **Stage 2** continues applying batches once the window has narrowed, now
  locating the high words by scanning for the operands' actual top bits.
- **Stage 3** finishes with exact single-word steps once the operands fit in one
  word: the last `W − 2` or fewer steps, or all of them for single-limb inputs.

---

## 1. Split form instead of Pornin's glued form

Pornin approximates each operand with a single register-width word (§2 of the
paper, Algorithm 2). For a `2k`-bit register he packs the exact low `k − 1`
bits together with a `k + 1`-bit approximation of the operand's top,
contiguously, into one value:

```
ā = (a mod 2^(k−1)) + 2^(k−1)·⌊a / 2^(n−k−1)⌋
```

and `b̄` likewise, where `n = max(bitlen a, bitlen b)`. The inner loop then runs
its subtract-and-halve steps on `ā`/`b̄` as ordinary single-word integers: a
borrow out of the low `k − 1` bits is *meant* to ripple up into the high
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

Side by side, for one operand:

```
  Pornin's glued form: one 2k-bit register per operand

    ┌──────── k + 1 bits ─────────┬──────── k − 1 bits ─────────┐
    │ top of a (approximate)      │ low bits of a (exact)       │  ā
    └─────────────────────────────┴─────────────────────────────┘
         a borrow out of the low part reaches the high part

  Split form: two independent W-bit words per operand

    ┌────────── W bits ───────────┐  ┌────────── W bits ───────────┐
    │ ah: top of a (approximate)  │  │ al: low bits of a (exact)   │
    └─────────────────────────────┘  └─────────────────────────────┘
       drives the swap decision        drives the parity decision
```

Together these are the operand's **compact form**, and the high words are the
"compact words" the batch compares. See `partial_xgcd` in `bingcd.rs`: the high
and low chains run the same subtract-and-shift shape side by side, gated by the
same subtract and swap decisions, but as two separate `W`-bit values with no
arithmetic relationship to each other. Unlike `ā`, `ah` is not "the top bits of
a single packed number that also holds `al`". That buys two things.

**More steps per batch.** A `W`-bit glued register is a `2k`-bit register with
`k = W/2`, so Pornin's own bound caps it at `k − 1 = 31` steps. Here the low
bits and the high approximation no longer compete for one register's budget, so
a batch can run `SPLIT_BATCH_SIZE = 58` steps at `W = 64` — roughly double. The
price is that the high word is no longer tied to the operand's true value by the
packed representation's own exactness, so it can drift out of sync over those
extra steps. §2 deals with the consequences and Appendix B bounds the drift. The
batch length is set by that bound rather than by the register: the split form
runs as many steps as its own accumulated error can still be certified against,
and no more.

**Independent chains issue in parallel.** Because `ah -= bh` and `al -= bl`
operate on unrelated registers, they carry no data dependency on each other and
can issue together on a wide core. A glued register's single subtraction has no
such freedom: by construction its low bits' borrow must reach its high bits, so
the whole thing runs as one dependent chain.

The swap decision costs nothing extra either. The step subtracts `ah - bh`
unconditionally and uses the borrow as the swap signal, the same trick
`divstep` uses on the sign of its own counter.

Two glued alternatives were implemented with the same step code and benchmarked
against this form: a single-word register (31 steps per batch) and a two-word
register (62 steps per batch). The split form was faster at every size above 2
limbs, by roughly 10–30% in constant time and more in variable time. Against the
single-word register the difference is mostly the number of matrix applications:
each batch ends in a full-width update of both operands, and 58 steps per batch
instead of 31 needs close to half as many of them for the same step budget. The
two-word register runs slightly longer batches, so its cost is in the steps
themselves, which work on double-word values instead of two independent single
words.

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

So the Jacobi variant doesn't push through a batch unconditionally. It tracks an
**ambiguity band** — the width of the interval within which a comparison cannot
be trusted — and halts the batch right after the first comparison that falls
inside it, rather than repairing after the fact. In the code this is the
`HALTING` branch of `partial_xgcd`: after each subtract step it tests whether
the high words' difference has fallen below `2^SPLIT_THRESHOLD_BITS`. Once it
has, `unhalted` goes false, and every following step in the batch becomes a
no-op rather than risk a step whose direction can't be certified.

The drift bound `T` and the halt threshold divide the compact words' separation
at a subtract step into three ranges:

```
  0                       T                     2^SPLIT_THRESHOLD_BITS
  ├───────────────────────┼─────────────────────────┼─────────────────────────►
    a wrong comparison      correct, but              not banded
    is possible             still banded

  └──────────────────── banded ─────────────────────┘
          |ah − bh| at a subtract step, in units of the compact word
```

A wrong comparison needs the compact words to be within `T` of each other, and
`T` is below the threshold, so every step before the first banded one has a
correct comparison. The halt test gives up a little: it also stops at
separations between `T` and the threshold, where the comparison is correct but
the test cannot tell.

Two exceptions widen that test, both covering cases where nothing needs
certifying. One is the `exact` flag, set when the extraction has returned the
operands themselves rather than an approximation of them — when both fit in the
compact word, so it carries no representation error and there is no band to
clear. The other is a halving step (`a` even), whose direction comes from the
exact low word rather than from a comparison at all.

`SPLIT_THRESHOLD_BITS` is `bitlen(T)`, where `T` is the drift bound computed in
Appendix B — enumerated exhaustively rather than guessed. Because drift
accumulates per step, `T` depends on the batch length, so the constant and the
batch size determine each other and are solved together in Appendix B:

| | `W = 64` | `W = 32` |
|---|---|---|
| `T` | 23 | 12 |
| `SPLIT_THRESHOLD_BITS` | 5 | 4 |
| `SPLIT_BATCH_SIZE` | 58 | 27 |

Both paths read the same scheduled extraction over deferred-sign operands
(§3), so one drift bound serves both and the table has no per-path column:
`T(S) = ⌈(3S + 25)/9⌉`, and `SPLIT_BATCH_SIZE = W − SPLIT_THRESHOLD_BITS − 1` at
both word sizes.

The batch length is one step below the longest the drift bound would certify.
That step is the margin Pornin's parameters carry for free — his inner loop
runs `k − 1` iterations against a `k + 1`-bit approximation — and Case 2 of
Appendix A.5 shows what it buys.

The Jacobi path additionally uses the threshold at run time, as the halt test
above; the GCD path uses it only through the batch length.

## 3. Avoiding O(N) costs per round

Compared to `safegcd`, Pornin's algorithm carries several costs that scale with
the operand's full width, which is why it falls behind at larger sizes despite
needing fewer reduction steps. This implementation removes or reduces three of
them, making it competitive at up to hundreds of limbs.

### The tracked window

`GcdPair::len` in `pair.rs` is a limb-granularity bound: nothing at or above
that limb index is read or written during the round, by either the extraction or
the matrix application (`a.leading_mut(len)` and `b.leading_mut(len)` clip every
operand access to it). It steps down one whole limb at a time, whenever the
consumed step count crosses a precomputed threshold (`window_shrink_at`), which
bounds a round's memory footprint and access pattern to less than the full `N`
bits.

It doesn't engage immediately. The first threshold isn't crossed until the
remaining budget `k_remain` has fallen to roughly `N`, and since the total
budget starts at `2N − 1`, that is the halfway point of the run. The window
sits at full width through the first half and narrows toward zero across the
second, averaging about `3/4` of full-width work — a **~25% reduction** in the
limbs that wide operations ever touch. Appendix A.6 derives this and shows
the schedule depends only on public quantities.

### The extraction window

The extraction window is where each batch reads the operands' high words from.
Pornin locates it by scanning both inputs for their aligned top words. The
standard constant-time way to do that is to walk every limb from the top once
per round, maintaining a "have we passed the true top yet" flag: one zero-check
and two conditional swaps per limb, every round (see `extract_compact_pair` in
`pair.rs`). At small sizes the overhead is minor; at larger `N` it is not.

In Stage 1, while the tracked window is wider than `SMALL_THRESHOLD_LIMBS`, this
implementation reads a fixed-size **3-limb trip** instead, via
`scheduled_extract_compact_pair`, at a position derived from the step budget.
What is tracked is the *bottom* of the extraction, `extract_pos = E − W`, where
`E = (k_remain + 1)/2` is read off the remaining budget before each batch is
charged. The budget falls by `S` per batch, so `E` starts at `N` and falls by
`S/2` per batch; it is the reference position that §4 reasons about. Where `S`
is odd, `E` can land on a half-bit, and the code rounds it down, which only
lowers the base. No scan is involved: every limb index the trip touches is
public and schedule-derived rather than data-derived.

Within the trip the position is effectively data-chosen, even though the trip
itself is not. The shared normalizing shift is taken from the larger of the
pair, so whenever the true top bit lies inside the trip the compact words are
exactly the top `W` bits of the larger operand — the same choice Pornin's
per-round scan makes, arrived at without the scan.

Where an operand's true top bit lies above the trip entirely, a single
overflow flag — a plain OR-reduction, not a per-limb zero-check-and-swap —
records the fact. When either flag is set, the extraction discards both compact
words and substitutes the flags themselves, each widened to a full-word mask.
The flagged operand becomes `MAX` and its partner becomes exactly zero, so the
comparison the batch makes is "nonzero beats zero" — which stays true for every
step of the batch, however far the marker has been halved, unlike a comparison
between two magnitudes. Appendix A.2 proves the substitution sound.

Handing over `E − W` rather than `E` is what makes the trip wide enough at both
ends, and the offset is `W` rather than `S` for a reason the termination proof
needs. The trip starts at the limb boundary at or below `E − W`, so its base `P`
satisfies `E − 2W < P ≤ E − W`, including when the rounding of a half-bit `E`
lowers it. The trip spans `3W` bits from there, so its top sits above `E + W`
and clears `E + S` by more than `W − S`: at least 7 bits at `W = 64` and 5½ at
`W = 32`. That margin is what Theorem 1 needs: an operand can overflow the trip
only if it exceeds `E + S`.

At the bottom, `P ≤ E − W` puts the window's base a full word below the
reference position, so an operand that fails to fill the window is known to lie
well below `E`. The termination proof pays for a short window out of exactly
that distance (Appendix A.5, Case 1), and a smaller offset would not leave
enough of it.

The geometry, with high bit positions on the left. `L` is the larger operand's
bit length, and `P` is `E − W` rounded down to a limb boundary:

```
     P+3W          E+S            E                   P                       0
      ├─────────────┼─────────────┼───────────────────┤ ···     ├─────────────┤
      └───────── trip: 3W bits read from P ───────────┘          low word

  full window
                           L
                           ├── high word ──┤
                                           └ σ = L − W

  short window
                                            L
                                      ├00000┼ j bits──┤
                                                      └ σ = P
```

When the larger operand reaches at least `W` bits above the base, the high word
is its top `W` bits: a full window. Otherwise the high word stops at the base
and holds only `j = L − P` significant bits of it under `W − j` leading zeros: a
short window. Appendix A works in these same quantities.

Only once the tracked window has itself narrowed to `SMALL_THRESHOLD_LIMBS`
(cheap to scan regardless) does the implementation drop the scheduled trip and
switch to Stage 2's exact, data-derived extraction.

### Deferred sign correction

Forcing both operands back to positive each round is another full-width pass.
Instead, each operand is tracked through an `ExtendedIntRef`: an ordinary
`UintRef` plus one extra sign/overflow limb, genuinely negative in two's
complement across rounds, and collapsed to a true magnitude only where something
downstream needs one — the switch to Stage 2, or the final result. The
correction folds into work already being done rather than adding a pass of its
own.

While Stage 1 runs, a negative operand's window is read by complementing the
extracted words in place rather than negating the wide value: the high word
bitwise (`!raw`), the low word by wrapping negation. The bitwise form leaves the
high word one short of the true magnitude where a borrow would have propagated,
which costs a ulp of precision and nothing else — it can waste a round, never
corrupt the result (§4). Nothing has to scan the operand's sign to know which
case applies.

The read yields a magnitude on both words, so the compact recurrence and the
matrix it accumulates both work on magnitudes. The wide application then folds
the deferred signs into the matrix's columns, one conditional negation each,
which makes the wide update reproduce the magnitude-space step exactly.
Appendix A.4 states what the proof needs from this as (MAG).

Both the GCD and Jacobi paths use the complement read. It is what makes the
drift enumeration start from a two-ulp interval rather than one (Appendix B),
and that cost is priced into the constants of §2 for GCD and Jacobi alike. Where
the Jacobi symbol needs exact low bits of the magnitude rather than an
off-by-one window — its per-step factors read `a mod 4` — it takes them from the
low limb alone, as `(!a_lo).wrapping_add(1)` under the sign mask: the borrow of
a two's-complement negation propagates upward only, so the lowest limb is exact
without touching the rest of the operand.

That trick is scoped to Stage 1. The transition into Stage 2 performs the one
genuine full-width negation `b` ever needs; from there `GcdPair::gcd_small`
extracts with the exact, unsigned `extract_compact_pair` and re-corrects both
operands to non-negative every round via `wrapping_apply_unsigned_shift` —
cheap, because that window is capped at `SMALL_THRESHOLD_LIMBS` (8) limbs rather
than `N`. Past that point neither the deferred sign nor the complement trick is
needed.

## 4. Why the batched algorithm terminates

The constant-time paths run a fixed, public budget of `2N − 1` elementary steps
(Pornin's bound, `iterations()` in `bingcd.rs`), spent from a counter
`k_remain`: `S` steps per batch, then Stage 3's exact single-word steps.
Appendix A proves that this budget always suffices. The argument in brief:

1. **Every step preserves the gcd**, whether or not its comparison was right.
   Parity comes from the exact low word, so a step only halves an even value or
   subtracts two odd ones. Correctness therefore reduces to termination:
   reaching a zero operand within the budget.
2. **One invariant carries the proof.** With `L` and `m` the larger and smaller
   operand bit lengths and `E = (k_remain + 1)/2` the schedule position, every
   batch starts with `L + m ≤ 2E`. When the budget is spent, `E ≤ 1/2`, which
   forces `m = 0`.
3. **Each batch preserves the invariant.** A batch has to reduce `L + m` by `S`,
   less whatever slack the invariant already leaves. Call a subtract step
   *banded* if the high words differ by less than `2^SPLIT_THRESHOLD_BITS`. A
   wrong comparison needs the high words to be closer than the drift bound `Δ`,
   which is below that threshold, so every step before the first banded step
   compares correctly. Then:
   - with no banded step, all `S` steps are correct and each removes at least
     one bit;
   - at the first banded step the two operands agree in all but their last
     few high-word bits (the threshold plus the drift), so subtracting them
     removes enough bits for the whole batch at once, and nothing later in the
     batch can give those bits back;
   - if the operands don't fill the extraction window, the larger operand sits
     far enough below `E` to cover the shortfall, which is why the window sits a
     full word below `E` (§3).
4. **The drift bound** `Δ(S)` (how far the high words can wander from the true
   operands over `S` steps) is computed exactly by enumerating the error
   recurrence (Appendix B). It fixes `SPLIT_THRESHOLD_BITS` and `S` together,
   with `S + SPLIT_THRESHOLD_BITS = W − 1`.

The same invariant also bounds the operands' width, which is what lets the
tracked window of §3 shrink (Appendix A.6).

## 5. The Jacobi symbol

The Jacobi/Kronecker symbol runs the same loop as GCD, over the same scheduled
extraction, with the same constants, the same deferred sign, and the same
absence of any correction pass. Two things differ, and both are local to the
batch boundary.

**It halts.** The symbol path uses the truncating (`HALTING = true`) form of §2,
since every executed step's swap has to be certifiably correct to keep the
running reciprocity sign trustworthy. GCD needs no such thing: a wrong
comparison there costs a sign, which the sign folding of §3 absorbs at the next
application. A halted batch's last executed step is the banded step of Appendix
A.5, and it is the only one whose comparison can have been wrong — so it is the
only step whose symbol bookkeeping can be wrong, and only if it did diverge.

**It corrects the symbol, not the state.** The batch keeps whatever the
mis-swapped step left, and the running symbol absorbs the difference in one bit:

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
longer of the two operands, and what that costs in reduction is accounted for
by the banded-step cases of Theorem 3 (Appendix A.5).

## 6. The variable-time implementation

The constant-time paths above are constrained to a schedule that depends only on
public bit widths, precisely so their memory access pattern and control flow
reveal nothing about the operands. The variable-time implementation in
`pair_vartime.rs` is under no such constraint, and takes advantage of it in ways
that would be unsound — or simply pointless — in the constant-time engine.

- **The window is re-measured, not scheduled.** `GcdPairVartime` tracks `a`'s
  actual bit length and shrinks its tracked width (`len`) from the operands'
  real data every round, instead of following the `min(N, 2E)` schedule of
  Appendix A.6. There is no secret to protect, so no reason to discard the
  tighter, data-derived bound.
- **Trailing zeros are stripped outright.** Every round starts with
  `strip_trailing_zeros`, removing however many factors of two `a` actually has
  in one step (`Word::trailing_zeros`), rather than spending one elementary GCD
  step per zero bit the way the constant-time loop must — it cannot let its
  iteration count depend on the operand. `partial_xgcd_vartime` does the
  equivalent inside a batch, jumping by `a.trailing_zeros()` instead of looping
  bit by bit.
- **Termination is checked directly**, on `a_bits == 0`, rather than by
  counting down a fixed worst-case step budget. Variable-time code has no
  reason to keep running once the answer is in hand.

The split-form batching, the deferred sign, and (for the Jacobi symbol) the
halt test and one-bit symbol correction are otherwise the same building blocks
as the constant-time engine. What changes is only how aggressively the schedule
driving them is allowed to react to the actual operands.

---

## Appendix A. The termination proof

This appendix proves that the step budget of §4 always suffices: Pornin's own
`2N − 1` elementary steps (`iterations()` in `bingcd.rs`), spent from the counter
`k_remain`, whichever mix of stages spends it.

Everything hangs off a single invariant. A.1 states it; A.2 and A.3 draw the
two consequences the implementation needs — that the extraction is sound, and
that the algorithm terminates — both of which assume the invariant holds. A.4
collects three facts about a single batch, A.5 uses them to show the invariant
is preserved, and A.6 derives the shrinking window of §3 as a by-product.

### A.1 Setup and the invariant

Everything in this appendix is stated on **magnitudes**. The deferred sign of §3
means an operand can be negative between batches, and the extraction reads it by
bitwise complement, so the compact words the batch compares are magnitudes. From
here on `a` and `b` denote the two operands' magnitudes, so `bitlen a` is the
bit length of `|a|` and a comparison is between `|a|` and `|b|`. A.4 shows the
wide update agrees.

For operands `a`, `b` at the start of a batch, write

```
    L = max(bitlen a, bitlen b),   m = min(bitlen a, bitlen b),   Φ = L + m
```

and write `Φ'` for the same quantity at the end of the batch.

`E = (k_remain + 1)/2` is the reference position of §3's schedule, `W` above the
extraction position. It starts at `N`, falls by `S/2` per batch, and may land on
a half-bit. The invariant carried across batches is:

> **(INV)** `Φ ≤ 2E`.

It holds initially, since `E₀ = N ≥ L ≥ m` gives `Φ₀ ≤ 2N = 2E₀`. Because `m` is
the smaller of the two, the invariant also gives `m ≤ Φ/2 ≤ E`: the smaller
operand never sits above the schedule position. Theorem 1 needs only this
consequence; Theorem 2 needs the invariant itself. Any slack `2E − Φ` is what a
batch can draw on when its own reduction falls short (A.5).

Absolute bit positions are not the natural coordinates for the window. Write

```
    λ = E − L
```

for how far the larger operand sits below the schedule. The extraction window
reads `3W` bits from a base `P ≤ E − W` (see the diagram in §3). The larger
operand reaches `L − P` bits above that base, so the window holds
`j = min(W, L − P)` significant bits of it. Because the base is at least a full
word below `E`, the window falls short of a full word by no more than the larger
operand sits below the schedule:

```
    W − j  ≤  max(0, λ)
```

The compact words' least significant bit has weight `2^σ` with `σ = L − j`: the
normalizing shift puts the larger operand's top bit at the top of the high word
when the window is full, and otherwise stops at the window base. `j ≤ 0` means
both operands lie below the window and the compact words are zero; an operand
with bits above the trip's top raises the overflow flag of §3.

This describes Stage 1's scheduled extraction. In Stage 2 the extraction finds
the operands' actual top bits, so the window is always full (`j = W`) and the
operands are non-negative; everything below applies with `j = W`. Stage 3's
single-word steps are exact, so each removes at least one bit while `E` falls by
`½`, which preserves (INV) step by step.

Write `e_a` for `a`'s representation error in units of `2^σ` — its compact word
`ah` minus `a/2^σ` — and likewise `e_b`, and let `D = e_a − e_b`. Then

```
    (a − b) / 2^σ  =  (ah − bh) − D
```

and Appendix B bounds `|D|` by `Δ = Δ(S)` throughout a batch.

Two constants follow from the implementation's parameters.
`q = SPLIT_THRESHOLD_BITS`, so the halt threshold is `2^q`; and
`g = bitlen(2^q + Δ)`, the gap bits, so that the separation a banded step can
leave is below `2^(σ + g)`. Appendix B shows `S = W − g` at both word sizes.

A step is **banded** if it is a subtract step — `a` odd, so a comparison is made
— whose compact words at that step satisfy `|ah − bh| < 2^q`, and which is not
one of the exact cases of §2. The first banded step of a Jacobi batch is the
last one it executes: the halt test of §2 turns every later step into a no-op.
Banded steps are also the only ones whose comparison can be wrong (A.5).

The symbols used in the rest of this appendix, in the order defined:

| Symbol | Meaning |
|---|---|
| `L`, `m` | larger and smaller operand bit length at the start of a batch |
| `Φ = L + m` | the quantity (INV) bounds; `Φ'` is its value at the end of the batch |
| `E = (k_remain + 1)/2` | schedule position; it falls by `S/2` per batch |
| `λ = E − L` | how far the larger operand sits below the schedule |
| `P` | base of the extraction window, a limb boundary at or below `E − W` |
| `j` | significant bits of the larger operand in the window; `W − j ≤ max(0, λ)` |
| `σ = L − j` | the compact words' unit is `2^σ` |
| `e_a`, `e_b`, `D` | representation errors in units of `2^σ`, and their difference |
| `Δ = Δ(S)` | bound on `\|D\|` over a batch (Appendix B) |
| `q = SPLIT_THRESHOLD_BITS` | the halt threshold is `2^q` |
| `g = bitlen(2^q + Δ)` | gap bits: a banded step leaves the operands within `2^(σ + g)`; `S = W − g` |
| banded step | a subtract step with `\|ah − bh\| < 2^q` |

### A.2 Theorem 1 — soundness of the extraction

*Under (INV), at most one operand can raise the overflow flag of §3. The
substitution that branch performs therefore yields one `MAX` and one zero, the
comparison it produces is correct at every step of the batch, and no step of
the batch is banded.*

*Proof.* A flag fires exactly when an operand has bits above the top of the
trip. The trip's top sits at or above `E + S`, and by (INV) the smaller
operand satisfies `m ≤ E < E + S`, so its flag cannot fire; at most one flag is
raised, and the substitution yields exactly one `MAX` and one zero rather than
two of either.

The partner's compact word is then zero, and it stays zero for the whole batch:
a halving keeps a zero word zero, and a subtract either reduces the marker's
word or swaps the zero word, unchanged, into `b`.

The marker's word loses at most one bit per step, so at the batch's last
comparison it is still at least `MAX >> (S − 1) = 2^(q+2) − 1`, using
`S = W − q − 1`. That is above the halt threshold `2^q`, so no step of the batch
is banded, and every comparison reads the flagged operand as the larger.

That reading is correct throughout. The trip's top lies above `E + W` (§3), so
the flagged operand starts above `2^(E+W)`. Each step leaves it alone, halves
it, or replaces it by half its difference with a partner below `2^E`, so after
`k` steps it is still above `2^(E+W−k) − 2^E`. Every comparison happens within
the first `S − 1 < W − 1` steps, where that exceeds `2^E`, while the partner is
below `2^m ≤ 2^E`. ∎

The margin is exactly one bit: the marker clears `2^(q+1)` but not `2^(q+2)`, so
raising `SPLIT_THRESHOLD_BITS` by more than one without shortening the batch
would break this theorem.

The certainty here rests on the zero, not on any gap between magnitudes — one
side stays exactly zero however far the marker has been halved. That is what
allows the substitution to discard the flagged operand's real bits, and with
them the leading-zero scan this branch would otherwise need (§3): the marker
carries dominance, not magnitude.

### A.3 Theorem 2 — termination

*If (INV) holds at the start of every batch that begins with `m > 0`, the
algorithm has reached `m = 0` by the time the step budget is exhausted.*

*Proof.* Once `m = 0` it stays 0: the zero operand is even, so later steps only
halve it. Otherwise every batch begins with `m > 0` and (INV) holds throughout.
`E = (k_remain + 1)/2` is `N` at the start and falls by `S/2` per batch along
with the counter. The last batch may overshoot, taking `k_remain` below zero;
either way `E ≤ 1/2` once the budget is spent, and (INV) gives `Φ ≤ 2E ≤ 1`.
Since `L ≥ m ≥ 0`, that forces `m = 0`. Nothing in this depends on how the
budget was split between stages, only that each batch decrements the counter by
the steps it was charged for.

The surviving operand is the gcd. Every elementary step either halves an operand
or replaces one by half the difference of the two, and the halving is always of
an even value: the parity comes from the exact low word, so a step halves `a`
only when `a` is even, and subtracts only when both are odd, in which case the
difference is even. Every such step preserves `gcd(a, b)`, whether or not the
comparison that chose it was correct, so reaching `m = 0` by any route leaves
the other operand equal to the gcd. ∎

### A.4 Three facts about a batch

The first says the wide update tracks magnitudes, the second bounds what a batch
can undo, and the third says when a comparison can be wrong at all.

> **(MAG)** Every executed step before the first banded step (A.1) replaces
> the larger *magnitude* by `(larger − smaller)/2`, and the first banded step
> replaces one of the two by `|a − b|/2`, never by `(a + b)/2`.

*Why it holds.* The matrix `M` a batch accumulates is a magnitude-space matrix,
`(a', b')ᵀ = M·(a, b)ᵀ`, while the wide operands carry the deferred signs,
`diag(s_a, s_b)·(a, b)ᵀ` with `s_a, s_b ∈ {±1}`. Writing `diag(s)` for that sign
matrix, `diag(s)² = I` gives

```
    M·diag(s) · diag(s)·(a, b)ᵀ  =  M·(a, b)ᵀ  =  (a', b')ᵀ
```

so folding the signs into `M`'s columns — one conditional negation each — makes
the wide update compute the magnitude-space result exactly, and the batch
proceeds as though begun on the non-negative pair `(a, b)`. A correct step
replaces the larger of a non-negative pair by their halved difference and leaves
both non-negative, so everything up to the first banded step is non-negative.

*Why it is scoped.* Past that step the claim is dropped, and must be: a
divergence leaves one operand negative, after which the wide subtraction adds
magnitudes. A.5 asks for no magnitude-subtraction bound beyond that point —
only the non-regrowth argument, which runs on the compact words.

Without the fold the wide update would compute `(a + b)/2` where it meant
`(a − b)/2` whenever the two signs differ. That would not be a correctness bug —
`gcd(a + b, b) = gcd(a, b)`, so the answer survives — but the magnitude *grows*,
and with it `Φ`. Lemma 1 and every reduction bound in A.5 would fail silently,
on inputs that still produce the right gcd.

**Lemma 1 (monotonicity).** *`L` is non-increasing across a batch, whether or
not its comparisons were correct.*

*Proof.* A step replaces one operand `u` by `u/2` or by `(u ± v)/2`, and leaves
the other unchanged. Both are below `2^L`, the second because
`|u ± v| ≤ |u| + |v| < 2^{L+1}`, so the replacement's bit length is at most `L`.
∎

Lemma 1 holds whether a step subtracts or adds magnitudes. The banked reduction
of A.5 is a stronger claim, that the replaced operand drops to half the
*difference* of the two, and it needs (MAG).

**Lemma 2 (divergence threshold).** *A step's comparison can be wrong only if
`|ah − bh| ≤ |D|`.*

*Proof.* Suppose the compact words say `ah ≥ bh` while truly `a < b`. Then
`(a − b)/2^σ = (ah − bh) − D < 0`, so `0 ≤ ah − bh < D`; the mirror case gives
`D ≤ ah − bh < 0`. Either way `|ah − bh| ≤ |D|`. ∎

`D` has two sources, both accounted for in Appendix B. The first is the
**initial truncation**. The compact word is the true value rounded down to the
window's unit, so each error starts in `(−1, 0]`; the extraction caps its
normalizing shift so that no bits below the window base are fabricated, which
keeps this to one ulp. A negative operand `a` is read by bitwise complement,
which is `|a| − 1` and can cost one more, so each error starts in `(−2, 0]`, the
same on both paths, and every batch's read re-establishes it from scratch. The
second is **accumulation**: each halving drops a bit from a compact word that
the true operand keeps, so the errors evolve by a small affine recurrence that
depends on the sequence of step kinds and never on the operands' values.

### A.5 Theorem 3 — the invariant is preserved

**Theorem 3.** *(INV) is preserved by every batch that begins with `m > 0`.*

*Proof.* After the batch `E` has fallen by `S/2`, so we must show

```
    Φ'  ≤  2E − S
```

By (INV) at the start of the batch, `Φ ≤ 2E`, so it is always enough to show
`Φ' ≤ Φ − S`: a batch owes `S` bits of reduction, less any slack `2E − Φ` it
starts with.

**The band.** By Lemma 2 a wrong comparison needs `|ah − bh| ≤ |D| ≤ Δ < 2^q`,
so

> every step before the first banded step has a correct comparison,

on either path. The halt test fires exactly at the first banded step, so the GCD
and Jacobi paths agree about which step matters and differ only in what they do
after it.

**No banded step.** Then every comparison in the batch is correct and all `S`
steps execute. Each either halves an even operand or replaces the larger by
`(larger − smaller)/2`; either way one operand loses at least a bit and the
other is untouched, so `Φ' ≤ Φ − S`, unless an operand reaches zero during the
batch, after which nothing more is owed (A.3). This includes the overflow case,
which Theorem 1 shows is never banded.

**A banded step.** Otherwise let `d` be the first. The step is executed on both
paths — on the halting path `unhalted` goes false only for the steps *after* it
— and by (MAG) it replaces one operand by `|a − b|/2`, whether or not its
comparison was correct. With `(a − b)/2^σ = (ah − bh) − D`,

```
    |a − b| / 2^σ  ≤  |ah − bh| + |D|  ≤  (2^q − 1) + Δ  <  2^g
```

The first term is the compact words' separation, at most `2^q − 1` at a banded
step; the second is the error in that separation. So the replaced operand ends
below `2^(σ + g − 1)`, and `σ + g − 1 = L − (S + 1) + (W − j)`. Write `ℓ` for
the bit length of the operand the step leaves alone. Just after step `d`, then,

```
    Φ  ≤  L − (S + 1) + (W − j) + ℓ                                  (‡)
```

At a full window, and with `ℓ = m`, that is the starting `Φ` less `S + 1`: one
banded step removes one bit more than the batch owes, less any shortfall `W − j`
in the window and any growth `ℓ − m` of the operand left alone. What remains is
to bound `ℓ`. That depends on where the larger operand sits relative to the
schedule and, when it is above, on how far apart the two operands are, which
gives three cases:

| Case | Condition | How `ℓ` is bounded |
|---|---|---|
| 1 | `L ≤ E + ½` | `ℓ ≤ L` (Lemma 1); the larger operand's distance below `E` covers any window shortfall |
| 2 | `L ≥ E + 1` and `m ≥ L − S` (operands close) | `ℓ ≤ m + 1` from the step's history |
| 3 | `L ≥ E + 1` and `m < L − S` (operands far apart) | `ℓ ≤ m`: the smaller operand is never replaced |

They cover every configuration, since `λ = E − L` is a multiple of `½`.

**Case 1: `L ≤ E + ½`, i.e. `2λ + 1 ≥ 0`.** Lemma 1 gives `ℓ ≤ L`, so (‡) is at
most `2L − (S + 1) + (W − j)`. That is at most `2E − S` exactly when
`2λ + 1 ≥ W − j`, and since `W − j ≤ max(0, λ)` it is enough that

```
    2λ + 1  ≥  max(0, λ)
```

Against `0` this is the case condition itself, and against `λ` it reads
`λ ≥ −1`, which holds too. So when the larger operand is at or below the
schedule, its distance below `E` pays for any shortfall in the window, operands
entirely below it included. It does so because the window base sits a full word
below `E`: with the base only `o < W` bits below `E`, the shortfall bound
becomes `max(0, λ + W − o)`, and the condition `λ ≥ W − o − 1` fails when the
larger operand sits just below the schedule.

**Setup for Cases 2 and 3.** Now `L ≥ E + 1`, i.e. `λ ≤ −1`, so `max(0, λ) = 0`:
the window is full, `j = W` and `σ + g = L − S`. It is not the overflow case,
which has no banded step (Theorem 1). Lemma 1's `ℓ ≤ L` is no longer enough,
since it would need Case 1's condition `2λ + 1 ≥ 0`, so `ℓ` is bounded from the
step's history instead. Every step before `d` was correct, and a correct step
never increases the smaller operand's bit length, so the smaller is still at
most `m` bits at step `d`. The banded step's bound `|a − b| < 2^(σ + g)` puts
the larger within `2^(σ + g)` of it. The operand the step leaves alone is one of
these two, so

```
    ℓ  ≤  max(m, L − S) + 1
```

The smaller operand's compact word carries `m − σ` significant bits, so
`m ≥ L − S` is exactly `m − σ ≥ g`: whether the smaller operand reaches `g` bits
inside the window decides which term wins, and which case applies.

**Case 2: `L ≥ E + 1` and `m ≥ L − S`, the operands close.** Then `ℓ ≤ m + 1`,
and (‡) gives `Φ ≤ L + m − S`, the starting `Φ` less `S`. The `+ 1` is the
straddle case: the two operands are close enough for drift to reverse them but
sit either side of a power of two, so the survivor can be one bit longer than
the batch's starting minimum. One banded step still covers the whole batch
without drawing on any slack, and it has to: (INV) allows `Φ = 2E` here (for
example `L = E + 1`, `m = E − 1`). This is what fixes `S` one below the longest
batch the drift bound certifies, since at `S = W − g + 1` this case would remove
only `S − 1` bits.

**Case 3: `L ≥ E + 1` and `m < L − S`, the operands far apart.** Write `x` and
`y` for the larger and smaller compact words, wherever they sit. The smaller
operand's compact word has fewer than `g` significant bits, so
`y ≤ 2^(g−1) − 1`, while `x` starts at least at `2^(W−1)`. If the larger operand
starts in `b`, the first subtract step swaps it into `a`, correctly, since the
gap is far above the band. From then on each step leaves `x` alone (while `y` is
halved), halves it, or replaces it by `⌊(x − y)/2⌋`, so `x + y` never falls
below half its value minus one, and after `k` steps `x ≥ 2^(W−1−k) − y − 2`. At
the batch's last step, `k = S − 1 = W − g − 1`, that is `x ≥ 2^g − y − 2 ≥ y`,
using `y ≤ 2^(g−1) − 1`. So **no comparison after the first can swap**, and the
smaller operand is never replaced for the whole batch; at most it is halved. (At
both word sizes the minimum of `x − y` over the batch is 2, at the final step
with `m − σ = g − 1`.) The operand the banded step leaves alone is therefore the
smaller one, whether or not the comparison was right, so `ℓ ≤ m`, and (‡) gives
`Φ ≤ L + m − S − 1`.

All three cases bound `Φ` at step `d + 1`. Two things remain: why a batch that
stops early is charged as though it ran to the end, and why the steps after `d`
cannot give the banked reduction back.

**A halted batch is charged in full.** The bound is taken at step `d + 1`, and
the paths differ only in what follows it. On the halting path nothing does —
`unhalted` is false, every remaining step is a no-op, and a no-op cannot change
`Φ` — so the bound at `d + 1` is the bound at the end of the batch. That is what
justifies charging a full batch of `S` steps to one that executed `d + 1 < S`:
what pays for a batch is the banded step's cancellation, not a step count.

**The banked reduction is kept.** On the GCD path the batch runs on after step
`d`, and two facts bound what the remaining steps can do.

First, by Lemma 1 applied from step `d + 1`, the larger operand never exceeds
its bit length at step `d + 1`.

Second, the smaller operand stays below `2^(σ + g − 1)`. The compact words run
an ordinary binary GCD on themselves, so the smaller compact word never grows: a
step either halves one word, or replaces the word the compact comparison judged
larger by half the difference, or (on a swap) keeps the word judged smaller and
replaces the other by half the difference. In each case the smaller of the two
is at most what it was. At step `d + 1` it is at most the word the banded step
produced, `⌊|ah − bh|/2⌋ ≤ 2^(q−1) − 1`.

The true operands follow their compact words closely: each error stays within
`(Δ + 1)/2` for the whole batch. Appendix B bounds only the difference `D`, but
by induction each single error is bounded too. It starts within `2`, which is at
most `(Δ + 1)/2` since `Δ ≥ 3` at both word sizes. A halving drops a bit
`γ ∈ {0, ½}` and gives `|e/2 − γ| ≤ |e|/2 + ½`; a subtract sets the stepped
error to `±(e_a − e_b)/2 − γ`, at most `(|D| + 1)/2`; and a swap only copies an
existing error. The same recurrence holds after a wrong comparison, because the
compact words and the true operands go through the same linear steps. So the
smaller true operand stays below `(2^(q−1) − 1 + (Δ + 1)/2)·2^σ`, which is under
`2^(σ + g − 1)` at both word sizes (Appendix B).

With these, each case's bound at step `d + 1` still holds at the end of the
batch:

- **Case 1.** The larger is at most `L` bits (Lemma 1 from the start of the
  batch) and the smaller at most `σ + g − 1`, which is what (‡) used with
  `ℓ ≤ L`.
- **Case 2.** At step `d + 1` the two operands have at most `σ + g − 1` and
  `ℓ ≤ m + 1` bits, and `σ + g − 1 = L − S − 1 < m`, so the larger is at most
  `m + 1` bits and stays there; with the smaller at most `σ + g − 1`, this is
  what (‡) used with `ℓ ≤ m + 1`.
- **Case 3.** No comparison after the first swaps, so the smaller operand is
  never replaced and keeps at most `m` bits. The other is at most
  `max(σ + g − 1, m) = σ + g − 1` bits by Lemma 1 from step `d + 1`, which is
  what (‡) used with `ℓ ≤ m`. ∎

With the base case `Φ₀ ≤ 2N`, this establishes (INV) for every batch that begins
with `m > 0`, which is what Theorems 1 and 2 rest on.

### A.6 Corollary — the operating window shrinks

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
`GcdPair::len` narrows on exactly this schedule and never on the operands'
actual values, so the narrowing itself stays public and data-independent.

## Appendix B. The drift bound

### B.1 The bound

`D` is the difference of the two operands' representation errors (A.1), and
`Δ(S)` the largest `|D|` reachable within a batch of `S` steps. Its ceiling
`T(S) = ⌈Δ(S)⌉` sets the halt threshold, `q = bitlen(T)`.

**Where the constant comes from.** `Δ(S)` is a maximum over an enumeration of
the error recurrence: a two-dimensional affine system with three
branch maps and a half-ulp injection per step, carried forward `S` steps as a
convex hull in exact rationals. The model over-approximates a concrete
execution — it lets every branch run at every step — so `Δ(S)` is a safe upper
bound rather than the implementation's exact reachable maximum, and the hull is
exact for the model, so the closed form below is exact rather than fitted.
B.2 gives the maps, the abstraction argument, and the certificate for the
two constants.

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

Solving `S + bitlen(T(S)) ≤ W − 1` for the largest self-consistent `S` gives the
batch length and threshold tabulated in §2 — `S = 58, T = 23` at `W = 64`,
`S = 27, T = 12` at `W = 32` — each landing on equality with the bound, as they
must if `S` is maximal subject to it. The `− 1` is the deliberate step given up:
solving against `W` instead would give `S = 59` and `S = 28`, one longer, and
Case 2 of A.5 shows what that step is worth.

The same inequality has a second reading, which is why one constant serves both
purposes. Rearranged as `S ≤ W − g`, with
`g = bitlen(2^SPLIT_THRESHOLD_BITS + Δ)`, it says the batch is strictly shorter
than what a single banded step can pay for at a full window; A.5 uses it in that
form to close the termination proof, and shows separately that a partial window
is paid for by the larger operand's distance below the schedule. Solved as
`S + bitlen(T) ≤ W`, it says the overflow marker of §3 stays above the halt
threshold for a whole batch (Theorem 1). The two differ by exactly one step at
both word sizes, and banking is the binding one, so the marker condition holds
with a step in hand.

### B.2 The enumeration

The verification behind B.1: the model whose maximum `Δ(S)` is, why that model
over-approximates a concrete batch safely, and the exact values at the two
supported word sizes. Notation is A.1's.

**The enumeration, concretely.** Track the pair of errors `(e_a, e_b)` of A.1,
with `D = e_a − e_b`. Lemma 2's discussion puts each in the half-open `(−2, 0]`;
the enumeration uses the **closed** `[−2, 0]`, which contains it, so what it
computes is an upper bound for the real reachable set and a maximum on a compact
one. Each step applies one of three maps, with `γ ∈ {0, ½}` the bit the
arithmetic shift drops:

```
    halve      a even          e_a' = e_a/2 − γ
    subtract   a odd, a ≥ b    e_a' = (e_a − e_b)/2 − γ
    swapsub    a odd, a < b    e_a' = (e_b − e_a)/2 − γ,   e_b' = e_a
```

Enumerate by carrying the convex hull of the reachable set forward `S` steps in
exact rationals, and read `Δ(S) = max|D|` off the vertices.

**The model over-approximates; the hull is exact for the model.** These are two
separate claims.

`R_concrete ⊆ R_abstract`. A concrete execution does not choose freely among the
three maps — parity fixes halve against subtract, the compact comparison fixes
which subtract ordering runs — whereas the enumeration allows all three at every
step, and widens the half-open `(−2, 0]` to the closed `[−2, 0]`. Both widenings
are sound because neither constraint restricts the adversary: the parity
sequence is a function of unconstrained operands, and both subtract orderings
are reachable wherever a comparison can be wrong, which is precisely where `D`
is large. So `Δ(S)` is a **safe upper bound** on the concrete drift, not the
implementation's exact reachable maximum.

For the model itself the hull loses nothing. Every map is affine, and the
dropped bit contributes a Minkowski sum with `[−½, 0]` — the convex hull of the
two values `γ` actually takes, so an extreme point always uses an endpoint,
never an interior `γ`. Since `conv(f(X)) = f(conv(X))` for affine `f`, hulling
after each step yields exactly `conv(R_abstract,t)`, whose extreme points belong
to the set hulled. `Δ(S)` is therefore attained within the model, which is what
makes the closed form in B.1 exact rather than merely a fitted bound. Nothing
depends on sampling.

**The constants, certified.** Everything A.5 needs is a rational
comparison at two values of `S`:

| `W` | `S` | `Δ(S)` exact | `T = ⌈Δ⌉` | `q` | `2^q` | `2^q + Δ` | `g` |
|---|---|---|---|---|---|---|---|
| 64 | 58 | `3186546936343924281 / 2^57` | 23 | 5 | 32 | 54.111 | 6 |
| 32 | 27 | `790393287 / 2^26` | 12 | 4 | 16 | 27.778 | 5 |

with `q = SPLIT_THRESHOLD_BITS = bitlen(T)`. Reading off: `Δ < 2^q` at both
sizes, with the tightest margin at `W = 64` (22.111 against 32); and
`2^q + Δ ≤ 2^g` with `g` as tabulated, the tightest again at `W = 64` (54.111
against 64). A.5's bound on the smaller operand after a banded step also needs
`2^(q−1) − 1 + (Δ + 1)/2 < 2^(g−1)`: 26.556 against 32 at `W = 64`, and 13.389
against 16 at `W = 32`. All three are exact rational facts about the two
numerators above.
