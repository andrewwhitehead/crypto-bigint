# The binary GCD implementation explained

This library previously contained two constant-time GCD engines: `safegcd`
(Bernstein–Yang, [https://gcd.cr.yp.to/papers.html#safegcd]) and `bingcd` (Pornin's optimized binary GCD,
[eprint 2020/972](https://eprint.iacr.org/2020/972)). Both are now replaced by a single engine under
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
a batch can run `GCD_BATCH_SIZE = 58` steps at `W = 64`. The
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
it, rather than repairing after the fact. In the code this is the `HALT` branch
of `partial_xgcd`:

```rust,ignore
let above_threshold = if HALT {
    word::choice_from_nz(hi_diff >> SPLIT_THRESHOLD_BITS).or(exact.or(a_odd.not()))
} else {
    Choice::TRUE
};
```

Below the threshold `active` goes false, and every following step in the batch
becomes a no-op via `active.select_u32`, rather than risk a step whose
direction can't be certified.

Two disjuncts widen that test, both covering cases where nothing needs
certifying. `exact` is set when the data-dependent extraction has returned the
operands' low words themselves — that is, when the operands have collapsed to a
single word, so the compact words *are* the true values, carry no
representation error, and have no band to clear. `a_odd.not()` covers a
halving step, whose direction comes from the exact low word rather than from a
comparison at all.

`SPLIT_THRESHOLD_BITS` is `bitlen(T)`, where `T` is the drift bound of §4's
Lemma 2 — enumerated exhaustively rather than guessed. Because drift
accumulates per step, `T` depends on the batch length, so the constant and the
batch size determine each other and are solved together in §4:

| | `W = 64` | `W = 32` | `W = 16` | `W = 8` |
|---|---|---|---|---|
| `T` | 21 | 11 | 6 | 3 |
| `SPLIT_THRESHOLD_BITS` | 5 | 4 | 3 | 2 |
| `GCD_BATCH_SIZE` | 58 | 27 | 12 | 5 |

The batch length is one step below the longest the drift bound would certify.
That step is the margin Pornin's parameters carry for free — his inner loop
runs `k − 1` iterations against a `k + 1`-bit approximation — and §4.6 shows
what it buys.

One batch length serves both paths. It is chosen so that a batch never asks for
more reduction than the drift bound can certify: §4.6 shows a single divergent
step read off a full extraction window banks `S + 1` bits, one more than the
batch owes, and that a window the operands do not fill loses banking more
slowly than it gains headroom, so the same length serves there too. The Jacobi path
additionally consults `T` at run time, as the halt test above; the GCD path
consults it only through the batch length.

## 3. Avoiding O(N) costs per round

Compared to `safegcd`, Pornin's algorithm carries several costs that scale with
the operand's full width, which is why it falls behind at larger sizes despite
needing fewer reduction steps. This implementation removes or reduces three of
them.

### The tracked window

`window_limbs` in `gcd.rs` is a limb-granularity bound: nothing at or above
that limb index is read or written during the round, in either loop
(`g.leading_mut(window_limbs)` and `f.leading_mut(window_limbs)` clip every
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
and two conditional swaps per limb, every round (see `top_window_pair`). At
small sizes the overhead is minor; at larger `N` it is not.

For tracked window sizes above `SMALL_THRESHOLD_LIMBS`, this implementation reads a fixed-size **3-limb
trip** instead, via `extract_pair_vartime_signed`,
at a position the round tracks directly. What is tracked is the *bottom* of the
extraction: `extract_pos` holds `E − W`, starting at `N − W` and descending a
fixed `S/2` bits every round, so `E_j = N − (S/2)·j` after `j` batches for the
reference position `E` that §4 reasons about.

`S` is odd at some word sizes, so `extract_pos` is held in **half-bits**: the
round subtracts `S` from `2·extract_pos` and takes the bit index as `bit2 >> 1`.
That keeps the schedule exact — no rounding drift accumulating across `B`
rounds — with one subtraction and a predicated borrow, and no division. No scan
is involved either way: every limb index the trip touches is public and
schedule-derived rather than data-derived.

Handing over `E − W` rather than `E` is what makes the trip wide enough at both
ends, and the offset is `W` rather than `S` for a reason the termination proof
needs. The trip begins at the limb containing that bit, so its base lies in
`(E − 2W, E − W]` and it spans `3W` bits from there. Its top therefore sits at
or above `E + W + 1`, clearing `E + S` by at least `W − S + 1` bits — 6, 5, 4
and 3 at the four word sizes. That margin is what Theorem 1 needs: an operand
can overflow the trip only if it exceeds `E + S`.

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

While Stage 1 runs, a negative operand's window is read by bitwise
complementing the extracted limbs in place (`!raw`), not by computing a real
two's-complement negation over the wide value. The rare case where a borrow
leaves the extracted trip off by one can only waste a round, never corrupt the
result (§4). Nothing has to scan the operand's sign to know which case applies.

That trick is scoped to Stage 1. The transition into Stage 2 performs the one
genuine full-width negation `f` ever needs; from there
`gcd_odd_small_with_budget` extracts with the exact, unsigned
`top_window_pair` and re-corrects both operands to non-negative every round via
`wrapping_apply_unsigned_shift` — cheap, because that window is capped at
`SMALL_THRESHOLD_LIMBS` (8) limbs rather than `N`. Past that point neither the
deferred sign nor the complement trick is needed.

## 4. The termination guarantee

The batched algorithm must halt within a fixed, public batch count

```
B = ⌈(2N − 1) / S⌉
```

matching Pornin's own `2N − 1` elementary-step bound (`iterations()` in
`bingcd.rs`), counted in batches of `S` steps rather than one step at a time.

Everything in this section hangs off a single invariant. Section 4.1 states it;
4.2 and 4.3 draw the two consequences the implementation needs — that the
extraction is sound, and that the algorithm terminates — both of which assume
the invariant holds. Sections 4.4 to 4.6 discharge that assumption, and 4.7
derives the shrinking window of §3 as a by-product.

### 4.1 Setup and the invariant

For operands `a`, `b` at the start of a batch, write

```
    L = max(bitlen a, bitlen b),   m = min(bitlen a, bitlen b),
    Φ = L + m,                     h = 2E − Φ,
```

where `E` is the reference position of §3's schedule — `S` above the tracked
`extract_pos`, which the round decrements by `S/2` in `gcd_odd_with_budget`,
tracked in half-bits so that an odd `S` stays exact (§3) — so
`E_j = N − (S/2)·j`, and `E_j` may land on a half-bit. Primes mark
post-batch quantities throughout, so `E' = E − S/2` and `ΔΦ = Φ − Φ'`.

Write `E_lo` and `E_hi` for the bottom and top of the window the extraction
reads. Only two
properties of them are used below, both established in §3:

```
    E_lo ≤ E − W,        E_hi > E + S
```

The first holds because the trip begins at the limb containing `E − W`; the
second because it spans `3W` bits from there, clearing `E + S` by at least 3.
The first is the load-bearing one: it is what converts an unfilled window into
a bound on `L`, and through it into headroom under (INV).
Define

```
    κ = L − E_lo
```

deliberately unclamped: `κ` may be negative, when both operands lie below the
window, or exceed the window width, when they overflow it. The clamped
`min(W, max(0, κ))` counts usable bits, but the arguments below need the
unclamped form.

The invariant carried across batches is:

> **(INV)** `Φ ≤ 2E`, equivalently `h ≥ 0`.

It holds initially, since `E₀ = N ≥ L ≥ m` gives `Φ₀ ≤ 2N = 2E₀`. Theorems 1
and 2 need only the weaker corollary `m ≤ E`, immediate from `m ≤ Φ/2 ≤ E`.

`h` is not slack to be discarded: §4.6 spends it, and the amount available is
what decides two of that section's three cases.

### 4.2 Theorem 1 — soundness of the extraction

*Under (INV), at most one operand can raise the overflow flag of §3. The
substitution that branch performs therefore yields one `MAX` and one zero, and
the comparison it produces is correct at every step of the batch.*

*Proof.* A flag fires exactly when an operand exceeds the top of the trip,
which sits at or above `E + S`. By (INV) the smaller operand satisfies
`m ≤ E < E + S`, so its flag cannot fire; at most one flag is raised, and the
substitution yields exactly one `MAX` and one zero rather than two of either.

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
    W =  8:  MAX >>  4 =  15  vs threshold  4
```

The two bits are not spare capacity to be reclaimed. One is the step §4.5 gives
up so that a divergence banks more than the batch owes; the other is the
inequality's own margin. What the coupling still forbids is raising the
threshold constant by more than two without shortening the batch in step —
which §4.5's admissible range would otherwise permit — since that would let the
marker fall inside the band before the batch ends, and the batch would truncate
on a comparison that was never uncertain.

### 4.3 Theorem 2 — termination

*Under (INV), the algorithm halts within `B = ⌈(2N−1)/S⌉` batches.*

*Proof.* The half-bit schedule is exact, so `E_j = N − (S/2)·j` and `E_B ≤ 0`
after `B` batches. Applying (INV) there gives `m ≤ 0`, hence `m = 0`: one operand has become zero and the other is the gcd. Every elementary step replaces one
operand by an integer combination of the two and halves it, and every such step
preserves `gcd(a, b)` whether or not the comparison that chose it was correct
— so reaching `m = 0` by any route leaves the surviving operand equal to the
gcd. ∎

### 4.4 Two lemmas

**Lemma 1 (monotonicity).** *`L` is non-increasing across a batch, whether or
not its comparisons were correct.*

*Proof.* A step replaces one operand `u` by `(u ± v)/2` and leaves the other
unchanged. Since `|u ± v| ≤ |u| + |v| ≤ 2^{L+1}`, the replacement has bit
length at most `L`. ∎

**Lemma 2 (divergence threshold).** *A step's comparison can be wrong only if
the two compact words differ by less than `T(S)`, the width of the
representation-error interval after `S` steps.*

*Proof.* Write `P = 2^σ` for the weight of the compact words' least significant
bit, and `ρ = (true slot value) − (compact word)·P` for the representation
error. A comparison is wrong when `|x| ≥ |y|` holds for the compact words while
`|A| < |B|` holds for the true operands, which requires
`(|x| − |y|)·P < ρ_b − ρ_a`. The threshold is the width of the `ρ` interval,
which has two sources.

The first is the **initial truncation**, worth one ulp. On the Jacobi path —
the only path that consults this threshold — each batch begins from a
top-bit-aligned window over non-negative operands, both guaranteed by the fused
repair of §5, so each compact word is `⌊x / 2^σ⌋` with `σ = L − W` and the
error interval is `[0, 1)` per operand.

The second is **accumulation**. Decomposing `ρ_j = P·F_j + (u₀ρ₀ᵃ + v₀ρ₀ᵇ)/2^j`,
the second term is at most one ulp, since each row of the batch matrix
satisfies `|u| + |v| ≤ 2^j`. The first term `F` — the accumulated floor error
of the compact recurrence — obeys the same recurrence as the compact words
themselves and depends only on the compact start state and the parity sequence,
never on the operands. It can therefore be bounded by exhaustive enumeration of
a fixed-width recurrence rather than a search over operand space. ∎

### 4.5 The drift bound

Two properties of that enumeration matter for the constant it produces.

**The horizon is the batch length.** `F` accumulates, so the bound grows with
the number of steps enumerated, and a bound read off at a shorter horizon than
the batch actually runs is not a bound. Writing `D` for the difference of the
two errors and `Δ(S) = max|D|` over `S` steps from a one-ulp initial interval,

```
    Δ(S) = S/3 + 10/9 − O(2^−S),        T(S) = ⌈(3S + 10)/9⌉
```

The linear growth is structural rather than incidental. The step map's
characteristic roots are `1/2` and `−1`, and the `−1` eigenvector is `D`
itself, so the half-ulp injected by each shift-out lands on a neutral mode and
resonates instead of damping. The extremal trajectory is a run of swap-subtract
steps with the dropped bit alternating; the slope `1/3` is `(2/3) × (1/2)` —
half the injections surviving the sign flip, projected onto `D`. Only the
constant `10/9` depends on the initial interval, and hence on the extraction.

**A larger initial interval would compound, not add.** Anything that widens the
starting interval is amplified by the same dynamics that amplify the injected
error, so for an interval of `q` ulps the band is neither `q + Δ(S)` nor
`q·Δ(S)`:

| `q` | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| band at `S = 27` | 10.11 | 11.78 | 15.44 | 23.11 |
| band at `S = 58` | 20.44 | 22.11 | 25.78 | 33.44 |

This is the margin that would be spent by moving the Jacobi path onto a
different extraction. A signed operand read by bitwise complement (§3) costs
`q = 2`, since the one's complement is `|x| − 1`; a window the operands don't
fill, normalized without a cap on the shift, costs `q = 2^k` for a shortfall of
`k` bits. Both thresholds still cover `q = 2`, and `bitlen(T)` is unchanged, so
the batch sizes would survive signed operands — but at `W = 32` a one-bit
window shortfall is the end of it. The scheduled extraction guards against this
by capping its normalizing shift at `min(clz(a_trip | b_trip), 2W)`, which
holds `σ` at or above the window base so that no bits below the base are
fabricated.

Solving `S + bitlen(T(S)) ≤ W − 1` for the largest self-consistent `S` gives the
batch length and threshold tabulated in §2 — `S = 58, T = 21` at `W = 64`,
`S = 27, T = 11` at `W = 32` — each landing on equality with the bound, as they
must if `S` is maximal subject to it. The `− 1` is the deliberate step given up:
solving against `W` instead would give `S = 59` and `S = 28`, one longer, and
§4.6 shows what that step is worth. Note that `T` is unchanged by the
shortening at every word size except `W = 8`, where the shorter batch
accumulates less drift and `T` falls from 4 to 3.

The same inequality has a second reading, which is why one constant serves both
purposes. Rearranged as `S ≤ W − bitlen(T + Δ)`, it says the batch is strictly
shorter than what a single divergent step can pay for at a full window; §4.6
uses it in that form to close the termination proof, and shows separately that
a partial window pays for itself out of headroom. The cost of the strictness is
one step per batch, which `B = ⌈(2N − 1)/S⌉` mostly absorbs: unchanged at
`N = 256` (9 batches) and `N = 512` (18), one batch more at 384, 1024 and 2048,
three more at 4096. Solved as `S + bitlen(T) ≤ W`, it says the
threshold and the batch length together fit the word, which is what makes the
halt test sound. Since `bitlen(T + Δ) = bitlen(T) + 1` at all four word sizes, the two
differ by exactly one step and banking is the binding one: a batch length whose
reduction a single divergence covers certifies its own comparisons with a bit
in hand.

The shift form of the test in §2 makes the effective threshold
`2^SPLIT_THRESHOLD_BITS − 1`, i.e. 31 and 15; both sit inside the admissible
range `[⌈T⌉, 2^{W+1−S} − T]`, so the power-of-two rounding costs no margin.

### 4.6 Theorem 3 — the invariant is preserved

**Lemma 3 (window position).** *`E − L ≥ W − κ`.*

*Proof.* `κ = L − E_lo` and `E_lo ≤ E − W`, so `E − L = (E − E_lo) − κ ≥ W − κ`. ∎

Read the other way, every bit by which the operands fall short of filling the
window is a bit by which `L` sits below `E`, and hence — since `m ≤ L` — two
bits of headroom `h = 2E − Φ ≥ 2(E − L) ≥ 2(W − κ)` under the induction
hypothesis. That doubling is what Case (iii) below runs on.

**Theorem 3.** *(INV) is preserved by every batch.*

*Proof.* We must show `Φ' ≤ 2E'`, that is

```
    ΔΦ  ≥  S − h,        h = 2E − Φ ≥ 0
```

so what a batch owes is `S` less whatever headroom the induction hypothesis
already leaves it. Write

```
    β = bitlen(T + Δ),        so  S = W − β
```

which is §4.5's inequality at equality, holding at all four word sizes. A
divergent step read off a window carrying `j` significant bits banks
`C(j) = j − β + 1`, so `C(W) = S + 1`: at a full window a divergence banks one
bit more than the batch owes.

The number of significant bits is `j = min(W, κ)`: the normalizing shift puts
`σ = L − W` when the larger operand reaches the top of the window, and §3's cap
holds `σ = E_lo` when it does not, leaving `W − κ` leading zeros rather than
fabricating bits below the base. Three cases follow, on `κ`.

**Case (i): `κ ≤ W − S/2`.** By Lemma 3, `L ≤ E − S/2 = E'`, and since
`m' ≤ L' ≤ L` by Lemma 1,

```
    Φ' = L' + m' ≤ 2L' ≤ 2L ≤ 2E'.
```

The batch need achieve nothing at all. This covers `κ ≤ 0`, where the operands
lie entirely below the window, both compact words are zero, and no comparison
carries any information — monotonicity alone carries it.

**No divergence.** For the remaining `κ`, if every comparison in the batch is
correct, each step replaces the larger operand by `(larger − smaller)/2`,
strictly reducing its bit length while the other is untouched, so `Φ` falls by
at least one per step and `ΔΦ ≥ S ≥ S − h`.

**Divergence.** Otherwise let `d` be the first step whose comparison is wrong,
and write `ℓ` for the larger operand's bit length there. By Lemma 2 the compact
words were within `T`, so at the common scale `2^σ` the true operands satisfy
`|a − b| < (T + Δ)·2^σ`. Two things follow. The smaller has bit length `ℓ − e`
with `e ∈ {0, 1}` — it exceeds `2^(σ + j − 1) − 2^(σ + β)`, which is at least
`2^(σ + j − 2)` because `j > W − S/2 ≥ β + 2` in the cases that remain, so the
two cannot fall two binades apart. And `ℓ − e ≤ m₀`, the smaller operand's
length at the batch's start — every step before `d` was correct, and a correct
step replaces the larger and leaves the smaller untouched, so the minimum
cannot have grown. The step itself leaves `|a|` below `(T + Δ)·2^σ/2`, which
with `σ = L − j` is at most `L − C(j)` bits. Therefore
`Φ_{d+1} ≤ (L − C(j)) + ℓ`, and since `Φ = L + m₀` at the start,

```
    ΔΦ  ≥  (L + m₀) − (L − C(j)) − ℓ  =  C(j) + (m₀ − ℓ)  ≥  C(j) − e
```

`e = 1` is the operands straddling a power of two: the larger sits just above
it, the smaller just below, close enough for drift to reverse them. It costs a
bit, because the surviving modulus is then one longer than the minimum the
batch started with.

**Case (ii): `κ ≥ W`.** The window is full, `j = W`, so
`ΔΦ ≥ C(W) − e = S + 1 − e ≥ S`. One divergent step covers the batch's entire
quota and the straddle bit besides, with no appeal to `h` at all.

**Case (iii): `W − S/2 < κ < W`.** The window is short by `W − κ` bits and the
step banks `C(κ) = S + 1 − (W − κ)`, so it falls short of the quota by
`(W − κ) − 1`. By Lemma 3 that same shortfall has already bought
`h ≥ 2(W − κ)`, so

```
    ΔΦ  ≥  S + 1 − (W − κ) − e  ≥  S − h
```

since `(W − κ) + e − 1 ≤ 2(W − κ)` for `W − κ ≥ 1` and `e ≤ 1`. The case closes
with a factor of two in hand, and the straddle bit is absorbed along with
it. ∎

The `W − κ` cancellation is what fixes the extraction offset at `W` rather than
`S` (§3). Banking is linear in the window's fill and headroom is twice linear
in it, so an unfilled window is self-financing — but only once `E_lo` is a full
word below `E`. At an offset of `S` the headroom would be `2(S − κ)` against
the same deficit, and Case (iii) would fail for `κ` just below `W` — the band
`2S − W < κ < W`, which is non-empty at every word size — reachable at any
batch where `E − S` happens to land near a limb boundary.

**Why `S` is one below the maximum.** Case (ii) is the only case with no
headroom of its own to spend — `h = 0` is admissible under (INV), and batch 0
realizes it, since `Φ₀ = 2N` — so it is the case that has to close on the
banking bound alone. At `S = W − β + 1`, the longest batch the drift bound
certifies, `C(W)` would equal `S` exactly and the straddle bit would have
nothing to come out of. One step shorter and `C(W) = S + 1` covers it
unconditionally.

That is also where Pornin's proof sits. His inner loop runs `k − 1` steps
against a `k + 1`-bit approximation whose error, being glued, does not
accumulate, so his banking bound exceeds his batch length by a bit and his
Appendix A.3 never has to ask whether the two operands share a bit length at a
divergence. The margin is incidental there — his batch length is capped by the
`k − 1` exact low bits, not by banking — and deliberate here, because the split
form's whole purpose is to push `S` up until banking is what binds. Setting
`S = W − SPLIT_THRESHOLD_BITS − 1` restores exactly his margin at exactly the
point his argument uses it.

Widening (INV) is not an alternative: the tight configuration scales with the
slack, so `Φ ≤ 2E + c` reproduces the same one-bit shortfall for every `c`. The
step has to come out of `S`.

What does still need saying is that the banked reduction cannot be given back,
since the bound above is taken at step `d + 1` rather than at the end of the
batch. It cannot, for two reasons that compose. The compact words are running
an ordinary binary GCD on themselves — self-consistent and non-negative,
because each step subtracts whichever of the two the *compact* comparison
judged smaller — so by Lemma 1 applied to them, neither the larger nor the
smaller ever grows. And Lemma 2's error accounting holds at every step of the
batch, so `|x| ≤ (|x_hi| + Δ)·2^σ` throughout. A true operand can therefore
regrow only if its compact word regrows, which the first point forbids.

This is Pornin's Appendix A.3 barrier with one substitution: his level comes
from the glued form's own error bound `2^{n−k−1}`, ours from Lemma 2's `Δ`, and
the barrier moves with the constant. He reaches it by a different route — after
a divergence the two operands have opposite signs, and that persists, since
halving preserves a sign and subtracting a negative from a positive leaves a
positive. The split form has the compact words available as independent
registers, so the monotonicity route is the more direct one here; the
conclusions agree.

One difference from Pornin is worth noting, since it explains why `S` is what
it is. His inner loop runs `k − 1` iterations against a `2k`-bit register, and
at that length the divergent step covers his quota with a bit to spare, so his
proof needs nothing from the rest of the batch. The split form gets its extra
steps by dropping the packed representation, which costs precision, so the
length at which a divergent step still covers the quota is `C` rather than the
register width. Choosing `S = C` keeps the same one-step proof; choosing `S`
larger — `W − 2`, say, which the low word and the coefficients would both
permit — would leave a three-bit gap at `W = 64` needing an argument about the
post-divergence tail. The batch length is set where the proof stays short.

Pornin also reads the batch's closing conditional negation as a hidden extra
iteration: `S + 1` iterations' worth of work to guarantee `S` bits, the last
disguised as the sign fixup. The extra iteration is consumed by the guarantee
rather than added to it — `Φ` is defined on magnitudes, and `a = |a|; b = |b|`
changes signs and no bit lengths, contributing exactly zero to `ΔΦ`. That is
why the batch budget divides `2N − 1` by `S` and not `S + 1`; the first version
of Pornin's note made the more aggressive claim, and its errata records the
counterexample.

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

Integrating over the run: the first half occupies `B/2` batches at full width
`N`, the second falls linearly from `N` to `0` (mean `N/2`), for total wide
work `(B/2)·N + (B/2)·(N/2) = (3/4)·B·N`. That is the 25% reduction quoted in
§3. `window_limbs` narrows on exactly this schedule and never on the operands'
actual values, so the narrowing itself stays public and data-independent.

## 5. The Jacobi symbol and the `b -= 2|a|` correction

The Jacobi/Kronecker symbol runs the same batched split-form loop as GCD, but
in the truncating (`HALT = true`) form of §2, since every executed step's swap
has to be certifiably correct to keep the running reciprocity sign
trustworthy.

Truncating provides an additional guarantee: if a batch diverges at
all, it diverges on exactly its **last** executed step, because the batch halts
the moment it does. That is what makes a cheap, exact repair possible. Where
GCD uses a plain `a = |a|; b = |b|`, the Jacobi path applies

```
a = |a|; b = b - 2|a|
```

to undo the single mis-swapped step. GCD's repair would not be sound here: the
symbol is not invariant under `b → b + 2a` the way `gcd(a, b + 2a) = gcd(a, b)`
is, which is why the two variants keep separate repair paths.

The correction is not a separate pass either. Rather than negate first and then
re-scan the operand for its next top-bit window,
`jacobi_correct_and_extract` fuses the two: a single
low-to-high sweep over the operand's limbs applies `b -= (a << 1)` — masked to
a no-op when no correction is needed — and simultaneously latches the
top-bit-aligned word pair the *next* round's extraction needs. The same motivation
as §3's avoided top-word scan, applied to the repair step.

That fusion is also what supplies Lemma 2's precondition. Because the sweep
resolves the sign and latches a top-bit-aligned pair in one pass, every Jacobi
batch starts from a full window over non-negative operands, so the drift
enumeration begins from a one-ulp interval per operand — with no dependence on
where a scheduled window would have fallen, or on a deferred sign. It also puts
the Jacobi path permanently in Case (ii) of §4.6, since `κ = W` by
construction; Cases (i) and (iii) exist for the scheduled window alone. The Jacobi
path consequently does not use the scheduled trip or the shrinking window of
§3, and the `κ` analysis of §4.6 applies to the GCD path alone.

## 6. The variable-time implementation

The constant-time paths above are constrained to a schedule that depends only
on public bit widths, precisely so their memory access pattern and control flow
reveal nothing about the operands. The variable-time implementation in
[`vartime.rs`](../src/modular/gcd/vartime.rs) is under no such constraint, and
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
`b -= 2|a|` repair are otherwise the same building blocks as the constant-time
engine. What changes is only how aggressively the schedule driving them is
allowed to react to the actual operands.
