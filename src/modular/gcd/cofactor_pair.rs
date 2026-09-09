use super::{ExtendedIntRef, SignedLimbMatrix, extended_int_ref::hi_overflow_vartime};
use crate::{Choice, Limb, Odd, UintRef};

/// Tracks a pair of Bezout-style coefficients (`u`, `v`) mod a fixed odd modulus `y`, updated in
/// lockstep with a caller's own running `(g, f)`-style GCD state by feeding it each round's
/// coefficient-update matrix. Shared by both the constant-time and vartime xgcd engines.
pub struct CofactorPair<'a> {
    /// Low limbs of the first tracked coefficient.
    pub u: &'a mut UintRef,
    pub u_hi: Limb,
    /// Low limbs of the second tracked coefficient.
    pub v: &'a mut UintRef,
    pub v_hi: Limb,
    /// Current width (in limbs) of the live `u`/`v` window.
    pub len: usize,
    /// Power-of-two divisor (mod `y`) owed to `u`/`v` but not yet applied.
    pub k: u32,
    /// The part of that divisor accumulated *before* the window filled, split out from [`Self::k`]
    /// and never paid during the loop -- see [`Self::apply_matrix`].
    pub k_deferred: u32,
    /// Constant-time-only growth-schedule state: bits of headroom left in the live window's own
    /// spare `hi` limb before another limb must be pulled in.
    pub cap_remain: u32,
    /// Fixed odd modulus `u`/`v`'s coefficients are tracked relative to.
    pub y: &'a Odd<UintRef>,
    /// `y`'s mod-limb inverse, as returned by `Odd<UintRef>::invert_mod_limb`.
    pub y_inv: Limb,
}

impl<'a> CofactorPair<'a> {
    /// Starts tracking `u`, `v` mod `y`, with no pending shift (`k = 0`).
    ///
    /// Seeds `u` at `1`.
    /// Seeds `v` at `0`. This is the coefficient paired with `b`, and the one [`Self::finalize`]
    /// actually returns.
    ///
    /// The starting window width starts at 0.
    #[allow(clippy::cast_possible_truncation)]
    pub const fn new(
        u: &'a mut UintRef,
        v: &'a mut UintRef,
        y: &'a Odd<UintRef>,
        y_inv: Limb,
    ) -> Self {
        assert!(u.nlimbs() == v.nlimbs());
        u.set_from_limb(Limb::ONE);
        v.fill(Limb::ZERO);
        Self {
            u,
            u_hi: Limb::ZERO,
            v,
            v_hi: Limb::ZERO,
            len: 0,
            k: 0,
            k_deferred: 0,
            cap_remain: 0,
            y,
            y_inv,
        }
    }

    /// Whether the tracked window has grown to its full, fixed width, so from here on every round
    /// must reduce instead.
    #[inline(always)]
    pub const fn is_full(&self) -> bool {
        self.len == self.u.nlimbs()
    }

    /// Extends the live window up to `target` limbs (fewer if that would exceed `u`/`v`'s backing
    /// width), moving each `hi` down into the newly included limb and re-deriving the next `hi` as
    /// its plain sign extension.
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    const fn grow_to(&mut self, target: usize) {
        let target = if target > self.u.nlimbs() {
            self.u.nlimbs()
        } else {
            target
        };
        let (u_sgn, v_sgn) = (
            self.u_hi.shr(Limb::HI_BIT).wrapping_neg(),
            self.v_hi.shr(Limb::HI_BIT).wrapping_neg(),
        );
        while self.len < target {
            self.u.limbs[self.len] = self.u_hi;
            self.u_hi = u_sgn;
            self.v.limbs[self.len] = self.v_hi;
            self.v_hi = v_sgn;
            self.len += 1;
        }
    }

    /// Materializes the round-0 identity state `(u, v) = (1, 0)` and `m` in one step: rather than
    /// actually multiplying `m` by that trivial vector, assigns `m`'s first column straight to
    /// `(u, v)` -- `r0.0*1 + r0.1*0 = r0.0` and `r1.0*1 + r1.1*0 = r1.0` -- since [`Self::new`]
    /// never materializes that starting `1` in `u`'s buffer at all (`len == 0` is the sentinel
    /// for it).
    ///
    /// Still runs the same growth-schedule accounting a normal round would via
    /// [`Self::apply_matrix`]'s own growth block, so [`Self::cap_remain`]/[`Self::k_deferred`]
    /// come out exactly as if `u` had started genuinely materialized at `len == 1` and this round
    /// had gone through [`Self::wrapping_apply_matrix`] like any other -- required for later
    /// rounds' own growth decisions to stay correct. [`SignedLimb`](super::matrix::SignedLimb)'s
    /// own invariant (never more than one limb's magnitude) guarantees this round's growth always
    /// fits the freshly-reset `cap_remain`, so the `grow_to` branch is unreachable in practice, but
    /// is kept for symmetry with [`Self::apply_matrix`] rather than assumed away.
    #[inline(always)]
    const fn set_from_matrix(&mut self, m: SignedLimbMatrix, k: u32) {
        if !self.is_full() {
            let growth = k + 1;
            if growth <= self.cap_remain {
                self.cap_remain -= growth;
            } else {
                self.grow_to(self.len + 1);
                self.cap_remain = self.cap_remain + Limb::BITS - growth;
            }
            if self.is_full() {
                // `self.k` is still `0` here (nothing has been applied yet), so this only ever
                // retires a no-op debt -- kept for symmetry with `apply_matrix`'s own block.
                self.k_deferred += self.k;
                self.k = 0;
            }
        }
        self.len = 1;
        (self.u.limbs[0], self.u_hi) = m.r0.0.signed_limb_pair();
        (self.v.limbs[0], self.v_hi) = m.r1.0.signed_limb_pair();
        self.k = k;
    }

    /// Applies one round's already column-sign-adjusted matrix `m` to the live `u`/`v` window and
    /// folds in that round's shift `k`. No growth, no reduction.
    #[inline(always)]
    const fn wrapping_apply_matrix(&mut self, m: SignedLimbMatrix, k: u32) {
        let (mut u, mut v) = (
            ExtendedIntRef::new(self.u.leading_mut(self.len), self.u_hi),
            ExtendedIntRef::new(self.v.leading_mut(self.len), self.v_hi),
        );
        m.wrapping_apply(&mut u, &mut v);
        (self.u_hi, self.v_hi) = (u.hi, v.hi);
        self.k += k;
    }

    /// [`Self::wrapping_apply_matrix`]'s `v`-only counterpart: updates `v` from `(u, v)` and
    /// leaves `u` holding whatever it held before.
    #[inline(always)]
    const fn wrapping_half_apply_matrix(&mut self, m: SignedLimbMatrix, k: u32) {
        let mut v = ExtendedIntRef::new(self.v.leading_mut(self.len), self.v_hi);
        m.wrapping_half_apply(self.u.leading(self.len), self.u_hi, &mut v);
        self.v_hi = v.hi;
        self.k += k;
    }

    /// [`Self::apply_matrix`]'s counterpart for the caller's *last* round, where `u`'s own new
    /// value is dead on arrival: [`Self::finalize`] reduces and returns `v` alone, so nothing ever
    /// reads `u` again. Pays off the pending shift and grows the window exactly as
    /// [`Self::apply_matrix`] does -- `u` is still an input to `v`'s row, so it has to be as
    /// reduced and as bounded here as it would be for a full apply -- then applies the bottom row
    /// only, halving the round's multiply-accumulate work.
    ///
    /// Leaves `u` stale. Correct only as the final call on this [`CofactorPair`]; a subsequent
    /// [`Self::apply_matrix`] would fold that stale `u` back into `v`.
    ///
    /// When this is also the *first* call (`len == 0`, `u`'s implicit `1` never materialized --
    /// see [`Self::set_from_matrix`]), goes through `set_from_matrix` just like
    /// [`Self::apply_matrix`] does: there is no `(u, v)` window yet for
    /// [`Self::wrapping_half_apply_matrix`] to read `u` from, so without this it would silently
    /// treat the implicit starting `u = 1` as `0` and produce a wrong `v`. Computing `u`'s row too
    /// costs nothing extra here -- with no existing window there's no multiply-accumulate loop to
    /// halve in the first place, just the same single `signed_limb_pair` call either way.
    #[inline(always)]
    pub const fn half_apply_matrix(&mut self, m: SignedLimbMatrix, k: u32) {
        if self.len == 0 {
            self.set_from_matrix(m, k);
            return;
        }
        self.reduce_k();
        if !self.is_full() {
            let growth = k + 1;
            if growth <= self.cap_remain {
                self.cap_remain -= growth;
            } else {
                self.grow_to(self.len + 1);
                self.cap_remain = self.cap_remain + Limb::BITS - growth;
            }
            if self.is_full() {
                // The window just filled. Everything owed up to here is retired to `k_deferred`
                // and never paid on `u` at all -- see this method's doc.
                self.k_deferred += self.k;
                self.k = 0;
            }
        }
        self.wrapping_half_apply_matrix(m, k);
    }

    /// Conditionally apply any pending modular division by `2^k` to both `u` and `v`. Called at
    /// the *start* of [`Self::apply_matrix`], reducing whatever was left pending by the previous
    /// round -- so the last round the caller's loop ever makes leaves its own shift untouched
    /// here; that final backlog is picked up by [`Self::finalize`] instead, which only needs to
    /// pay for `v`.
    ///
    /// No-op while the tracked window can still grow (`len < u.nlimbs()`): there's no need to pay
    /// for a mod-`y` reduction as long as overflow can simply be absorbed by widening the window
    /// instead.
    ///
    /// While there is a backlog to borrow against ([`Self::k_deferred`]), this divides by `k + 1`
    /// rather than `k` and skips [`ExtendedIntRef::try_reduce_mod`] entirely. The extra halving is
    /// what the correction was there for. Writing `M` for the bound on `|u|`, `|v|` just before a
    /// round's reduction, and using the row bound `||r||_1 <= 2^k`:
    ///
    /// ```text
    ///     by k, then try_reduce_mod:   M -> 2^k * (M/2^k     + y - y) = M              (stationary)
    ///     by k + 1, no correction:     M -> 2^k * (M/2^(k+1) + y)     = M/2 + 2^k * y  (contracting)
    /// ```
    ///
    /// The contraction's fixed point, `2^(k+1) * y`, is the same bound the stationary form sits
    /// at, so this needs no headroom the current schedule does not already have, and `M` is
    /// non-increasing rather than merely held in place. The borrowed bit is subtracted from
    /// `k_deferred` so the total owed is unchanged -- one per post-full round, against a backlog
    /// of thousands.
    #[inline(always)]
    const fn reduce_k(&mut self) {
        if self.is_full() && self.k != 0 {
            let (mut u, mut v) = (
                ExtendedIntRef::new(self.u, self.u_hi),
                ExtendedIntRef::new(self.v, self.v_hi),
            );
            let borrow = self.k_deferred != 0;
            let k = if borrow {
                self.k_deferred -= 1;
                self.k + 1
            } else {
                self.k
            };
            u.div2k_mod_assign_vartime(self.y, self.y_inv, k);
            v.div2k_mod_assign_vartime(self.y, self.y_inv, k);
            if !borrow {
                u.try_reduce_mod(self.y.as_nz_ref());
                v.try_reduce_mod(self.y.as_nz_ref());
            }
            self.u_hi = u.hi;
            self.v_hi = v.hi;
            self.k = 0;
        }
    }

    /// Vartime equivalent of [`Self::reduce_k`], including its borrow-one contraction: the
    /// bound argument there is on magnitudes and the row norm, neither of which depends on
    /// whether the schedule that produced the matrix was data-independent.
    #[inline(always)]
    const fn reduce_k_vartime(&mut self) {
        if self.is_full() && self.k != 0 {
            let (mut u, mut v) = (
                ExtendedIntRef::new(self.u, self.u_hi),
                ExtendedIntRef::new(self.v, self.v_hi),
            );
            if self.k_deferred != 0 {
                let k = self.k + 1;
                u.div2k_mod_assign_vartime(self.y, self.y_inv, k);
                self.u_hi = u.hi;
                v.div2k_mod_assign_vartime(self.y, self.y_inv, k);
                self.v_hi = v.hi;
                self.k_deferred -= 1;
            } else {
                u.div2k_mod_assign_vartime(self.y, self.y_inv, self.k);
                u.try_reduce_mod_vartime(self.y.as_nz_ref());
                self.u_hi = u.hi;
                v.div2k_mod_assign_vartime(self.y, self.y_inv, self.k);
                v.try_reduce_mod_vartime(self.y.as_nz_ref());
                self.v_hi = v.hi;
            }
            self.k = 0;
        }
    }

    /// How many bits `u`/`v` currently overflow their live window by, beyond a plain sign
    /// extension -- the vartime path's own growth trigger (see [`Self::apply_matrix_vartime`]).
    #[inline(always)]
    const fn overflow_vartime(&self) -> u32 {
        hi_overflow_vartime(self.u_hi) | hi_overflow_vartime(self.v_hi)
    }

    /// Applies one round's matrix `m` to `(u, v)`. First pays off any shift left pending by the
    /// previous round via [`Self::reduce_k`] (a no-op while the window still has growth room, or
    /// once nothing is pending), then grows the tracked window if the running bit budget says
    /// this round's own worst-case growth won't fit and there's still room to grow into, and
    /// finally applies the matrix -- leaving *this* round's own shift pending in turn, for the
    /// next call (or, if this was the last one, for [`Self::finalize`]) to deal with.
    ///
    /// The pending shift is split in two at the moment the window fills. Everything owed up to
    /// that point moves to [`Self::k_deferred`] and is never paid during the loop at all; only
    /// what accumulates afterwards is paid per round, which is all that keeping `u`/`v` bounded
    /// requires. The growth schedule caps them at roughly `2^bits(y)` when the window fills --
    /// the window is sized to hold exactly that -- so the backlog division would only change
    /// their representation, not their magnitude, and the per-round `k` alone holds the steady
    /// state (each round multiplies by at most `2^(k+1)` and divides by `2^k`, and
    /// `try_reduce_mod`'s single correction absorbs the remainder). Deferring it costs `u`'s
    /// copy of that division nothing, because [`Self::finalize`] pays the backlog on `v` alone.
    #[inline(always)]
    pub const fn apply_matrix(&mut self, m: SignedLimbMatrix, k: u32) {
        self.reduce_k();
        if self.len == 0 {
            self.set_from_matrix(m, k);
        } else {
            if !self.is_full() {
                let growth = k + 1;
                if growth <= self.cap_remain {
                    self.cap_remain -= growth;
                } else {
                    self.grow_to(self.len + 1);
                    self.cap_remain = self.cap_remain + Limb::BITS - growth;
                }
                if self.is_full() {
                    // The window just filled. Everything owed up to here is retired to `k_deferred`
                    // and never paid on `u` at all -- see this method's doc.
                    self.k_deferred += self.k;
                    self.k = 0;
                }
            }
            self.wrapping_apply_matrix(m, k);
        }
    }

    /// Vartime equivalent of [`Self::apply_matrix`]: pays off any shift pending from the previous
    /// round first (see [`Self::apply_matrix`]'s own doc for why), then applies the matrix and
    /// grows the window by one limb only if `(u, v)` actually overflowed it this round --
    /// measured directly from their own data via [`Self::overflow_vartime`], rather than guessed
    /// from a schedule.
    ///
    /// Retires the growth-phase backlog to [`Self::k_deferred`] on the round that fills the
    /// window, exactly as [`Self::apply_matrix`] does -- the transition is just detected from the
    /// overflow-driven growth rather than from the schedule, and has to be latched before the
    /// growth so that later rounds, which find the window already full, do not keep retiring
    /// their own `k` and never reduce at all.
    #[inline(always)]
    pub const fn apply_matrix_vartime(&mut self, m: SignedLimbMatrix, k: u32) {
        self.reduce_k_vartime();
        if self.len == 0 {
            self.set_from_matrix(m, k);
        } else {
            let growing = !self.is_full();
            self.wrapping_apply_matrix(m, k);
            let overflow = self.overflow_vartime();
            if overflow != 0 {
                self.grow_to(self.len + 1);
            }
            if growing && self.is_full() {
                self.k_deferred += self.k;
                self.k = 0;
            }
        }
    }

    /// [`Self::apply_matrix_vartime`]'s counterpart for the caller's last round, standing to it as
    /// [`Self::half_apply_matrix`] stands to [`Self::apply_matrix`]: `u`'s new value is never read,
    /// so only `v`'s row is computed. The window is not grown afterwards either -- growth exists to
    /// keep the *next* round's inputs in range, and there is no next round.
    ///
    /// Leaves `u` stale, so this must be the final call on this [`CofactorPair`].
    ///
    /// Same `len == 0` fast path as [`Self::half_apply_matrix`], and for the same reason: with no
    /// `(u, v)` window materialized yet, [`Self::wrapping_half_apply_matrix`] would read `u` as
    /// `0` instead of the implicit starting `1` and produce a wrong `v`.
    #[inline(always)]
    pub const fn half_apply_matrix_vartime(&mut self, m: SignedLimbMatrix, k: u32) {
        if self.len == 0 {
            self.set_from_matrix(m, k);
            return;
        }
        self.reduce_k_vartime();
        self.wrapping_half_apply_matrix(m, k);
    }

    /// Folds an extra pending shift `k` into the tracked total directly, without applying any
    /// matrix -- for a caller that strips some steps (e.g. common trailing zero bits) up front,
    /// outside the normal per-round matrix-apply loop.
    #[inline(always)]
    pub const fn defer_k(&mut self, k: u32) {
        self.k_deferred += k;
    }

    /// Grows to full width (flushing any limbs never touched) and reduces any pending `k` --
    /// on `v` alone. `u`'s matching reduction is skipped: by the time this runs the caller's
    /// loop is done, so [`Self::finalize`] (the only caller) never reads `u` again, and paying
    /// for its reduction too would be wasted work.
    #[inline(always)]
    const fn reduce_v(&mut self) {
        self.grow_to(self.u.nlimbs());
        // The growth-phase backlog rides along to here and is paid once, on `v` only.
        let k = self.k + self.k_deferred;
        if k != 0 {
            let mut v = ExtendedIntRef::new(self.v, self.v_hi);
            v.div2k_mod_assign_vartime(self.y, self.y_inv, k);
            v.try_reduce_mod(self.y.as_nz_ref());
            self.v_hi = v.hi;
        }
    }

    /// Vartime equivalent of [`Self::reduce_v`].
    #[inline(always)]
    const fn reduce_v_vartime(&mut self) {
        self.grow_to(self.u.nlimbs());
        // The growth-phase backlog rides along to here and is paid once, on `v` only.
        let k = self.k + self.k_deferred;
        if k != 0 {
            let mut v = ExtendedIntRef::new(self.v, self.v_hi);
            v.div2k_mod_assign_vartime(self.y, self.y_inv, k);
            v.try_reduce_mod_vartime(self.y.as_nz_ref());
            self.v_hi = v.hi;
        }
    }

    /// Conditionally negates `u` in place, correcting it to match a sign flip just applied to
    /// whatever value `u` tracks the same linear combination of.
    #[inline(always)]
    pub const fn negate_u_if(&mut self, cond: Choice) {
        let mut u = ExtendedIntRef::new(self.u, self.u_hi);
        u.conditional_carrying_neg_assign(cond);
        self.u_hi = u.hi;
    }

    /// [`Self::negate_u_if`]'s counterpart for `v`.
    #[inline(always)]
    pub const fn negate_v_if(&mut self, cond: Choice) {
        let mut v = ExtendedIntRef::new(self.v, self.v_hi);
        v.conditional_carrying_neg_assign(cond);
        self.v_hi = v.hi;
    }

    /// Finishes tracking: grows to full width and reduces any pending `k` on `v` alone (see
    /// [`Self::reduce_v`] -- `u`'s own final value is never read, so its matching reduction is
    /// skipped), reduces `v` into `[0, y)`, and drops the `hi` extension.
    ///
    /// Returns the reduced non-negative `v`.
    pub const fn finalize(mut self) -> &'a mut UintRef {
        self.reduce_v();
        let mut v = ExtendedIntRef::new(self.v, self.v_hi);
        v.try_reduce_mod(self.y.as_nz_ref());
        v.conditional_wrapping_add_assign_unsigned(self.y.as_ref(), v.is_negative());
        v.unsigned_drop_extension()
    }

    /// Vartime equivalent of [`Self::finalize`].
    pub const fn finalize_vartime(mut self) -> &'a mut UintRef {
        self.reduce_v_vartime();
        let mut v = ExtendedIntRef::new(self.v, self.v_hi);
        v.try_reduce_mod_vartime(self.y.as_nz_ref());
        v.unsigned_drop_extension()
    }
}
