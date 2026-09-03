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
    /// Seeds `u` at `1` or, when `monty_form_r2` is given, at that Montgomery `R^2` instead.
    /// Seeds `v` at `0`. This is the coefficient paired with `f`, and the one [`Self::finalize`]
    /// actually returns.
    ///
    /// The starting window width starts at 1 when `monty_form_r2` is not provided,
    /// and at `u.nlimbs()` when it is – in the second case the coefficients have no
    /// room to grow and reductions are applied after each batch update.
    #[allow(clippy::cast_possible_truncation)]
    pub const fn new(
        u: &'a mut UintRef,
        v: &'a mut UintRef,
        y: &'a Odd<UintRef>,
        y_inv: Limb,
        monty_form_r2: Option<&UintRef>,
    ) -> Self {
        assert!(u.nlimbs() == v.nlimbs());
        v.fill(Limb::ZERO);
        let len = if let Some(r2) = monty_form_r2 {
            u.copy_from(r2);
            u.nlimbs()
        } else {
            u.fill(Limb::ZERO);
            u.limbs[0] = Limb::ONE;
            1
        };
        Self::new_with_len(u, v, y, y_inv, len)
    }

    /// Construct a new instance with initialized buffers and a tracked window length.
    const fn new_with_len(
        u: &'a mut UintRef,
        v: &'a mut UintRef,
        y: &'a Odd<UintRef>,
        y_inv: Limb,
        len: usize,
    ) -> Self {
        Self {
            u,
            u_hi: Limb::ZERO,
            v,
            v_hi: Limb::ZERO,
            len,
            k: 0,
            cap_remain: Limb::BITS - 1,
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
        while self.len < target {
            self.u.limbs[self.len] = self.u_hi;
            self.u_hi = self.u_hi.shr(Limb::HI_BIT).wrapping_neg();
            self.v.limbs[self.len] = self.v_hi;
            self.v_hi = self.v_hi.shr(Limb::HI_BIT).wrapping_neg();
            self.len += 1;
        }
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

    /// Conditionally apply any pending modular division by `2^k` to both `u` and `v`. Called at
    /// the *start* of [`Self::apply_matrix`], reducing whatever was left pending by the previous
    /// round -- so the last round the caller's loop ever makes leaves its own shift untouched
    /// here; that final backlog is picked up by [`Self::finalize`] instead, which only needs to
    /// pay for `v`.
    ///
    /// No-op while the tracked window can still grow (`len < u.nlimbs()`): there's no need to pay
    /// for a mod-`y` reduction as long as overflow can simply be absorbed by widening the window
    /// instead.
    #[inline(always)]
    const fn reduce_k(&mut self) {
        if self.is_full() && self.k != 0 {
            let (mut u, mut v) = (
                ExtendedIntRef::new(self.u, self.u_hi),
                ExtendedIntRef::new(self.v, self.v_hi),
            );
            u.div2k_mod_assign_vartime(self.y, self.y_inv, self.k);
            u.try_reduce_mod(self.y.as_nz_ref());
            self.u_hi = u.hi;
            v.div2k_mod_assign_vartime(self.y, self.y_inv, self.k);
            v.try_reduce_mod(self.y.as_nz_ref());
            self.v_hi = v.hi;
            self.k = 0;
        }
    }

    /// Vartime equivalent of [`Self::reduce_k`].
    #[inline(always)]
    const fn reduce_k_vartime(&mut self) {
        if self.is_full() && self.k != 0 {
            let (mut u, mut v) = (
                ExtendedIntRef::new(self.u, self.u_hi),
                ExtendedIntRef::new(self.v, self.v_hi),
            );
            u.div2k_mod_assign_vartime(self.y, self.y_inv, self.k);
            u.try_reduce_mod_vartime(self.y.as_nz_ref());
            self.u_hi = u.hi;
            v.div2k_mod_assign_vartime(self.y, self.y_inv, self.k);
            v.try_reduce_mod_vartime(self.y.as_nz_ref());
            self.v_hi = v.hi;
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
    #[inline(always)]
    pub const fn apply_matrix(&mut self, m: SignedLimbMatrix, k: u32) {
        self.reduce_k();
        if !self.is_full() {
            let growth = k + 1;
            if growth <= self.cap_remain {
                self.cap_remain -= growth;
            } else {
                self.grow_to(self.len + 1);
                self.cap_remain = self.cap_remain + Limb::BITS - growth;
            }
        }
        self.wrapping_apply_matrix(m, k);
    }

    /// Vartime equivalent of [`Self::apply_matrix`]: pays off any shift pending from the previous
    /// round first (see [`Self::apply_matrix`]'s own doc for why), then applies the matrix and
    /// grows the window by one limb only if `(u, v)` actually overflowed it this round --
    /// measured directly from their own data via [`Self::overflow_vartime`], rather than guessed
    /// from a schedule.
    #[inline(always)]
    pub const fn apply_matrix_vartime(&mut self, m: SignedLimbMatrix, k: u32) {
        self.reduce_k_vartime();
        self.wrapping_apply_matrix(m, k);
        let overflow = self.overflow_vartime();
        if overflow != 0 {
            self.grow_to(self.len + 1);
        }
    }

    /// Folds an extra pending shift `k` into the tracked total directly, without applying any
    /// matrix -- for a caller that strips some steps (e.g. common trailing zero bits) up front,
    /// outside the normal per-round matrix-apply loop.
    #[inline(always)]
    pub const fn defer_k(&mut self, k: u32) {
        self.k += k;
    }

    /// Grows to full width (flushing any limbs never touched) and reduces any pending `k` --
    /// on `v` alone. `u`'s matching reduction is skipped: by the time this runs the caller's
    /// loop is done, so [`Self::finalize`] (the only caller) never reads `u` again, and paying
    /// for its reduction too would be wasted work.
    #[inline(always)]
    const fn reduce_v(&mut self) {
        self.grow_to(self.u.nlimbs());
        if self.k != 0 {
            let mut v = ExtendedIntRef::new(self.v, self.v_hi);
            v.div2k_mod_assign_vartime(self.y, self.y_inv, self.k);
            v.try_reduce_mod(self.y.as_nz_ref());
            self.v_hi = v.hi;
        }
    }

    /// Vartime equivalent of [`Self::reduce_v`].
    #[inline(always)]
    const fn reduce_v_vartime(&mut self) {
        self.grow_to(self.u.nlimbs());
        if self.k != 0 {
            let mut v = ExtendedIntRef::new(self.v, self.v_hi);
            v.div2k_mod_assign_vartime(self.y, self.y_inv, self.k);
            v.try_reduce_mod_vartime(self.y.as_nz_ref());
            self.v_hi = v.hi;
        }
    }

    /// Conditionally negates `u` in place, correcting it to match a sign flip just applied to
    /// whatever value `u` tracks the same linear combination of.
    #[inline(always)]
    pub const fn negate_u_if(&mut self, cond: Choice) {
        debug_assert!(self.is_full());
        let mut u = ExtendedIntRef::new(self.u, self.u_hi);
        u.conditional_carrying_neg_assign(cond);
        self.u_hi = u.hi;
    }

    /// [`Self::negate_u_if`]'s counterpart for `v`.
    #[inline(always)]
    pub const fn negate_v_if(&mut self, cond: Choice) {
        debug_assert!(self.is_full());
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
