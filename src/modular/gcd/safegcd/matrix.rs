use crate::modular::gcd::{ExtendedIntRef, SignedLimb, SignedLimbMatrix};
use crate::{Limb, Odd, UintRef};

#[derive(Debug, Copy, Clone)]
pub struct SafegcdMatrix {
    pub m0: (Limb, Limb),
    pub m1: (Limb, Limb),
    pub k: u32,
}

impl SafegcdMatrix {
    pub const UNIT: Self = Self {
        m0: (Limb::ONE, Limb::ZERO),
        m1: (Limb::ZERO, Limb::ONE),
        k: 0,
    };

    #[inline(always)]
    const fn signed_limb_matrix(&self) -> SignedLimbMatrix {
        SignedLimbMatrix {
            m0: (
                SignedLimb::from_limb(self.m0.0),
                SignedLimb::from_limb(self.m0.1),
            ),
            m1: (
                SignedLimb::from_limb(self.m1.0),
                SignedLimb::from_limb(self.m1.1),
            ),
        }
    }

    #[inline(always)]
    pub const fn wrapping_apply_shift(
        &self,
        f: &mut ExtendedIntRef<'_>,
        g: &mut ExtendedIntRef<'_>,
    ) {
        self.signed_limb_matrix().wrapping_apply(f, g);
        f.shr_assign_limb(self.k);
        g.shr_assign_limb(self.k);
    }

    #[inline(always)]
    pub const fn wrapping_half_apply_shift(
        &self,
        f: &mut ExtendedIntRef<'_>,
        g: &ExtendedIntRef<'_>,
    ) {
        self.signed_limb_matrix().wrapping_half_apply(f, g);
        f.shr_assign_limb(self.k);
    }

    #[inline(always)]
    pub const fn wrapping_apply_div2k_mod(
        &self,
        d: &mut ExtendedIntRef<'_>,
        e: &mut ExtendedIntRef<'_>,
        m: &Odd<UintRef>,
        m_inv: Limb,
    ) {
        self.signed_limb_matrix().wrapping_apply(d, e);

        d.limb_div2k_mod_assign(m, m_inv, self.k);
        d.try_reduce_mod(m.as_nz_ref());

        e.limb_div2k_mod_assign(m, m_inv, self.k);
        e.try_reduce_mod(m.as_nz_ref());
    }

    #[inline(always)]
    pub const fn wrapping_half_apply_div2k_mod(
        &self,
        d: &mut ExtendedIntRef<'_>,
        e: &ExtendedIntRef<'_>,
        m: &Odd<UintRef>,
        m_inv: Limb,
    ) {
        self.signed_limb_matrix().wrapping_half_apply(d, e);

        d.limb_div2k_mod_assign(m, m_inv, self.k);
        d.try_reduce_mod(m.as_nz_ref());
    }
}

impl PartialEq for SafegcdMatrix {
    fn eq(&self, other: &Self) -> bool {
        self.signed_limb_matrix().eq(&other.signed_limb_matrix())
    }
}
