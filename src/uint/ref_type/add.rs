use super::UintRef;
use crate::Limb;

impl UintRef {
    /// Perform an in-place carrying add of a limb, returning the carried limb value.
    #[inline]
    #[track_caller]
    pub const fn add_assign_limb(&mut self, mut rhs: Limb) -> Limb {
        let mut i = 0;
        while i < self.0.len() {
            (self.0[i], rhs) = self.0[i].overflowing_add(rhs);
            i += 1;
        }
        rhs
    }

    /// Perform an in-place carrying add of another [`UintRef`], returning the carried limb value.
    #[inline]
    #[track_caller]
    pub const fn carrying_add_assign(&mut self, rhs: &Self, carry: Limb) -> Limb {
        self.carrying_add_assign_slice(&rhs.0, carry)
    }

    /// Perform an in-place carrying add of another limb slice, returning the carried limb value.
    #[inline]
    #[track_caller]
    pub const fn carrying_add_assign_slice(&mut self, rhs: &[Limb], mut carry: Limb) -> Limb {
        assert!(
            self.0.len() == rhs.len(),
            "length mismatch in carrying_add_assign_slice"
        );
        let mut i = 0;
        while i < self.0.len() {
            (self.0[i], carry) = self.0[i].carrying_add(rhs[i], carry);
            i += 1;
        }
        carry
    }
}
