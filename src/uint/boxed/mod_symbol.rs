use super::BoxedUint;
use crate::{Odd, Resize, jacobi::JacobiSymbol};

impl BoxedUint {
    /// Compute the Jacobi symbol `(self|rhs)`.
    ///
    /// For prime `rhs`, this corresponds to the Legendre symbol and
    /// indicates whether `self` is quadratic residue modulo `rhs`.
    ///
    /// This method executes in variable-time for both inputs.
    #[must_use]
    pub fn jacobi_symbol(&self, rhs: &Odd<BoxedUint>) -> JacobiSymbol {
        let rhs_bits = rhs.as_ref().bits_precision();
        let mut lhs = if self.bits_precision() > rhs_bits {
            self.rem(rhs.as_nz_ref())
        } else {
            self.resize(rhs_bits)
        };
        crate::modular::gcd::jacobi_symbol(
            lhs.as_mut_uint_ref(),
            rhs.as_ref().clone().as_mut_uint_ref(),
        )
    }

    /// Compute the Jacobi symbol `(self|rhs)`.
    ///
    /// For prime `rhs`, this corresponds to the Legendre symbol and
    /// indicates whether `self` is quadratic residue modulo `rhs`.
    ///
    /// This method executes in variable-time for both inputs.
    #[must_use]
    pub fn jacobi_symbol_vartime(&self, rhs: &Odd<BoxedUint>) -> JacobiSymbol {
        let max_bits = self.bits_precision().max(rhs.as_ref().bits_precision());
        crate::modular::gcd::jacobi_symbol_vartime(
            self.resize(max_bits).as_mut_uint_ref(),
            rhs.as_ref().resize(max_bits).as_mut_uint_ref(),
        )
    }
}
