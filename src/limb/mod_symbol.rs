use super::Limb;
use crate::{JacobiSymbol, Odd, word};

impl Limb {
    /// Compute the Jacobi symbol `(self|rhs)`.
    ///
    /// For prime `rhs`, this corresponds to the Legendre symbol and
    /// indicates whether `self` is quadratic residue modulo `rhs`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn jacobi_symbol(self, rhs: Odd<Self>) -> JacobiSymbol {
        let (gcd, jacobi_neg) = rhs.bingcd(self);
        // The sign of the Jacobi symbol is represented by jacobi_neg. We select 0 as the
        // symbol when the GCD is not one, otherwise 1 or -1.
        let jacobi = (jacobi_neg as i8 * -2 + 1) as i64;
        JacobiSymbol::from_i8(word::choice_from_eq(gcd.get_copy().0, 1).select_i64(0, jacobi) as i8)
    }

    /// Compute the Jacobi symbol `(self|rhs)`.
    ///
    /// For prime `rhs`, this corresponds to the Legendre symbol and
    /// indicates whether `self` is quadratic residue modulo `rhs`.
    ///
    /// This method executes in variable-time for both inputs.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn jacobi_symbol_vartime(self, rhs: Odd<Self>) -> JacobiSymbol {
        let (gcd, jacobi_neg) = rhs.bingcd_vartime(self);
        JacobiSymbol::from_i8(if gcd.as_ref().0 == 1 {
            jacobi_neg as i8 * -2 + 1
        } else {
            0
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{JacobiSymbol, Limb};

    #[test]
    fn jacobi_quad_residue() {
        // Two semiprimes with no common factors, and
        // f is quadratic residue modulo g
        let f = Limb::from(59u32 * 67);
        let g = Limb::from(61u32 * 71).to_odd().unwrap();
        let res = f.jacobi_symbol(g);
        let res_vartime = f.jacobi_symbol_vartime(g);
        assert_eq!(res, JacobiSymbol::One);
        assert_eq!(res, res_vartime);
    }

    #[test]
    fn jacobi_non_quad_residue() {
        // f and g have no common factors, but
        // f is not quadratic residue modulo g
        let f = Limb::from(59u32 * 67 + 2);
        let g = Limb::from(61u32 * 71).to_odd().unwrap();
        let res = f.jacobi_symbol(g);
        let res_vartime = f.jacobi_symbol_vartime(g);
        assert_eq!(res, JacobiSymbol::MinusOne);
        assert_eq!(res, res_vartime);
    }

    #[test]
    fn jacobi_non_coprime() {
        let f = Limb::from(4391633u32);
        let g = Limb::from(2022161u32).to_odd().unwrap();
        let res = f.jacobi_symbol(g);
        let res_vartime = f.jacobi_symbol_vartime(g);
        assert_eq!(res, JacobiSymbol::Zero);
        assert_eq!(res, res_vartime);
    }

    #[test]
    fn jacobi_zero() {
        let f = Limb::ZERO;
        let g = Limb::ONE.to_odd().unwrap();
        let res = f.jacobi_symbol(g);
        let res_vartime = f.jacobi_symbol_vartime(g);
        assert_eq!(res, JacobiSymbol::One);
        assert_eq!(res, res_vartime);
    }

    #[test]
    fn jacobi_one() {
        let f = Limb::ONE;
        let g = Limb::ONE.to_odd().unwrap();
        let res = f.jacobi_symbol(g);
        let res_vartime = f.jacobi_symbol_vartime(g);
        assert_eq!(res, JacobiSymbol::One);
        assert_eq!(res, res_vartime);
    }
}
