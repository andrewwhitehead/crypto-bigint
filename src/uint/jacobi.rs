//! Support for computing the greatest common divisor of two `Uint`s.

use crate::{
    Int, JacobiSymbol, JacobiVartime, Odd, PrecomputeInverter, Uint, modular::SafeGcdInverter,
};

impl<const SAT_LIMBS: usize, const UNSAT_LIMBS: usize> Uint<SAT_LIMBS>
where
    Odd<Self>: PrecomputeInverter<Inverter = SafeGcdInverter<SAT_LIMBS, UNSAT_LIMBS>>,
{
    /// Compute the Jacobi symbol of this number over an odd modulus.
    ///
    /// Runs in a constant number of iterations depending on the maximum highest bit of either
    /// `self` or `modulus`.
    pub fn jacobi_vartime(&self, modulus: &Odd<Self>) -> JacobiSymbol {
        <Odd<Self> as PrecomputeInverter>::Inverter::jacobi_uint_vartime(self, modulus)
    }
}

impl<const SAT_LIMBS: usize, const UNSAT_LIMBS: usize> Int<SAT_LIMBS>
where
    Odd<Uint<SAT_LIMBS>>: PrecomputeInverter<Inverter = SafeGcdInverter<SAT_LIMBS, UNSAT_LIMBS>>,
{
    /// Compute the Jacobi symbol of this number over an odd modulus.
    ///
    /// Runs in a constant number of iterations depending on the maximum highest bit of either
    /// `self` or `modulus`.
    pub fn jacobi_vartime(&self, modulus: &Odd<Self>) -> JacobiSymbol {
        <Odd<Uint<SAT_LIMBS>> as PrecomputeInverter>::Inverter::jacobi_int_vartime(self, modulus)
    }
}

impl<const SAT_LIMBS: usize, const UNSAT_LIMBS: usize> JacobiVartime for Uint<SAT_LIMBS>
where
    Odd<Self>: PrecomputeInverter<Inverter = SafeGcdInverter<SAT_LIMBS, UNSAT_LIMBS>>,
{
    #[inline]
    fn jacobi_vartime(&self, m: &Odd<Self>) -> JacobiSymbol {
        Self::jacobi_vartime(self, m)
    }
}

impl<const SAT_LIMBS: usize, const UNSAT_LIMBS: usize> JacobiVartime for Int<SAT_LIMBS>
where
    Odd<Uint<SAT_LIMBS>>: PrecomputeInverter<Inverter = SafeGcdInverter<SAT_LIMBS, UNSAT_LIMBS>>,
{
    #[inline]
    fn jacobi_vartime(&self, m: &Odd<Self>) -> JacobiSymbol {
        Self::jacobi_vartime(self, m)
    }
}

#[cfg(test)]
mod tests {
    use crate::{I256, JacobiSymbol, U256};

    #[test]
    fn jacobi_quad_residue() {
        // Two semiprimes with no common factors
        // f is quadratic residue modulo g
        let f = U256::from(59u32 * 67);
        let g = U256::from(61u32 * 71).to_odd().unwrap();
        let res = f.jacobi_vartime(&g);
        assert_eq!(res, JacobiSymbol::One);
    }

    #[test]
    fn jacobi_non_quad_residue() {
        // f and g have no common factors, but
        // f is not quadratic residue modulo g
        let f = U256::from(59u32 * 67 + 2);
        let g = U256::from(61u32 * 71).to_odd().unwrap();
        let res = f.jacobi_vartime(&g);
        assert_eq!(res, JacobiSymbol::MinusOne);
    }

    #[test]
    fn jacobi_non_coprime() {
        let f = U256::from(4391633u32);
        let g = U256::from(2022161u32).to_odd().unwrap();
        let res = f.jacobi_vartime(&g);
        assert_eq!(res, JacobiSymbol::Zero);
    }

    #[test]
    fn jacobi_zero() {
        assert_eq!(
            U256::ZERO.jacobi_vartime(&U256::ONE.to_odd().unwrap()),
            JacobiSymbol::One
        );
    }

    #[test]
    fn jacobi_one() {
        let f = U256::ONE;
        assert_eq!(
            f.jacobi_vartime(&U256::ONE.to_odd().unwrap()),
            JacobiSymbol::One
        );
        assert_eq!(
            f.jacobi_vartime(&U256::from(3u8).to_odd().unwrap()),
            JacobiSymbol::One
        );
    }

    #[test]
    fn jacobi_signed() {
        // Two semiprimes with no common factors
        // f is quadratic residue modulo g
        let f = I256::from(59i32 * 67);
        let g = I256::from(61i32 * 71).to_odd().unwrap();
        let res = f.jacobi_vartime(&g);
        assert_eq!(res, JacobiSymbol::One);

        let res = f.jacobi_vartime(&g.as_ref().wrapping_neg().to_odd().unwrap());
        assert_eq!(res, JacobiSymbol::One);

        let res = f.wrapping_neg().jacobi_vartime(&g);
        assert_eq!(res, JacobiSymbol::MinusOne);
    }
}
