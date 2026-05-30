use core::mem;

use ctutils::Choice;

use super::compact::compact_pair_vartime;
use super::matrix::BingcdMatrix;
use crate::modular::gcd::ExtendedIntRef;
use crate::{JacobiSymbol, Limb, Odd, UintRef, WideWord, Word, bitlen, uint::gcd::UintXgcdOutput};

#[derive(Debug)]
struct GcdVartimeBuffer<'a> {
    uint: &'a mut UintRef,
    bits: u32,
    index: bool,
}

impl<'a> GcdVartimeBuffer<'a> {
    pub const fn new(uint: &'a mut UintRef, bits: u32, index: bool) -> Self {
        Self { uint, bits, index }
    }

    pub const fn nlimbs(&self) -> usize {
        self.uint.nlimbs()
    }

    pub const fn assign_limb(&mut self, value: Limb) {
        self.uint.set_from_limb(value);
        self.bits = value.bits();
    }

    pub const fn assign_wide_word(&mut self, value: WideWord) {
        self.uint.set_from_wide_word(value);
        self.bits = self.uint.bits_vartime();
    }

    pub const fn low_limb(&self) -> Limb {
        self.uint.limbs[0]
    }

    pub const fn reduce_rem_limb(&mut self, other: Odd<Limb>) {
        let rem = self.uint.rem_limb(*other.as_nz_ref());
        self.assign_limb(rem);
    }

    #[inline(always)]
    pub const fn reduce_dmod(
        &mut self,
        other: &Odd<UintRef>,
        other_inv: Limb,
        other_bits: u32,
    ) -> Limb {
        debug_assert!(self.uint.nlimbs() >= other.as_ref().nlimbs());

        let other = other.as_ref();
        let bit_diff = self.bits.checked_sub(other_bits).expect("invalid usage");

        let shift = if bit_diff <= Limb::BITS {
            bit_diff - 1
        } else {
            Limb::BITS
        };
        let q = other_inv
            .wrapping_mul(self.uint.limbs[0])
            .restrict_bits(shift);

        let (_hi, neg) = sub_mul_assign(self.uint, other, q);
        assert!(!neg, "underflow"); // tried to clear too many bits

        self.bits = self.uint.bits_vartime();
        q
    }

    #[inline(always)]
    pub const fn reduce_bingcd(&mut self, other: &mut Self) -> (BingcdMatrix, Word) {
        let (a, b) = (self, other);
        let cbits = if a.bits < WideWord::BITS {
            WideWord::BITS
        } else {
            a.bits
        };
        let (a_, b_) = compact_pair_vartime(a.uint, b.uint, cbits);
        let (matrix, (a_, b_), jac_neg) = super::partial_xgcd_vartime(a_, b_);

        if a.bits <= WideWord::BITS {
            a.assign_wide_word(a_);
            b.assign_wide_word(b_);
        } else {
            matrix.apply_unsigned_vartime(a.uint, b.uint);
            (a.bits, b.bits) = (a.uint.bits_vartime(), b.uint.bits_vartime());
        };

        debug_assert!(b.uint.is_odd().to_bool_vartime());
        (matrix, jac_neg)
    }

    #[inline]
    const fn strip_zeros(&mut self) -> u32 {
        if self.bits == 0 {
            0
        } else {
            let tz = self.uint.trailing_zeros_vartime();
            self.uint.unbounded_shr_assign_vartime(tz);
            self.bits -= tz;
            tz
        }
    }

    #[inline]
    #[track_caller]
    const fn truncate(&mut self, len: usize) {
        if len < self.nlimbs() {
            let mut buf = UintRef::new_mut(&mut []);
            mem::swap(&mut buf, &mut self.uint);
            buf = buf.leading_mut(len);
            mem::swap(&mut buf, &mut self.uint);
        }
    }
}

const fn sub_mul_assign(a: &mut UintRef, b: &UintRef, q: Limb) -> (Limb, bool) {
    debug_assert!(a.limbs.len() >= b.limbs.len());

    let mut carry = Limb::ZERO;
    let mut borrow = Limb::ZERO;
    let mut sub;
    let mut i = 0;

    while i < b.limbs.len() {
        (sub, carry) = q.carrying_mul_add(b.limbs[i], Limb::ZERO, carry);
        (a.limbs[i], borrow) = a.limbs[i].borrowing_sub(sub, borrow);
        i += 1;
    }
    while i < a.limbs.len() {
        (a.limbs[i], borrow) = a.limbs[i].borrowing_sub(carry, borrow);
        carry = Limb::ZERO;
        i += 1;
    }
    (carry, borrow) = carry.borrowing_sub(Limb::ZERO, borrow);

    let negated = !borrow.is_zero_vartime();
    if negated {
        let c = a.wrapping_neg_assign();
        carry = carry.not().wrapping_add(c);
    }

    (carry, negated)
}

pub const fn gcd_vartime(a: &mut UintRef, b: &mut UintRef) -> bool {
    let mut a_bits = a.bits_vartime();
    let mut b_bits = b.bits_vartime();

    if a_bits == 0 || b_bits == 0 {
        // gcd b if a is zero, a if b is zero
        return a_bits == 0;
    }

    // ensure a is odd
    let az = a.trailing_zeros_vartime();
    if az != 0 {
        a.unbounded_shr_assign_vartime(az);
        a_bits -= az;
    }
    // ensure b is odd
    let bz = b.trailing_zeros_vartime();
    if bz != 0 {
        b.unbounded_shr_assign_vartime(bz);
        b_bits -= bz;
    }
    // GCD shift is the minimum of the trailing zeros
    let k = if az < bz { az } else { bz };

    let (index, _) = gcd_odd_vartime(
        GcdVartimeBuffer::new(a, a_bits, false),
        GcdVartimeBuffer::new(b, b_bits, true),
    );

    // Apply shift
    if k != 0 {
        (if index { b } else { a }).shl_assign(k);
    }

    index
}

fn xgcd_vartime(a: &mut UintRef, b: &mut UintRef) -> bool {
    let mut a_bits = a.bits_vartime();
    let mut b_bits = b.bits_vartime();

    if a_bits == 0 || b_bits == 0 {
        // gcd b if a is zero, a if b is zero
        return a_bits == 0;
    }

    // println!("a: {a} b: {b}");

    // ensure a is odd
    let az = a.trailing_zeros_vartime();
    if az != 0 {
        a.unbounded_shr_assign_vartime(az);
        a_bits -= az;
    }
    // ensure b is odd
    let bz = b.trailing_zeros_vartime();
    if bz != 0 {
        b.unbounded_shr_assign_vartime(bz);
        b_bits -= bz;
    }
    // GCD shift is the minimum of the trailing zeros
    let k = if az < bz { az } else { bz };

    let (index, _) = gcd_odd_vartime(
        GcdVartimeBuffer::new(a, a_bits, false),
        GcdVartimeBuffer::new(b, b_bits, true),
    );

    // Apply shift
    if k != 0 {
        (if index { b } else { a }).shl_assign(k);
    }

    index
}

pub const fn jacobi_symbol_vartime(a: &mut UintRef, b: &mut UintRef) -> JacobiSymbol {
    assert!(b.is_odd().to_bool_vartime(), "denominator must be odd");

    let mut a_bits = a.bits_vartime();
    let b_bits = b.bits_vartime();
    if a_bits == 0 {
        return if b.is_one().to_bool_vartime() {
            JacobiSymbol::One
        } else {
            JacobiSymbol::Zero
        };
    }

    // Ensure a is odd
    let az = a.trailing_zeros_vartime();
    if az != 0 {
        a.unbounded_shr_assign_vartime(az);
        a_bits -= az;
    }
    let jflip = if az & 1 == 1 {
        let b_lo = b.limbs[0].0;
        ((b_lo >> 1) ^ (b_lo >> 2)) & 1
    } else {
        0
    };

    let (index, mut jacobi_neg) = gcd_odd_vartime(
        GcdVartimeBuffer::new(a, a_bits, false),
        GcdVartimeBuffer::new(b, b_bits, true),
    );
    jacobi_neg ^= jflip;
    let check_gcd = (if index { b } else { a }).is_one().to_bool_vartime();
    if check_gcd {
        JacobiSymbol::from_sign(jacobi_neg)
    } else {
        JacobiSymbol::Zero
    }
}

#[inline(always)]
const fn gcd_odd_vartime<'a>(
    mut a: GcdVartimeBuffer<'a>,
    mut b: GcdVartimeBuffer<'a>,
) -> (bool, Word) {
    let mut jacobi_neg = 0;

    while a.bits != 0 {
        truncate(&mut a, &mut b);
        if maybe_swap(&mut a, &mut b) {
            let a_b = a.low_limb().0 & b.low_limb().0;
            jacobi_neg ^= (a_b & (a_b >> 1)) & 1;
        }

        if a.bits <= WideWord::BITS {
            // perform vartime bingcd
            let (gcd, j) = super::bingcd_wideword_vartime(
                a.uint.to_wide_word_unchecked(),
                b.uint.to_wide_word_unchecked(),
            );
            a.assign_limb(Limb::ZERO);
            b.assign_wide_word(gcd);
            jacobi_neg ^= j;
            continue;
        }

        if b.bits <= Limb::BITS && a.bits > Limb::BITS {
            // optimize for small b, maybe only do once at the beginning
            let b_odd = b.low_limb().to_odd().expect_copied("expected odd limb");
            a.reduce_rem_limb(b_odd);
            // FIXME proceed straight to word GCD?
        } else
        // if limb_diff >= 5 {
        //     // if a much bigger than b: perform div_rem_vartime
        // }
        if a.bits - b.bits >= Limb::BITS {
            // Reduce by one limb by subtracting q•b from a
            // this does not change the Jacobi symbol
            let b_odd = b.uint.as_odd_vartime().expect("expected odd denominator");
            let b_inv = b_odd.invert_mod_limb();
            let _ = a.reduce_dmod(b_odd, b_inv, b.bits);
        } else {
            // perform an optimized bingcd reduction
            let (_, j) = a.reduce_bingcd(&mut b);
            jacobi_neg ^= j;
        }

        let tz = a.strip_zeros();
        if tz & 1 == 1 {
            let b_lo = b.low_limb().0;
            jacobi_neg ^= ((b_lo >> 1) ^ (b_lo >> 2)) & 1;
        }
    }

    (b.index, jacobi_neg)
}

#[inline(always)]
pub const fn invert_vartime<'a>(
    x: &'a mut UintRef,
    f: &'a Odd<UintRef>,
    f_inv: Limb,
    buf: &'a mut UintRef,
) -> bool {
    let a_bits = x.bits_vartime();
    if a_bits == 0 {
        // no reciprocal for zero
        return false;
    }

    let f_len = f.as_ref().nlimbs();
    let (b, buf) = buf.split_at_mut(f_len);
    b.copy_from(f.as_ref());
    let b_bits = b.bits_vartime();

    let (u, buf) = buf.split_at_mut(f_len);
    u.fill(Limb::ZERO);
    u.limbs[0] = Limb::ONE;
    let v = buf.leading_mut(f_len);
    v.fill(Limb::ZERO);

    let (mut a, mut b, mut u, mut v) = (
        GcdVartimeBuffer::new(&mut *x, a_bits, false),
        GcdVartimeBuffer::new(b, b_bits, true),
        ExtendedIntRef::new(u, Limb::ZERO),
        ExtendedIntRef::new(v, Limb::ZERO),
    );

    let k = partial_xgcd_vartime(&mut a, &mut b, &mut u, &mut v);

    if !b.uint.is_one().to_bool_vartime() {
        return false;
    }

    v.div2k_mod_assign_vartime(f, f_inv, k);
    v.try_reduce_mod(f.as_nz_ref());
    if v.is_negative_vartime() {
        v.wrapping_add_assign_signed(f.as_ref(), Choice::FALSE);
        debug_assert!(!v.is_negative_vartime());
    }
    let v = v.unsigned_drop_extension();
    x.copy_from(v);

    true
}

#[inline(always)]
const fn partial_xgcd_vartime<'r, 's>(
    a: &mut GcdVartimeBuffer<'r>,
    b: &mut GcdVartimeBuffer<'r>,
    u: &mut ExtendedIntRef<'s>,
    v: &mut ExtendedIntRef<'s>,
) -> u32 {
    let mut k = a.strip_zeros();

    loop {
        truncate(a, b);
        if maybe_swap(a, b) {
            // println!("swap");
            mem::swap(u, v);
        }

        // FIXME best way to process < 2 words?

        if a.bits > WideWord::BITS && a.bits - b.bits >= Limb::BITS {
            // reduce by one limb by subtracting q•b from a
            let b_odd = b.uint.as_odd_vartime().expect("expected odd denominator");
            let b_inv = b_odd.invert_mod_limb();
            let q = a.reduce_dmod(b_odd, b_inv, b.bits);
            u.wrapping_sub_assign_mul_limb(v, q);
        } else {
            // perform an optimized bingcd reduction
            let (m, _) = a.reduce_bingcd(b);
            m.wrapping_apply_vartime(u, v);
            k += m.k;
        }

        if a.bits == 0 {
            break;
        }

        let tz = a.strip_zeros();
        v.shl_assign_vartime(tz);
        k += tz;
    }

    k
}

#[inline(always)]
const fn maybe_swap<'a>(a: &mut GcdVartimeBuffer<'a>, b: &mut GcdVartimeBuffer<'a>) -> bool {
    if a.bits < b.bits || a.uint.cmp_vartime(b.uint).is_lt() {
        mem::swap(a, b);
        true
    } else {
        false
    }
}

#[inline(always)]
const fn truncate(a: &mut GcdVartimeBuffer<'_>, b: &mut GcdVartimeBuffer<'_>) {
    let bits = if a.bits > b.bits { a.bits } else { b.bits };
    let limbs = if bits == 0 { 1 } else { bitlen::to_limbs(bits) };
    a.truncate(limbs);
    b.truncate(limbs);
}
