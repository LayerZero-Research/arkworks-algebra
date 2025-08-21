use quote::quote;

// Standalone RISC-V 64 CIOS multiplication for Fr scalar field
#[inline(always)]
fn generate_riscv_cios_multiplication(
    modulus_limbs: &[u64],
) -> proc_macro2::TokenStream {
    let modulus_0 = modulus_limbs[0];
    let modulus_1 = if modulus_limbs.len() > 1 { modulus_limbs[1] } else { 0 };
    let modulus_2 = if modulus_limbs.len() > 2 { modulus_limbs[2] } else { 0 };
    let modulus_3 = if modulus_limbs.len() > 3 { modulus_limbs[3] } else { 0 };
    
    quote! {
        #[inline(always)]
        unsafe fn riscv_mac(a0: u64, a1: u64, a2: u64) -> (u64, u64) {
            let mut sum_low: u64;
            let mut mul_lo: u64;
            let mut mul_hi: u64;
            let mut carry_bit: u64;
            let mut carry_high: u64;
            core::arch::asm!(
                "mul    {mul_lo}, {a1}, {a2}",
                "mulhu  {mul_hi}, {a1}, {a2}",
                "add    {sum_low}, {mul_lo}, {a0}",
                "sltu   {carry_bit}, {sum_low}, {a0}",
                "add    {carry_high}, {mul_hi}, {carry_bit}",
                a0 = in(reg) a0,
                a1 = in(reg) a1,
                a2 = in(reg) a2,
                mul_lo = out(reg) mul_lo,
                mul_hi = out(reg) mul_hi,
                sum_low = out(reg) sum_low,
                carry_bit = out(reg) carry_bit,
                carry_high = out(reg) carry_high,
                options(pure, nomem, nostack),
            );
            (sum_low, carry_high)
        }

        #[inline(always)]
        unsafe fn riscv_mac_init(a1: u64, a2: u64) -> (u64, u64) {
            let mut mul_lo: u64;
            let mut mul_hi: u64;
            core::arch::asm!(
                "mul    {mul_lo}, {a1}, {a2}",
                "mulhu  {mul_hi}, {a1}, {a2}",
                a1 = in(reg) a1,
                a2 = in(reg) a2,
                mul_lo = out(reg) mul_lo,
                mul_hi = out(reg) mul_hi,
                options(pure, nomem, nostack),
            );
            (mul_lo, mul_hi)
        }

        #[inline(always)]
        unsafe fn riscv_mac_discard(a0: u64, a1: u64, a2: u64) -> u64 {
            let mut throw_low: u64;
            let mut mul_lo: u64;
            let mut mul_hi: u64;
            let mut carry_bit: u64;
            let mut carry_high: u64;
            core::arch::asm!(
                "mul    {mul_lo}, {a1}, {a2}",
                "mulhu  {mul_hi}, {a1}, {a2}",
                "add    {throw_low}, {mul_lo}, {a0}",
                "sltu   {carry_bit}, {throw_low}, {a0}",
                "add    {carry_high}, {mul_hi}, {carry_bit}",
                a0 = in(reg) a0,
                a1 = in(reg) a1,
                a2 = in(reg) a2,
                mul_lo = out(reg) mul_lo,
                mul_hi = out(reg) mul_hi,
                throw_low = out(reg) throw_low,
                carry_bit = out(reg) carry_bit,
                carry_high = out(reg) carry_high,
                options(pure, nomem, nostack),
            );
            carry_high
        }

        #[inline(always)]
        unsafe fn riscv_mac_with_carry(a0: u64, a1: u64, a2: u64, carry_inout: &mut u64) -> u64 {
            let mut sum0: u64;
            let mut mul_lo: u64;
            let mut mul_hi: u64;
            let mut c1: u64;
            let mut c2: u64;
            let mut carry_out: u64;
            let cin: u64 = *carry_inout;
            core::arch::asm!(
                "mul    {mul_lo}, {a1}, {a2}",
                "mulhu  {mul_hi}, {a1}, {a2}",
                "add    {sum0}, {a0}, {mul_lo}",
                "sltu   {c1}, {sum0}, {a0}",
                "add    {sum0}, {sum0}, {cin}",
                "sltu   {c2}, {sum0}, {cin}",
                "add    {carry_out}, {mul_hi}, {c1}",
                "add    {carry_out}, {carry_out}, {c2}",
                a0 = in(reg) a0,
                a1 = in(reg) a1,
                a2 = in(reg) a2,
                cin = in(reg) cin,
                mul_lo = out(reg) mul_lo,
                mul_hi = out(reg) mul_hi,
                sum0 = out(reg) sum0,
                c1 = out(reg) c1,
                c2 = out(reg) c2,
                carry_out = out(reg) carry_out,
                options(pure, nomem, nostack),
            );
            *carry_inout = carry_out;
            sum0
        }

        #[inline(always)]
        unsafe fn riscv_mac_with_carry_zeroacc(a1: u64, a2: u64, carry_inout: &mut u64) -> u64 {
            let mut sum0: u64;
            let mut mul_lo: u64;
            let mut mul_hi: u64;
            let mut c2: u64;
            let mut carry_out: u64;
            let cin: u64 = *carry_inout;
            core::arch::asm!(
                "mul    {mul_lo}, {a1}, {a2}",
                "mulhu  {mul_hi}, {a1}, {a2}",
                "add    {sum0}, {mul_lo}, {cin}",
                "sltu   {c2}, {sum0}, {cin}",
                "add    {carry_out}, {mul_hi}, {c2}",
                a1 = in(reg) a1,
                a2 = in(reg) a2,
                cin = in(reg) cin,
                mul_lo = out(reg) mul_lo,
                mul_hi = out(reg) mul_hi,
                sum0 = out(reg) sum0,
                c2 = out(reg) c2,
                carry_out = out(reg) carry_out,
                options(pure, nomem, nostack),
            );
            *carry_inout = carry_out;
            sum0
        }

        #[inline(always)]
        unsafe fn riscv_conditional_sub_reduce(
            r0_in: u64,
            r1_in: u64,
            r2_in: u64,
            r3_in: u64,
            m0: u64,
            m1: u64,
            m2: u64,
            m3: u64,
        ) -> (u64, u64, u64, u64) {
            let mut out0: u64;
            let mut out1: u64;
            let mut out2: u64;
            let mut out3: u64;
            let mut ge: u64;    // 1 if r >= m, 0 otherwise
            let mut b0: u64;
            let mut b1: u64;
            let mut b2: u64;
            let mut m1p: u64;
            let mut m2p: u64;
            let mut m3p: u64;
            core::arch::asm!(
                // Lexicographic compare from most-significant limb to least.
                // Set ge = 1 if r >= m, else 0. Early decide without doing subtraction.
                "li     {ge}, 1",
                // Compare r3 vs m3
                "bne    {r3}, {m3}, 1f",
                // r3 == m3, compare r2
                "bne    {r2}, {m2}, 2f",
                // r2 == m2, compare r1
                "bne    {r1}, {m1}, 3f",
                // r1 == m1, compare r0
                "bne    {r0}, {m0}, 4f",
                // All equal: ge stays 1
                "j      5f",
                // r3 != m3 -> ge = (r3 > m3)
                "1:",
                "sltu   {ge}, {m3}, {r3}",
                "j      5f",
                // r2 != m2 -> ge = (r2 > m2)
                "2:",
                "sltu   {ge}, {m2}, {r2}",
                "j      5f",
                // r1 != m1 -> ge = (r1 > m1)
                "3:",
                "sltu   {ge}, {m1}, {r1}",
                "j      5f",
                // r0 != m0 -> ge = (r0 > m0)
                "4:",
                "sltu   {ge}, {m0}, {r0}",
                // Decide path based on ge
                "5:",
                // If ge == 0, return original r (early exit)
                "beqz   {ge}, 6f",
                // ge == 1: perform r - m with borrow chain
                // out0 = r0 - m0; b0 = r0 < m0
                "sub    {out0}, {r0}, {m0}",
                "sltu   {b0}, {r0}, {m0}",
                // out1 = r1 - (m1 + b0); b1 = r1 < (m1 + b0)
                "add    {m1p}, {m1}, {b0}",
                "sub    {out1}, {r1}, {m1p}",
                "sltu   {b1}, {r1}, {m1p}",
                // out2 = r2 - (m2 + b1); b2 = r2 < (m2 + b1)
                "add    {m2p}, {m2}, {b1}",
                "sub    {out2}, {r2}, {m2p}",
                "sltu   {b2}, {r2}, {m2p}",
                // out3 = r3 - (m3 + b2)
                "add    {m3p}, {m3}, {b2}",
                "sub    {out3}, {r3}, {m3p}",
                "j      7f",
                // ge == 0 path: keep r
                "6:",
                "mv     {out0}, {r0}",
                "mv     {out1}, {r1}",
                "mv     {out2}, {r2}",
                "mv     {out3}, {r3}",
                "7:",
                r0 = in(reg) r0_in,
                r1 = in(reg) r1_in,
                r2 = in(reg) r2_in,
                r3 = in(reg) r3_in,
                m0 = in(reg) m0,
                m1 = in(reg) m1,
                m2 = in(reg) m2,
                m3 = in(reg) m3,
                out0 = out(reg) out0,
                out1 = out(reg) out1,
                out2 = out(reg) out2,
                out3 = out(reg) out3,
                ge = out(reg) ge,
                b0 = out(reg) b0,
                b1 = out(reg) b1,
                b2 = out(reg) b2,
                m1p = out(reg) m1p,
                m2p = out(reg) m2p,
                m3p = out(reg) m3p,
                options(pure, nomem, nostack),
            );
            (out0, out1, out2, out3)
        }

        let lhs = (a.0).0;
        let rhs = (b.0).0;
        let modulo: [u64; 4] = [#modulus_0, #modulus_1, #modulus_2, #modulus_3];
        let mut accumulator: [u64; 4] = [0; 4];
        let mut k;
        let mut carry1: u64;
        let mut carry2: u64;

        let inv: u64 = Self::INV;

        // Macros to choose init vs non-init paths at compile time without branching
        macro_rules! riscv_mac_generator {
            (0) => {{
                (accumulator[0], carry1) = unsafe { riscv_mac_init(lhs[0], rhs[0]) };
            }};
            ($i:tt) => {{
                (accumulator[0], carry1) = unsafe { riscv_mac(accumulator[0], lhs[0], rhs[$i]) };
            }};
        }

        macro_rules! riscv_conditional_mac_with_carry_generator {
            (0, $idx:tt) => {{
                accumulator[$idx] = unsafe { riscv_mac_with_carry_zeroacc(lhs[$idx], rhs[0], &mut carry1) };
            }};
            ($i:tt, $idx:tt) => {{
                accumulator[$idx] = unsafe { riscv_mac_with_carry(accumulator[$idx], lhs[$idx], rhs[$i], &mut carry1) };
            }};
        }

        // Unroll all rounds with a local macro that specializes i == 0 vs i > 0 at compile time
        macro_rules! riscv_cios_round {
            ($i:tt) => {{
                riscv_mac_generator!($i);
                k = accumulator[0].wrapping_mul(inv);
                carry2 = unsafe { riscv_mac_discard(accumulator[0], k, modulo[0]) };

                riscv_conditional_mac_with_carry_generator!($i, 1);
                accumulator[0] = unsafe { riscv_mac_with_carry(accumulator[1], k, modulo[1], &mut carry2) };

                riscv_conditional_mac_with_carry_generator!($i, 2);
                accumulator[1] = unsafe { riscv_mac_with_carry(accumulator[2], k, modulo[2], &mut carry2) };

                riscv_conditional_mac_with_carry_generator!($i, 3);
                accumulator[2] = unsafe { riscv_mac_with_carry(accumulator[3], k, modulo[3], &mut carry2) };
                accumulator[3] = carry1.wrapping_add(carry2);
            }};
        }

        riscv_cios_round!(0);
        riscv_cios_round!(1);
        riscv_cios_round!(2);
        riscv_cios_round!(3);

        // Final conditional subtract
        (accumulator[0], accumulator[1], accumulator[2], accumulator[3]) = unsafe {
            riscv_conditional_sub_reduce(accumulator[0], accumulator[1], accumulator[2], accumulator[3], modulo[0], modulo[1], modulo[2], modulo[3])
        };
        (a.0).0 = [accumulator[0], accumulator[1], accumulator[2], accumulator[3]];
    }
}

// Helper function to generate the CIOS multiplication
fn generate_cios_multiplication(
    can_use_no_carry_mul_opt: bool,
    num_limbs: usize,
    modulus_limbs: &[u64],
    modulus_has_spare_bit: bool,
) -> proc_macro2::TokenStream {
    let mut body = proc_macro2::TokenStream::new();
    let modulus_0 = modulus_limbs[0];
    if can_use_no_carry_mul_opt {
        // This modular multiplication algorithm uses Montgomery
        // reduction for efficient implementation. It also additionally
        // uses the "no-carry optimization" outlined
        // [here](https://hackmd.io/@gnark/modular_multiplication) if
        // `MODULUS` has (a) a non-zero MSB, and (b) at least one
        // zero bit in the rest of the modulus.

        let mut default = proc_macro2::TokenStream::new();
        default.extend(quote! { let mut r = [0u64; #num_limbs]; });
        for i in 0..num_limbs {
            default.extend(quote! {
                let mut carry1 = 0u64;
                r[0] = fa::mac(r[0], (a.0).0[0], (b.0).0[#i], &mut carry1);
                let k = r[0].wrapping_mul(Self::INV);
                let mut carry2 = 0u64;
                fa::mac_discard(r[0], k, #modulus_0, &mut carry2);
            });
            for (j, modulus_j) in modulus_limbs.iter().enumerate().take(num_limbs).skip(1) {
                let idx = j - 1;
                default.extend(quote! {
                    r[#j] = fa::mac_with_carry(r[#j], (a.0).0[#j], (b.0).0[#i], &mut carry1);
                    r[#idx] = fa::mac_with_carry(r[#j], k, #modulus_j, &mut carry2);
                });
            };
            default.extend(quote!(r[#num_limbs - 1] = carry1 + carry2;));
        }
        default.extend(quote!((a.0).0 = r;));
        // Avoid using assembly for `N == 1`.
        if (2..=6).contains(&num_limbs) {
            body.extend(quote!({
                if cfg!(all(
                    feature = "asm",
                    target_feature = "bmi2",
                    target_feature = "adx",
                    target_arch = "x86_64"
                )) {
                    #[cfg(
                        all(
                            feature = "asm",
                            target_feature = "bmi2",
                            target_feature = "adx",
                            target_arch = "x86_64"
                        )
                    )]
                    #[allow(unsafe_code, unused_mut)]
                    ark_ff::x86_64_asm_mul!(#num_limbs, (a.0).0, (b.0).0);
                } else {
                    #[cfg(
                        not(all(
                            feature = "asm",
                            target_feature = "bmi2",
                            target_feature = "adx",
                            target_arch = "x86_64"
                        ))
                    )]
                    {
                        #default
                    }
                }
            }))
        } else {
            body.extend(quote!({ #default }))
        }
        body.extend(quote!(__subtract_modulus(a);));
    } else {
        // We use standard CIOS
        let double_limbs = num_limbs * 2;
        body.extend(quote! {
            let mut scratch = [0u64; #double_limbs];
        });
        for i in 0..num_limbs {
            body.extend(quote! { let mut carry = 0u64; });
            for j in 0..num_limbs {
                let k = i + j;
                body.extend(quote!{scratch[#k] = fa::mac_with_carry(scratch[#k], (a.0).0[#i], (b.0).0[#j], &mut carry);});
            }
            body.extend(quote! { scratch[#i + #num_limbs] = carry; });
        }
        body.extend(quote!( let mut carry2 = 0u64; ));
        for i in 0..num_limbs {
            body.extend(quote! {
                let tmp = scratch[#i].wrapping_mul(Self::INV);
                let mut carry = 0u64;
                fa::mac(scratch[#i], tmp, #modulus_0, &mut carry);
            });
            for j in 1..num_limbs {
                let modulus_j = modulus_limbs[j];
                let k = i + j;
                body.extend(quote!(scratch[#k] = fa::mac_with_carry(scratch[#k], tmp, #modulus_j, &mut carry);));
            }
            body.extend(quote!(carry2 = fa::adc(&mut scratch[#i + #num_limbs], carry, carry2);));
        }
        body.extend(quote! {
            (a.0).0 = scratch[#num_limbs..].try_into().unwrap();
        });
        if modulus_has_spare_bit {
            body.extend(quote!(__subtract_modulus(a);));
        } else {
            body.extend(quote!(__subtract_modulus_with_carry(a, carry2 != 0);));
        }
    }
    body
}

#[inline(always)]
pub(super) fn mul_assign_impl(
    can_use_no_carry_mul_opt: bool,
    num_limbs: usize,
    modulus_limbs: &[u64],
    modulus_has_spare_bit: bool,
) -> proc_macro2::TokenStream {
    // For N=4 and BN254 Fr modulus, generate RISC-V implementation
    // Only enable the RISC-V path for riscv64 targets via cfg below.
    const BN254_FR_MODULUS_LIMBS: [u64; 4] = [
        0x43e1f593f0000001,
        0x2833e84879b97091,
        0xb85045b68181585d,
        0x30644e72e131a029,
    ];
    if num_limbs == 4 && modulus_limbs.len() == 4 && modulus_limbs == BN254_FR_MODULUS_LIMBS {
        let riscv_impl = generate_riscv_cios_multiplication(modulus_limbs);
        let default_impl = generate_cios_multiplication(
            can_use_no_carry_mul_opt,
            num_limbs,
            modulus_limbs,
            modulus_has_spare_bit,
        );
        return quote! {
            #[cfg(all(feature = "asm", target_arch = "riscv64"))]
            {
                #[allow(unsafe_code, unused_mut, unused_variables)]
                {
                    #riscv_impl
                }
            }
            #[cfg(not(all(feature = "asm", target_arch = "riscv64")))]
            {
                #default_impl
            }
        };
    }
    generate_cios_multiplication(
        can_use_no_carry_mul_opt,
        num_limbs,
        modulus_limbs,
        modulus_has_spare_bit,
    )
}
