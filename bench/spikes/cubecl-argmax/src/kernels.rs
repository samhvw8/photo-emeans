//! Fused nearest-centroid kernels, written once in CubeCL.
//!
//! Both kernels take a row batch `x` (`b x d`, f32, row-major) and centroids
//! `c` (`k x d`), and write per row the argmax over `j` of `x . c_j` and its
//! score. The `b x k` score matrix never exists in global memory. Ties keep the
//! lower centroid id. `d` must be a multiple of 4 (rows load as `Vector<f32, 4>`).
//!
//! - `assign_fma`: scalar FMA, register tiled (portable to every runtime).
//! - `assign_mma`: cooperative matrix (`cmma`) tiles. f32 8x8x8 maps to Metal
//!   `simdgroup_float8x8`. On CUDA the same source takes tf32 16x16x8 or
//!   bf16 16x16x16 fragments (operand type is a generic).

use cubecl::prelude::*;

/// Is (a, ja) better than (b, jb)? Higher score wins; ties keep the lower id.
#[cube]
fn better(a: f32, ja: u32, b: f32, jb: u32) -> bool {
    a > b || (a == b && ja < jb)
}

/// Scalar FMA kernel. Cube = 16 x 16 units. A cube owns `BM = 16 * TM` rows,
/// walks the centroids in tiles of `BN = 16 * TN`, and stages `BK` columns of
/// both operands in shared memory per step. Unit (tx, ty) owns rows
/// `ty + 16 i` and centroids `tx + 16 j` of each tile.
#[cube(launch_unchecked)]
pub fn assign_fma(
    x: &[Vector<f32, Const<4>>],
    c: &[Vector<f32, Const<4>>],
    labels: &mut [u32],
    scores: &mut [f32],
    rows: u32,
    k: u32,
    d: u32,
    #[comptime] tm: usize,
    #[comptime] tn: usize,
    #[comptime] bk: usize,
) {
    let bm = comptime!(16 * tm);
    let bn = comptime!(16 * tn);
    let tx = UNIT_POS_X as usize;
    let ty = UNIT_POS_Y as usize;
    let tid = UNIT_POS as usize;
    let row0 = CUBE_POS_X as usize * bm;
    let rows = rows as usize;
    let k = k as usize;
    let d = d as usize;
    let d4 = d / 4;
    let bk4 = comptime!(bk / 4);

    // k-major tiles: xs[kk * bm + r], cs[kk * bn + j].
    let mut xs = Shared::<[f32]>::new_slice(comptime!(bk * 16 * tm));
    let mut cs = Shared::<[f32]>::new_slice(comptime!(bk * 16 * tn));
    let mut red_v = Shared::<[f32]>::new_slice(comptime!(16 * 16 * tm));
    let mut red_j = Shared::<[u32]>::new_slice(comptime!(16 * 16 * tm));

    let mut best = Array::<f32>::new(tm);
    let mut bj = Array::<u32>::new(tm);
    #[unroll]
    for i in 0..tm {
        best[i] = f32::new(-1.0e30f32);
        bj[i] = 0u32;
    }
    let mut acc = Array::<f32>::new(comptime!(tm * tn));
    let mut a = Array::<f32>::new(tm);
    let mut bv = Array::<f32>::new(tn);

    let mut n0 = 0usize;
    while n0 < k {
        #[unroll]
        for t in 0..comptime!(tm * tn) {
            acc[t] = f32::new(0.0f32);
        }
        let mut k0 = 0usize;
        while k0 < d {
            sync_cube();
            // Each unit loads float4s; transpose into k-major shared tiles.
            let mut e = tid;
            while e < comptime!(16 * tm * bk / 4) {
                let r = e / bk4;
                let c4 = (e % bk4) * 4;
                let gr = row0 + r;
                let mut v = Vector::<f32, Const<4>>::new(0.0);
                if gr < rows {
                    v = x[gr * d4 + (k0 + c4) / 4];
                }
                #[unroll]
                for q in 0..4usize {
                    xs[(c4 + q) * bm + r] = v.extract(q);
                }
                e += 256;
            }
            let mut e = tid;
            while e < comptime!(16 * tn * bk / 4) {
                let r = e / bk4;
                let c4 = (e % bk4) * 4;
                let gj = n0 + r;
                let mut v = Vector::<f32, Const<4>>::new(0.0);
                if gj < k {
                    v = c[gj * d4 + (k0 + c4) / 4];
                }
                #[unroll]
                for q in 0..4usize {
                    cs[(c4 + q) * bn + r] = v.extract(q);
                }
                e += 256;
            }
            sync_cube();
            #[unroll]
            for kk in 0..bk {
                #[unroll]
                for i in 0..tm {
                    a[i] = xs[kk * bm + ty + 16 * i];
                }
                #[unroll]
                for j in 0..tn {
                    bv[j] = cs[kk * bn + tx + 16 * j];
                }
                #[unroll]
                for i in 0..tm {
                    #[unroll]
                    for j in 0..tn {
                        acc[i * tn + j] = fma(a[i], bv[j], acc[i * tn + j]);
                    }
                }
            }
            k0 += bk;
        }
        // Centroid ids grow with j and with n0, so a strict > keeps the lowest id.
        #[unroll]
        for j in 0..tn {
            let cj = n0 + tx + 16 * j;
            if cj < k {
                #[unroll]
                for i in 0..tm {
                    let v = acc[i * tn + j];
                    if v > best[i] {
                        best[i] = v;
                        bj[i] = cj as u32;
                    }
                }
            }
        }
        n0 += bn;
    }
    // 16 candidates per row (one per tx) -> shared memory -> one unit per row.
    #[unroll]
    for i in 0..tm {
        let r = ty + 16 * i;
        red_v[r * 16 + tx] = best[i];
        red_j[r * 16 + tx] = bj[i];
    }
    sync_cube();
    if tid < bm {
        let mut bb = red_v[tid * 16];
        let mut jj = red_j[tid * 16];
        #[unroll]
        for t in 1..16usize {
            let v = red_v[tid * 16 + t];
            let j = red_j[tid * 16 + t];
            if better(v, j, bb, jj) {
                bb = v;
                jj = j;
            }
        }
        let gr = row0 + tid;
        if gr < rows {
            labels[gr] = jj;
            scores[gr] = bb;
        }
    }
}

/// Cooperative-matrix kernel. Cube = `wm * wn` planes. A cube owns `bm` rows
/// and walks centroids in tiles of `bn`; plane (pm, pn) computes a
/// `(bm / wm) x (bn / wn)` block as `fm x fn_` fragments of `mm x nn`.
/// Operands are staged in shared memory as `A` (f32 or cast to the fragment
/// type on load). The accumulator fragments are opaque, so after each centroid
/// tile they are stored to shared memory and scanned (two units per row).
/// `pad` (shared-memory row padding, in elements) must keep CUDA WMMA's
/// leading-dimension rule: a multiple of 8 for 16-bit operands, 4 for 32-bit.
#[cube(launch_unchecked)]
pub fn assign_mma<A: Float>(
    x: &[Vector<f32, Const<4>>],
    c: &[Vector<f32, Const<4>>],
    labels: &mut [u32],
    scores: &mut [f32],
    rows: u32,
    k: u32,
    d: u32,
    #[comptime] bm: usize,
    #[comptime] bn: usize,
    #[comptime] bk: usize,
    #[comptime] wm: usize,
    #[comptime] wn: usize,
    #[comptime] mm: usize,
    #[comptime] nn: usize,
    #[comptime] kk_: usize,
    #[comptime] pad: usize,
) {
    let nt = comptime!(wm * wn * 32);
    let ldk = comptime!(bk + pad);
    let fm = comptime!(bm / wm / mm);
    let fn_ = comptime!(bn / wn / nn);
    let tid = UNIT_POS as usize;
    let plane = tid / 32;
    let pm = plane / wn;
    let pn = plane % wn;
    let row0 = CUBE_POS_X as usize * bm;
    let rows = rows as usize;
    let k = k as usize;
    let d = d as usize;
    let d4 = d / 4;
    let bk4 = comptime!(bk / 4);

    let mut xs = Shared::<[A]>::new_slice(comptime!(bm * (bk + pad)));
    let mut cs = Shared::<[A]>::new_slice(comptime!(bn * (bk + pad)));
    let mut sc = Shared::<[f32]>::new_slice(comptime!(bm * bn));
    let mut red_v = Shared::<[f32]>::new_slice(comptime!(bm * 2));
    let mut red_j = Shared::<[u32]>::new_slice(comptime!(bm * 2));

    // Scan state: units 0..2*bm own (row = t / 2, half = t % 2).
    let mut best = f32::new(-1.0e30f32);
    let mut bj = 0u32;

    let mut n0 = 0usize;
    while n0 < k {
        let mut accs = Sequence::<cmma::Matrix<f32>>::new();
        #[unroll]
        for _t in 0..comptime!(fm * fn_) {
            accs.push(cmma::Matrix::<f32>::from_value(
                cmma::MatrixIdent::Accumulator,
                mm,
                nn,
                kk_,
                cmma::MatrixLayout::Undefined,
                0.0,
            ));
        }
        let mut k0 = 0usize;
        while k0 < d {
            sync_cube();
            let mut e = tid;
            while e < comptime!(bm * bk / 4) {
                let r = e / bk4;
                let c4 = (e % bk4) * 4;
                let gr = row0 + r;
                let mut v = Vector::<f32, Const<4>>::new(0.0);
                if gr < rows {
                    v = x[gr * d4 + (k0 + c4) / 4];
                }
                #[unroll]
                for q in 0..4usize {
                    xs[r * ldk + c4 + q] = A::cast_from(v.extract(q));
                }
                e += nt;
            }
            let mut e = tid;
            while e < comptime!(bn * bk / 4) {
                let r = e / bk4;
                let c4 = (e % bk4) * 4;
                let gj = n0 + r;
                let mut v = Vector::<f32, Const<4>>::new(0.0);
                if gj < k {
                    v = c[gj * d4 + (k0 + c4) / 4];
                }
                #[unroll]
                for q in 0..4usize {
                    cs[r * ldk + c4 + q] = A::cast_from(v.extract(q));
                }
                e += nt;
            }
            sync_cube();
            #[unroll]
            for s in 0..comptime!(bk / kk_) {
                let ko = s * kk_;
                #[unroll]
                for i in 0..fm {
                    let ra = (pm * fm + i) * mm;
                    let a = cmma::Matrix::<A>::from_slice(
                        cmma::MatrixIdent::A,
                        mm,
                        nn,
                        kk_,
                        cmma::MatrixLayout::RowMajor,
                        &xs[ra * ldk + ko..ra * ldk + ko + (mm - 1) * ldk + kk_],
                        ldk as u32,
                    );
                    #[unroll]
                    for j in 0..fn_ {
                        let cb = (pn * fn_ + j) * nn;
                        // B(kk, n) = c[n][kk]: column-major with stride ldk.
                        let b = cmma::Matrix::<A>::from_slice(
                            cmma::MatrixIdent::B,
                            mm,
                            nn,
                            kk_,
                            cmma::MatrixLayout::ColMajor,
                            &cs[cb * ldk + ko..cb * ldk + ko + (nn - 1) * ldk + kk_],
                            ldk as u32,
                        );
                        let acc = accs.index(i * fn_ + j);
                        cmma::execute::<A, A, f32, f32, cmma::Plane>(&a, &b, acc, acc);
                    }
                }
            }
            k0 += bk;
        }
        sync_cube();
        #[unroll]
        for i in 0..fm {
            #[unroll]
            for j in 0..fn_ {
                let r = (pm * fm + i) * mm;
                let cc = (pn * fn_ + j) * nn;
                let off = r * bn + cc;
                cmma::store(
                    &mut sc[off..off + (mm - 1) * bn + nn],
                    accs.index(i * fn_ + j),
                    bn as u32,
                    cmma::MatrixLayout::RowMajor,
                );
            }
        }
        sync_cube();
        let mut t = tid;
        while t < comptime!(2 * bm) {
            let r = t / 2;
            let h = t % 2;
            let half = comptime!(bn / 2);
            #[unroll]
            for q in 0..half {
                let col = h * half + q;
                let cj = n0 + col;
                let v = sc[r * bn + col];
                if cj < k && v > best {
                    best = v;
                    bj = cj as u32;
                }
            }
            t += nt;
        }
        n0 += bn;
    }
    // nt >= 2 * bm is required (asserted on the host), so each unit owns at most one (row, half).
    if tid < comptime!(2 * bm) {
        red_v[tid] = best;
        red_j[tid] = bj;
    }
    sync_cube();
    if tid < bm {
        let mut bb = red_v[2 * tid];
        let mut jj = red_j[2 * tid];
        let v = red_v[2 * tid + 1];
        let j = red_j[2 * tid + 1];
        if better(v, j, bb, jj) {
            bb = v;
            jj = j;
        }
        let gr = row0 + tid;
        if gr < rows {
            labels[gr] = jj;
            scores[gr] = bb;
        }
    }
}
