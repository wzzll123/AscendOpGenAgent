import tilelang
from tilelang import DataType, language as T


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6, 7], pass_configs=pass_configs)
def flash_attention_fwd(
    batch,
    seq_len,
    heads,
    dim,
):
    block_M, block_N = 64, 64
    prelaunch = 2
    ring_slots = prelaunch + 1
    num_physical_cores = 20

    dtype = "float16"
    accum_dtype = "float"

    sm_scale = (1.0 / dim) ** 0.5

    shape = [batch, heads, seq_len, dim]
    block_num = seq_len // block_M * heads * batch
    kv_loops = T.ceildiv(seq_len, block_N)
    used_core_num = min(num_physical_cores, block_num)
    tasks_per_core = T.ceildiv(block_num, used_core_num)

    @T.prim_func
    def main(
        Q: T.Tensor(shape, dtype),  # type: ignore
        K: T.Tensor(shape, dtype),  # type: ignore
        V: T.Tensor(shape, dtype),  # type: ignore
        Output: T.Tensor(shape, dtype),  # type: ignore
        workspace_1: T.Tensor([used_core_num, ring_slots, block_M, block_N], accum_dtype),
        workspace_2: T.Tensor([used_core_num, ring_slots, block_M, block_N], dtype),
        workspace_3: T.Tensor([used_core_num, ring_slots, block_M, dim], accum_dtype),
        workspace_meta: T.Tensor([used_core_num, ring_slots, block_M, 2], accum_dtype),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            q_l1 = T.alloc_L1([block_M, dim], dtype)
            k_l1 = T.alloc_L1([block_N, dim], dtype)
            v_l1 = T.alloc_L1([block_N, dim], dtype)

            acc_s_l1 = T.alloc_L1([block_M, block_N], dtype)

            acc_s_l0c = T.alloc_L0C([block_M, block_N], accum_dtype)
            acc_o_l0c = T.alloc_L0C([block_M, dim], accum_dtype)

            acc_o = T.alloc_ub([block_M // 2, dim], accum_dtype)
            sumexp = T.alloc_ub([block_M // 2], accum_dtype)
            m_i = T.alloc_ub([block_M // 2], accum_dtype)

            acc_s_ub = T.alloc_ub([block_M // 2, block_N], accum_dtype)
            m_i_prev = T.alloc_ub([block_M // 2], accum_dtype)
            acc_s_ub_ = T.alloc_ub([block_M // 2, block_N], accum_dtype)
            sumexp_i_ub = T.alloc_ub([block_M // 2], accum_dtype)
            acc_s_half = T.alloc_ub([block_M // 2, block_N], dtype)
            acc_o_ub = T.alloc_ub([block_M // 2, dim], accum_dtype)
            acc_o_half = T.alloc_ub([block_M // 2, dim], dtype)
            alpha_ub = T.alloc_ub([block_M // 2], accum_dtype)
            sumexp_meta_ub = T.alloc_ub([block_M // 2], accum_dtype)

            for local_idx in T.serial(tasks_per_core):
                block_idx = cid * tasks_per_core + local_idx
                bx = block_idx % (seq_len // block_M)
                by = block_idx // (seq_len // block_M) % heads
                bz = block_idx // (seq_len // block_M) // heads % batch

                if block_idx < block_num:
                    with T.Scope("C"):
                        T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], q_l1)
                    with T.Scope("V"):
                        T.tile.fill(acc_o, 0.0)
                        T.tile.fill(sumexp, 0.0)
                        T.tile.fill(m_i, -2**30)

                    for t in T.serial(kv_loops + prelaunch):
                        if t < kv_loops:
                            slot_prod = t % ring_slots
                            with T.Scope("C"):
                                T.copy(K[bz, by, t * block_N:(t + 1) * block_N, :], k_l1)
                                T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                                T.copy(acc_s_l0c, workspace_1[cid, slot_prod, :, :])
                                T.set_cross_flag("FIX", 0)

                            with T.Scope("V"):
                                T.tile.fill(acc_s_ub, 0.0)
                                T.copy(m_i, m_i_prev)
                                T.wait_cross_flag(0)
                                T.copy(
                                    workspace_1[
                                        cid,
                                        slot_prod,
                                        vid * block_M // 2:vid * block_M // 2 + block_M // 2,
                                        :,
                                    ],
                                    acc_s_ub_,
                                )
                                T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
                                T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
                                T.reduce_max(acc_s_ub, m_i, dim=-1)
                                T.tile.max(m_i, m_i, m_i_prev)
                                T.tile.sub(m_i_prev, m_i_prev, m_i)
                                T.tile.exp(m_i_prev, m_i_prev)
                                for h_i in range(block_M // 2):
                                    T.tile.sub(acc_s_ub[h_i, :], acc_s_ub[h_i, :], m_i[h_i])
                                T.tile.exp(acc_s_ub, acc_s_ub)
                                T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)

                                T.copy(acc_s_ub, acc_s_half)
                                T.copy(
                                    acc_s_half,
                                    workspace_2[
                                        cid,
                                        slot_prod,
                                        vid * block_M // 2:vid * block_M // 2 + block_M // 2,
                                        :,
                                    ],
                                )
                                for h_i in range(block_M // 2):
                                    workspace_meta[
                                        cid,
                                        slot_prod,
                                        vid * block_M // 2 + h_i,
                                        0,
                                    ] = m_i_prev[h_i]
                                    workspace_meta[
                                        cid,
                                        slot_prod,
                                        vid * block_M // 2 + h_i,
                                        1,
                                    ] = sumexp_i_ub[h_i]
                                T.set_cross_flag("MTE3", 1)

                        if t >= prelaunch:
                            now_k = t - prelaunch
                            slot_cons = now_k % ring_slots
                            with T.Scope("C"):
                                T.wait_cross_flag(1)
                                T.copy(workspace_2[cid, slot_cons, :, :], acc_s_l1)
                                T.copy(V[bz, by, now_k * block_N:(now_k + 1) * block_N, :], v_l1)
                                T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)
                                T.copy(acc_o_l0c, workspace_3[cid, slot_cons, :, :])
                                T.set_cross_flag("FIX", 2)

                            with T.Scope("V"):
                                T.wait_cross_flag(2)
                                for h_i in range(block_M // 2):
                                    alpha_ub[h_i] = workspace_meta[
                                        cid,
                                        slot_cons,
                                        vid * block_M // 2 + h_i,
                                        0,
                                    ]
                                    sumexp_meta_ub[h_i] = workspace_meta[
                                        cid,
                                        slot_cons,
                                        vid * block_M // 2 + h_i,
                                        1,
                                    ]
                                T.copy(
                                    workspace_3[
                                        cid,
                                        slot_cons,
                                        vid * block_M // 2:vid * block_M // 2 + block_M // 2,
                                        :,
                                    ],
                                    acc_o_ub,
                                )
                                for h_i in range(block_M // 2):
                                    T.tile.mul(acc_o[h_i, :], acc_o[h_i, :], alpha_ub[h_i])
                                T.tile.add(acc_o, acc_o, acc_o_ub)
                                T.tile.mul(sumexp, sumexp, alpha_ub)
                                T.tile.add(sumexp, sumexp, sumexp_meta_ub)

                    with T.Scope("V"):
                        for h_i in range(block_M // 2):
                            T.tile.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i])

                        T.copy(acc_o, acc_o_half)
                        T.copy(
                            acc_o_half,
                            Output[
                                bz,
                                by,
                                bx * block_M + vid * block_M // 2:bx * block_M + vid * block_M // 2 + block_M // 2,
                                :,
                            ],
                        )

    return main
