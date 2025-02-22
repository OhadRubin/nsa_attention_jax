import jax
from jax import lax
import jax.numpy as jnp
from einops import rearrange
from jax.numpy import einsum
from functools import partial
import numpy as np

# Constants
EPSILON = 1e-10
DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


def gather_slice(array, index, size, fill_value, mult=False):
    """
    usage:
    >> result = gather_slice(jnp.arange(21), 4, 4, mult=True)
    [16 17 18 19]
    >> result = gather_slice(jnp.arange(21), 16, 4, mult=False)
    [16 17 18 19]
    """
    if mult:
      index = index*size
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(0,),
        collapsed_slice_dims=(),
        start_index_map=(0,)
    )
    slice_sizes = (size,)
    return jax.lax.gather(
        operand=array,
        start_indices=jnp.array(index)[..., None],
        dimension_numbers=dimension_numbers,
        slice_sizes=slice_sizes,
        mode="fill",
        fill_value=fill_value
    )

def _calculate_num_tiles(x: int, tx: int) -> int:
  tiles, rem = divmod(x, tx)
  if rem:
    raise ValueError(f"{x} must be divisible by x-dimension tile size ({tx}).")
  return tiles



def make_group_metadata(
    *,
    group_sizes: jnp.ndarray,
    m: int,
    tm: int,
    start_group: jnp.ndarray,
    num_nonzero_groups: int,
    visit_empty_groups: bool = True,
):
  """Create the metadata needed for grouped matmul computation.

  Args:
    group_sizes: A 1d, jnp.ndarray with shape [num_groups] and jnp.int32 dtype.
    m: The number of rows in lhs.
    tm: The m-dimension tile size being used.
    start_group: The group in group sizes to start computing from. This is
      particularly useful for when rhs num_groups is sharded.
    num_nonzero_groups: Number of groups in group sizes to compute on. Useful in
      combination with group_offset.
    visit_empty_groups: If True, do not squeeze tiles for empty groups out of
      the metadata. This is necessary for tgmm, where we at least need to zero
      the output for each group.

  Returns:
    tuple of:
      group_offsets: A 1d, jnp.ndarray with shape [num_groups+1] and jnp.int32
        dtype. group_offsets[i] indicates the row at which group [i] starts in
        the lhs matrix and group_offsets[i-1] = m.
      group_ids: A 1d, jnp.ndarray with shape [m_tiles + num_groups] and
        jnp.int32 dtype. group_ids[i] indicates which group grid index 'i' will
        work on.
      m_tile_ids: A 1d, jnp.ndarray with shape [m_tiles + num_groups] and
        jnp.int32. m_tile_ids[i] indicates which m-dimension tile grid index 'i'
        will work on.
    num_tiles: The number of m-dimension tiles to execute.
  """
  num_groups = group_sizes.shape[0]
  end_group = start_group + num_nonzero_groups - 1

  # Calculate the offset of each group, starting at zero. This metadata is
  # similar to row offsets in a CSR matrix. The following properties hold:
  #
  # group_offsets.shape = [num_groups + 1]
  # group_offsets[0] = 0
  # group_offsets[num_groups] = m
  #
  # The row at which group 'i' starts is group_offsets[i].
  group_ends = jnp.cumsum(group_sizes)
  group_offsets = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), group_ends])

  # Assign a group id to each grid index.
  #
  # If a group starts somewhere other than the start of a tile or ends somewhere
  # other than the end of a tile we need to compute that full tile. Calculate
  # the number of tiles for each group by rounding their end up to the nearest
  # 'tm' and their start down to the nearest 'tm'.
    #
  # (1) Round the group_ends up to the nearest multiple of 'tm'.
  #
  # NOTE: This does not change group_offsets[num_groups], which is m
  # (because we enforce m is divisible by tm).
  rounded_group_ends = ((group_ends + tm - 1) // tm * tm).astype(jnp.int32)

  # (2) Round the group_starts down to the nearest multiple of 'tm'.
  group_starts = jnp.concatenate(
      [jnp.zeros(1, dtype=jnp.int32), group_ends[:-1]]
  )
  rounded_group_starts = group_starts // tm * tm

  # (3) Calculate the number of rows in each group.
  #
  # NOTE: Handle zero-sized groups as a special case. If the start for a
  # zero-sized group is not divisible by 'tm' its start will be rounded down and
  # its end will be rounded up such that its size will become 1 tile here.
  rounded_group_sizes = rounded_group_ends - rounded_group_starts
  rounded_group_sizes = jnp.where(group_sizes == 0, 0, rounded_group_sizes)

  # (4) Convert the group sizes from units of rows to unit of 'tm' sized tiles.
  #
  # An m-dimension tile is 'owned' by group 'i' if the first row of the tile
  # belongs to group 'i'. In addition to owned tiles, each group can have 0 or 1
  # initial partial tiles if it's first row does not occur in the first row of a
  # tile. The '0-th' group never has a partial tile because it always starts at
  # the 0-th row.
  #
  # If no group has a partial tile, the total number of tiles is equal to
  # 'm // tm'. If every group has a partial except the 0-th group, the total
  # number of tiles is equal to 'm // tm + num_groups - 1'. Thus we know that
  #
  # tiles_m <= group_tiles.sum() <= tiles_m + num_groups - 1
  #
  # Where tiles_m = m // tm.
  #
  # NOTE: All group sizes are divisible by 'tm' because of the rounding in steps
  # (1) and (2) so this division is exact.
  group_tiles = rounded_group_sizes // tm

  if visit_empty_groups:
    # Insert one tile for empty groups.
    group_tiles = jnp.where(group_sizes == 0, 1, group_tiles)

  # Create the group ids for each grid index based on the tile counts for each
  # group.
  #
  # NOTE: This repeat(...) will pad group_ids with the final group id if
  # group_tiles.sum() < tiles_m + num_groups - 1. The kernel grid will be sized
  # such that we only execute the necessary number of tiles.
  tiles_m = _calculate_num_tiles(m, tm)
  group_ids = jnp.repeat(
      jnp.arange(num_groups, dtype=jnp.int32),
      group_tiles,
      total_repeat_length=tiles_m + num_groups - 1,
  )

  # Assign an m-dimension tile id to each grid index.
  #
  # NOTE: Output tiles can only be re-visited consecutively. The following
  # procedure guarantees that m-dimension tile indices respect this.

  # (1) Calculate how many times each m-dimension tile will be visited.
  #
  # Each tile is guaranteed to be visited once by the group that owns the tile.
  # The remaining possible visits occur when a group starts inside of a tile at
  # a position other than the first row. We can calculate which m-dimension tile
  # each group starts in by floor-dividing its offset with `tm` and then count
  # tile visits with a histogram.
  #
  # To avoid double counting tile visits from the group that owns the tile,
  # filter these out by assigning their tile id to `tile_m` (one beyond the max)
  # such that they're ignored by the subsequent histogram. Also filter out any
  # group which is empty.
  #
  # TODO(tgale): Invert the 'partial_tile_mask' predicates to be more clear.
  partial_tile_mask = jnp.logical_or(
      (group_offsets[:-1] % tm) == 0, group_sizes == 0
  )

  # Explicitly enable tiles for zero sized groups, if specified. This covers
  # zero sized groups that start on a tile-aligned row and those that do not.
  if visit_empty_groups:
    partial_tile_mask = jnp.where(group_sizes == 0, 0, partial_tile_mask)

  partial_tile_ids = jnp.where(
      partial_tile_mask, tiles_m, group_offsets[:-1] // tm
  )

  tile_visits = (
      jnp.histogram(partial_tile_ids, bins=tiles_m, range=(0, tiles_m - 1))[0]
      + 1
  )

  # Create the m-dimension tile ids for each grid index based on the visit
  # counts for each tile.
  m_tile_ids = jnp.repeat(
      jnp.arange(tiles_m, dtype=jnp.int32),
      tile_visits.astype(jnp.int32),
      total_repeat_length=tiles_m + num_groups - 1,
  )

  # Account for sharding.
  #
  # Find the start of the groups owned by our shard and shift the group_ids and
  # m_tile_ids s.t. the metadata for our tiles are at the front of the arrays.
  #
  # TODO(tgale): Move this offset into the kernel to avoid these rolls.
  first_tile_in_shard = (group_ids < start_group).sum()
  group_ids = jnp.roll(group_ids, shift=-first_tile_in_shard, axis=0)
  m_tile_ids = jnp.roll(m_tile_ids, shift=-first_tile_in_shard, axis=0)

  # Calculate the number of tiles we need to compute for our shard.
  #
  # Remove tile visits that belong to a group not in our shard.
  iota = jnp.arange(num_groups, dtype=jnp.int32)
  active_group_mask = jnp.logical_and(iota <= end_group, iota >= start_group)
  group_tiles = jnp.where(active_group_mask, group_tiles, 0)
  num_tiles = group_tiles.sum()
  return (group_offsets, group_ids, m_tile_ids), num_tiles


def data_next_moe_style(block_mask, query_chunk_size, key_chunk_size, keep_in_vmem="kv"):
    if keep_in_vmem=="kv":
        k = block_mask.shape[1]
    else:
        assert keep_in_vmem=="q"
        k = block_mask.shape[0]
        block_mask = block_mask.T

    weights, selected_experts = jax.lax.top_k(block_mask, k)
    flatten_selected_experts = jnp.ravel(selected_experts)
    flatten_weights = jnp.ravel(weights)
    flatten_selected_experts = jnp.where(flatten_weights>0, flatten_selected_experts, 100000)

    sorted_selected_experts = jnp.argsort(flatten_selected_experts)
    group_sizes = jnp.bincount(flatten_selected_experts, length=k)

    selected = flatten_selected_experts[sorted_selected_experts].astype(jnp.int32)
    selecting = (jnp.arange(flatten_selected_experts.size,dtype=jnp.int32)//k)[sorted_selected_experts].astype(jnp.int32)
    
    if keep_in_vmem=="kv":
        
        kv_idxs = selected
        
        query_idxs = selecting
        num_kv_loads = group_sizes.shape[0]
        num_q_load = jnp.sum(group_sizes)
    else:
        kv_idxs = selecting
        query_idxs = selected
        num_kv_loads = group_sizes.shape[0]
        num_q_load = jnp.sum(group_sizes)
    cost = query_chunk_size*num_q_load+key_chunk_size*num_kv_loads
    mask = kv_idxs!=100000
    query_idxs = jnp.where(mask, query_idxs, 100000)
    return kv_idxs, query_idxs, group_sizes, mask, cost


def bwd_update_step(q_chunk, k_chunk, v_chunk, o_chunk, lse_chunk, do_chunk, causal_mask, scale, qk_bias, q_mask, einsum_kwargs = {}):
    """
    q_chunk: [q_chunk_size, batch_size, dim] 
    k_chunk: [k_chunk_size, batch_size, dim] 
    v_chunk: [k_chunk_size, batch_size, dim] 
    o_chunk: [q_chunk_size, batch_size, dim] 
    lse_chunk: [q_chunk_size, batch_size, 1] 
    do_chunk: [inner_bs, q_chunk_size, batch_size, dim] 
    mask: [inner_bs, q_chunk_size, batch_size, k_chunk_size] (or: broadcastable to this shape) 
    """
    attn_weights = einsum('m q d, k d -> m q k', q_chunk * scale , k_chunk, **einsum_kwargs) # [q_chunk_size, batch_size, k_chunk_size] 
    if qk_bias is not None:
        attn_weights = attn_weights + qk_bias
    if causal_mask is not None:
        attn_weights = attn_weights + jnp.where(causal_mask, 0.0, DEFAULT_MASK_VALUE)
                        
    
    p = jnp.exp(attn_weights - lse_chunk)
    p = jnp.where(q_mask, p, 0.)
    do_chunk = jnp.where(q_mask, do_chunk, 0.)
    o_chunk = jnp.where(q_mask, o_chunk, 0.)
    q_chunk = jnp.where(q_mask, q_chunk, 0.)
    
    dv_chunk = einsum('m q k, m q d  -> k d ', p, do_chunk, **einsum_kwargs) 
        
    di = jnp.sum(do_chunk * o_chunk, axis = -1, keepdims = True)
    dp = einsum('m q d, k d -> m q k', do_chunk, v_chunk, **einsum_kwargs) 

    ds = p * scale * (dp - di)
    if qk_bias is not None:
        dbias = einsum('m q k -> m', ds, precision="highest", preferred_element_type=jnp.float32) 

    else:
        dbias = None
        
    

    dq_chunk = einsum('m q k, k d -> m q d', ds, k_chunk, **einsum_kwargs) 
    dk_chunk = einsum('m q k, m q d -> k d', ds, q_chunk, **einsum_kwargs) 
    
    return dq_chunk, dk_chunk, dv_chunk, dbias/scale
    

def fwd_update_step(q_chunk, k_chunk, v_chunk, o_chunk, row_sum, row_max, causal_mask, qk_bias, q_mask, einsum_kwargs = {}):
    """
    q_chunk: [inner_chunk_bs, q_chunk_size, dim]
    k_chunk: [k_chunk_size, dim]
    """

    
    attn_weights = einsum('m q d, k d -> m q k', q_chunk, k_chunk, **einsum_kwargs) # [q_chunk_size, batch_size, k_chunk_size]
    if qk_bias is not None:
        attn_weights = attn_weights + qk_bias
    if causal_mask is not None:
        attn_weights = attn_weights + jnp.where(causal_mask, 0.0, DEFAULT_MASK_VALUE)
    
    
    block_row_max = jnp.max(attn_weights, axis = -1, keepdims = True)

    new_row_max = jnp.maximum(block_row_max, row_max)
    exp_weights = jnp.exp(attn_weights - new_row_max)
    if causal_mask is not None:
        exp_weights = jnp.where(causal_mask, exp_weights, 0.)

    block_row_sum = jnp.sum(exp_weights, axis = -1, keepdims = True)
    exp_values = einsum('m q k, k d -> m q d', exp_weights, v_chunk, **einsum_kwargs)

    exp_row_max_diff = jnp.exp(row_max - new_row_max)

    new_row_sum = exp_row_max_diff*row_sum + block_row_sum
    updated_out = exp_row_max_diff*o_chunk + exp_values   
     
    
    

    # new_row_max = jnp.where(q_mask, new_row_max, row_max)
    # new_row_sum = jnp.where(q_mask, new_row_sum, row_sum)
    # updated_out = jnp.where(q_mask, updated_out, o_chunk)
    return updated_out, new_row_sum, new_row_max



    
from EasyLM import jax_utils


def create_nsa_metadata(block_mask, q_chunk_size, k_chunk_size, inner_chunk_bs, visit_empty_groups=False, keep_in_vmem="kv"):
    data_next = data_next_moe_style(
        block_mask, q_chunk_size, k_chunk_size, keep_in_vmem=keep_in_vmem
    )
    key_idxs_sorted, query_idxs_sorted, group_sizes, mask_sorted, cost = data_next

    num_queries_times_keys = query_idxs_sorted.shape[0]
    padded_num_queries_times_keys = ((num_queries_times_keys + inner_chunk_bs - 1) // inner_chunk_bs) * inner_chunk_bs
    padding_needed = padded_num_queries_times_keys - num_queries_times_keys
    key_idxs_sorted = jnp.pad(key_idxs_sorted, (0, padding_needed), mode='constant', constant_values=100000)
    query_idxs_sorted = jnp.pad(query_idxs_sorted, (0, padding_needed), mode='constant', constant_values=100000)
    mask_sorted = jnp.pad(mask_sorted, (0, padding_needed), mode='constant', constant_values=False)

    group_metadata, num_active_tiles = make_group_metadata(
            group_sizes=group_sizes,
            m=padded_num_queries_times_keys,
            tm=inner_chunk_bs,
            start_group=jnp.array(0, dtype=jnp.int32),
            num_nonzero_groups=group_sizes.shape[0],
            visit_empty_groups=visit_empty_groups
        )
    
    return group_metadata, num_active_tiles, query_idxs_sorted, mask_sorted

from EasyLM.jax_utils import always_debug_print

def maybe_apply_causal_mask(queries_for_key, causal, q_chunk_size, key_id, k_chunk_size, q_block_idx, k_block_idx):
    """
    queries_for_key has shape [inner_chunk_bs]
    mask_for_key has shape [inner_chunk_bs]
    causal is boolean
    q_chunk_size is int
    k_chunk_size is int
    key_id is int
    """
    expanded_mask = None
    if causal:
        q_pos = (q_chunk_size * queries_for_key[:, None] + jnp.arange(q_chunk_size)[None, :])[:,:,None]
        # q_pos = q_pos + q_chunk_size*q_block_idx
        k_pos = (k_chunk_size * key_id + jnp.arange(k_chunk_size)[None, :]).reshape(1,1, -1)
        # k_pos = k_pos + k_chunk_size*k_block_idx
        expanded_mask = (q_pos >= k_pos) 
    return expanded_mask

@partial(jax.vmap,        in_axes=(0, 0,  0, None, None,                             None,         None,       0,        0,            0,          None,           None,              None,        None,           None,      None))
def _sparse_blockwise_attention_fwd(q, k, v, q_block_idx, k_block_idx, block_mask, block_bias, numerator, denominator, max_score, nsa_metadata, inner_chunk_bs, q_chunk_size, k_chunk_size, disable_caching, causal):
    # q = [n_q_chunks, q_chunk_size,  dim] 
    # k = [n_k_chunks, k_chunk_size,  dim] 
    # v = [n_k_chunks, k_chunk_size,  dim] 
    # block_mask = [n_q_chunks, n_k_chunks]
    # block_bias = [n_q_chunks, n_k_chunks]
    # numerator = [n_q_chunks, q_chunk_size, dim]
    # denominator = [n_q_chunks, q_chunk_size, 1]
    # max_score = [n_q_chunks, q_chunk_size, 1]
    if block_bias is not None:
        assert block_bias.shape[0]==block_mask.shape[0]
        assert block_bias.shape[1]==block_mask.shape[1]
    dim = q.shape[-1]
    (group_offsets, group_ids, m_tile_ids), num_active_tiles, query_idxs_sorted, mask_sorted = nsa_metadata
    # nsa_metadata:
    # group_offsets = [num_groups+1]
    # group_ids = [num_active_tiles]
    # m_tile_ids = [num_active_tiles]
    # num_active_tiles = int
    # query_idxs_sorted = [padded_num_queries_times_keys]
    # mask_sorted = [padded_num_queries_times_keys]

    scale = 1 / jnp.sqrt(dim)
    q_scaled = q * scale  

    def lookup_q(m_index, carry):
        _, (carried_row_max, carried_row_sum, carried_out, _), prev_obj, _ = carry
        def false_cond():
            queries_for_key = gather_slice(query_idxs_sorted, m_index, inner_chunk_bs, fill_value=100000, mult=True)
            mask_for_key = gather_slice(mask_sorted, m_index, inner_chunk_bs, fill_value=False, mult=True)
            # Extract the relevant queries and key/value vectors
            queries_for_key = jnp.where(mask_for_key, queries_for_key, 100000)
            q_chunk = q_scaled.at[queries_for_key].get(mode='fill')  # [inner_chunk_bs, q_chunk_size, dim]
            o_chunk = carried_out.at[queries_for_key].get(mode='fill') # [inner_chunk_bs, q_chunk_size, dim]
            row_max = carried_row_max.at[queries_for_key].get(mode='fill') # [inner_chunk_bs, q_chunk_size, 1]
            row_sum = carried_row_sum.at[queries_for_key].get(mode='fill') # [inner_chunk_bs, q_chunk_size, 1]
            
            return queries_for_key,  mask_for_key, q_chunk, o_chunk, row_max, row_sum

        if prev_obj is None or disable_caching:
            return false_cond()
        else:
            _, q_res, _, prev_m_index = prev_obj
            def true_cond():
                return q_res
            return jax.lax.cond(m_index==prev_m_index, true_cond, false_cond)
        
    def lookup_kv(key_id, carry):
        _,_,prev_obj, _ = carry
        def false_cond():
            k_chunk = k[key_id] # [key_chunk_size,  dim] 
            v_chunk = v[key_id] # [key_chunk_size,  dim] 
            group_start = group_offsets[key_id]
            group_end = group_offsets[key_id + 1]
            if block_bias is not None:
                k_bias = block_bias[:,key_id]
            else:
                k_bias =  None
            return group_start, group_end, k_chunk, v_chunk, k_bias
        if prev_obj is None or disable_caching:
            return false_cond()
        else:
            kv_res, _, prev_key_id, _ = prev_obj
            def true_cond():
                return kv_res
            return jax.lax.cond(key_id==prev_key_id, true_cond, false_cond)
    
    def key_loop_body(carry):
        grid_id, (carried_row_max, carried_row_sum, carried_out, num_active_tiles), prev_obj, next_obj = carry
        if (next_obj is None) or disable_caching:
            key_id = group_ids[grid_id]
            m_index = m_tile_ids[grid_id]
            kv_res, q_res = lookup_kv(key_id, carry), lookup_q(m_index, carry)
        else:
            kv_res, key_id, q_res, m_index = next_obj
        
        group_start, group_end, k_chunk, v_chunk, k_bias = kv_res
        queries_for_key,  mask_for_key, q_chunk, o_chunk, row_max, row_sum = q_res
        if k_bias is not None:
            qk_bias = k_bias.at[queries_for_key].get(mode='fill')
            qk_bias = qk_bias[:, None, None]
            
        else:
            qk_bias = None
        iota = inner_chunk_bs * m_index + jnp.arange(inner_chunk_bs)
        group_mask = jnp.logical_and(iota >= group_start, iota < group_end)
        q_mask = group_mask & mask_for_key
        masked_queries_for_key = jnp.where(q_mask, queries_for_key, 100000)
        
        causal_mask = maybe_apply_causal_mask(queries_for_key, causal, q_chunk_size, key_id, k_chunk_size, q_block_idx, k_block_idx)

        
        updated_out, new_row_sum, new_row_max = fwd_update_step(q_chunk, k_chunk, v_chunk,  
                                                                o_chunk, row_sum, row_max, causal_mask, qk_bias, q_mask[:, None, None])

        # Update accumulators
        carried_out = carried_out.at[masked_queries_for_key].set(updated_out, mode='fill')
        carried_row_max = carried_row_max.at[masked_queries_for_key].set(new_row_max, mode='fill')
        carried_row_sum = carried_row_sum.at[masked_queries_for_key].set(new_row_sum, mode='fill')

        updated_out = carried_out.at[queries_for_key].get(mode='fill') # [inner_chunk_bs, q_chunk_size, dim]
        new_row_max = carried_row_max.at[queries_for_key].get(mode='fill') # [inner_chunk_bs, q_chunk_size, 1]
        new_row_sum = carried_row_sum.at[queries_for_key].get(mode='fill') # [inner_chunk_bs, q_chunk_size, 1]
        
        q_res = queries_for_key,  mask_for_key, q_chunk, updated_out, new_row_max, new_row_sum
        
        prev_obj = kv_res, q_res, key_id, m_index
        carried_el = (carried_row_max, carried_row_sum, carried_out, num_active_tiles)
        
        carry = grid_id, carried_el, prev_obj, next_obj

        def load_next():
            next_key_id = group_ids[grid_id+1]
            next_m_index = m_tile_ids[grid_id+1]
            next_kv_res, next_q_res = lookup_kv(next_key_id, carry), lookup_q(next_m_index, carry)
            return grid_id+1, carried_el, prev_obj, (next_kv_res, next_key_id, next_q_res, next_m_index)
        
        def no_next():
            return grid_id+1, carried_el, prev_obj, next_obj

        if next_obj is None:
            return load_next()
        else:
            return jax.lax.cond(grid_id+1 < num_active_tiles, load_next, no_next)
    
    def cond_func(carry):
        grid_id, ( _, _, _, num_active_tiles), _, _ = carry
        return grid_id < num_active_tiles
    
    grid_id = 0
    carry = grid_id, (max_score, denominator, numerator, num_active_tiles), None, None

    carry = key_loop_body(carry)
    
    carry = lax.while_loop(cond_func, key_loop_body, carry)
    
    _,(max_score, denominator, numerator, _), _, _ = carry


    return numerator, denominator, max_score

# 
@partial(jax.vmap,        in_axes=(0,  0, 0, None,                          None,   0,  0, 0,  0,    0,         None,         None,      0,        0,       0,         None,               None,        None,           None,      None, None))
def _sparse_blockwise_attention_bwd(q, k, v,  q_block_idx, k_block_idx, do, dq, dk, dv, dblock_bias, block_mask, block_bias, out, denominator, max_score, nsa_metadata ,inner_chunk_bs, q_chunk_size, k_chunk_size, causal, disable_caching):
    dim = q.shape[-1]
    scale = 1 / jnp.sqrt(dim)
    lse = jnp.log(denominator) + max_score

    # Rearrange tensors as in the forward pass
    o = rearrange(out, '(q c) d -> q c d', c=q_chunk_size) 
    do = rearrange(do, '(q c) d -> q c d', c=q_chunk_size) 
    # Load nsa_metadata
    del block_mask
    (group_offsets, group_ids, m_tile_ids), num_active_tiles, query_idxs_sorted, mask_sorted = nsa_metadata

    # Initialize loop variables
    grid_id = 0
    
    def lookup_bwd_q(m_index, carry):
        _, prev_obj = carry

        def false_cond():
            queries_for_key = gather_slice(query_idxs_sorted, m_index, inner_chunk_bs, fill_value=100000, mult=True)
            mask_for_key = gather_slice(mask_sorted, m_index, inner_chunk_bs, fill_value=False, mult=True)

            # Extract relevant chunks
            q_chunk = q.at[queries_for_key].get(mode='fill')
            o_chunk = o.at[queries_for_key].get(mode='fill')
            do_chunk = do.at[queries_for_key].get(mode='fill')
            lse_chunk = lse.at[queries_for_key].get(mode='fill')
            return queries_for_key, mask_for_key, q_chunk, o_chunk, do_chunk, lse_chunk

        if prev_obj is None or disable_caching:
            return false_cond()
        else:
            _, q_res, _, prev_m_index = prev_obj

            def true_cond():
                return q_res

            return jax.lax.cond(m_index == prev_m_index, true_cond, false_cond)
        
    def lookup_bwd_kv(key_id, carry):
        _, prev_obj = carry

        def false_cond():
            k_chunk = k[key_id]
            v_chunk = v[key_id]
            group_start = group_offsets[key_id]
            group_end = group_offsets[key_id + 1]
            if block_bias is not None:
                k_bias = block_bias[:,key_id]
            else:
                k_bias =  None
            return group_start, group_end, k_chunk, v_chunk, k_bias

        if prev_obj is None or disable_caching:
            return false_cond()
        else:
            kv_res, _, prev_key_id, _ = prev_obj

            def true_cond():
                return kv_res

            return jax.lax.cond(key_id == prev_key_id, true_cond, false_cond)
        
    def key_loop_body(carry):
        (grid_id, dq, dk, dv, dblock_bias, num_active_tiles), prev_obj = carry

        key_id = group_ids[grid_id]
        m_index = m_tile_ids[grid_id]
        
        # Fetch or reuse k and v chunks
        kv_res = lookup_bwd_kv(key_id, carry)
        group_start, group_end, k_chunk, v_chunk, k_bias = kv_res
        q_res = lookup_bwd_q(m_index, carry)
        queries_for_key, mask_for_key, q_chunk, o_chunk, do_chunk, lse_chunk = q_res
        if k_bias is not None:
            qk_bias = k_bias.at[queries_for_key].get(mode='fill')
            qk_bias = qk_bias[:, None,  None] 
            
        else:
            qk_bias = None
        # Fetch or reuse q-related tensors

        iota = inner_chunk_bs * m_index + jnp.arange(inner_chunk_bs)
        group_mask = jnp.logical_and(iota >= group_start, iota < group_end)

        q_mask = group_mask & mask_for_key
        causal_mask = maybe_apply_causal_mask(queries_for_key, causal, q_chunk_size, key_id, k_chunk_size, q_block_idx, k_block_idx)
        
        
        # Compute the backward updates
        dq_chunk, dk_chunk, dv_chunk, dbias = bwd_update_step(
            q_chunk, k_chunk, v_chunk, o_chunk, lse_chunk, do_chunk, causal_mask, scale, qk_bias, q_mask[:, None,  None] 
        )
        # Accumulate the gradients
        masked_queries_for_key = jnp.where(q_mask, queries_for_key, 100000)
        dq = dq.at[masked_queries_for_key].add(dq_chunk, mode='fill')

        dk = dk.at[key_id].add(dk_chunk)
        dv = dv.at[key_id].add(dv_chunk)

        if dblock_bias is not None:
            dblock_bias = dblock_bias.at[masked_queries_for_key, key_id].add(dbias, mode='fill')
        
        q_res = queries_for_key, mask_for_key, q_chunk, o_chunk, do_chunk, lse_chunk
        prev_obj = (kv_res, q_res, key_id, m_index)
        

        return (grid_id + 1, dq, dk, dv, dblock_bias, num_active_tiles), prev_obj
    def cond_func(carry):
        (grid_id, _, _, _, _, num_active_tiles), _ = carry
        return grid_id < num_active_tiles
    carry = (grid_id, dq, dk, dv, dblock_bias, num_active_tiles), None
    carry = key_loop_body(carry)
    
    carry = lax.while_loop(cond_func, key_loop_body, carry)
    (_, dq, dk, dv, dblock_bias, _), _ = carry

    # Rearrange gradients back to original format
    dq = rearrange(dq, 'q c d -> (q c) d') 
    dk = rearrange(dk, 'k c d -> (k c) d') 
    dv = rearrange(dv, 'k c d -> (k c) d') 
    return dq, dk, dv, None, dblock_bias

def create_single_device_nsa_attention(q_chunk_size=128, k_chunk_size=128, inner_chunk_bs=512, causal=False, disable_caching=False):
    
    def _single_nsa_attention(q, k, v, block_mask, block_bias):
        heads, q_len, dim = q.shape   
        n_q_chunks = q_len // q_chunk_size
        max_score = jnp.full((heads, n_q_chunks, q_chunk_size, 1), -jnp.inf)  # [n_q_chunks, q_chunk_size, 1]  
        denominator = jnp.zeros((heads, n_q_chunks, q_chunk_size, 1))  # [n_q_chunks, q_chunk_size, 1]  
        numerator = jnp.zeros((heads, n_q_chunks, q_chunk_size, dim), dtype=q.dtype)  # [n_q_chunks, q_chunk_size, dim]  
        
        q = rearrange(q, 'h (q c) d -> h q c d' ,c=q_chunk_size)  # [n_q_chunks, q_chunk_size, dim]  
        k = rearrange(k, 'h (k c) d -> h k c d', c=k_chunk_size)  # [n_k_chunks, k_chunk_size, dim]  
        v = rearrange(v, 'h (k c) d -> h k c d', c=k_chunk_size)  # [n_k_chunks, k_chunk_size, dim]  
        nsa_metadata = create_nsa_metadata(block_mask, visit_empty_groups=False, 
                                                inner_chunk_bs=inner_chunk_bs, q_chunk_size=q_chunk_size, k_chunk_size=k_chunk_size)
        q_block_idx=0
        k_block_idx=0
        numerator, denominator, max_score = _sparse_blockwise_attention_fwd(q, k, v, q_block_idx, k_block_idx, block_mask, block_bias, numerator, denominator, max_score, nsa_metadata, inner_chunk_bs, 
                                                                   q_chunk_size, k_chunk_size, disable_caching, causal)
        
        # Rearrange the output tensor back to [q_len, dim] 
        output = numerator / denominator
        output = rearrange(output, 'h q c d ->  h (q c) d') 
        return output, (denominator, max_score, q,k,v)
    

    @jax.jit
    def single_nsa_attention_forward(q, k, v, block_mask, block_bias=None):
        out, (denominator, max_score, q,k,v) = _single_nsa_attention(q, k, v, block_mask, block_bias)
        return out, (q, k, v, block_mask, block_bias, out, denominator, max_score)

    @jax.jit
    def single_nsa_attention_backward(res, do):
        q, k, v, block_mask, block_bias, out, denominator, max_score = res
        dq, dk, dv = map(lambda x:jnp.zeros_like(x,dtype=jnp.float32), [q, k, v])
        if block_bias is not None:
            dblock_bias = jnp.zeros((q.shape[0],)+block_bias.shape)
        else:
            dblock_bias = None
        nsa_metadata = create_nsa_metadata(block_mask, visit_empty_groups=True, inner_chunk_bs=inner_chunk_bs, q_chunk_size=q_chunk_size, k_chunk_size=k_chunk_size)
        q_block_idx=0
        k_block_idx=0
        dq, dk, dv, _, dblock_bias =  _sparse_blockwise_attention_bwd(q, k, v, q_block_idx, k_block_idx, do, dq, dk, dv, dblock_bias, block_mask, block_bias, out, denominator, max_score,
                                               nsa_metadata, inner_chunk_bs, q_chunk_size, k_chunk_size, causal, disable_caching)
        if dblock_bias is not None:
            dblock_bias = dblock_bias.sum(0)
        return dq, dk, dv, None, dblock_bias
    
    @jax.custom_vjp
    @jax.jit
    def single_nsa_attention(q, k, v, block_mask, block_bias=None):
        
        out, _ = _single_nsa_attention(q, k, v, block_mask, block_bias)
        return out


    single_nsa_attention.defvjp(single_nsa_attention_forward, single_nsa_attention_backward)
    return single_nsa_attention





def create_multi_device_nsa_attention(axis_name, q_chunk_size=128, k_chunk_size=128, inner_chunk_bs=512, causal=False, disable_caching=False):

    def _multi_nsa_attention(q, k, v, block_mask, block_bias=None):
        heads, q_len, dim = q.shape   
        n_q_chunks = q_len // q_chunk_size
        # n_k_chunks = q_len // k_chunk_size
        max_score = jnp.full((heads, n_q_chunks, q_chunk_size, 1), -jnp.inf, dtype=q.dtype)  # [n_q_chunks, q_chunk_size, 1]  
        denominator = jnp.zeros((heads, n_q_chunks, q_chunk_size, 1), dtype=q.dtype)  # [n_q_chunks, q_chunk_size, 1]  
        numerator = jnp.zeros((heads, n_q_chunks, q_chunk_size, dim), dtype=q.dtype)  # [n_q_chunks, q_chunk_size, dim]  
        
        q = rearrange(q, 'h (q c) d -> h q c d' ,c=q_chunk_size)  # [n_q_chunks, q_chunk_size, dim]  
        k = rearrange(k, 'h (k c) d -> h k c d', c=k_chunk_size)  # [n_k_chunks, k_chunk_size, dim]  
        v = rearrange(v, 'h (k c) d -> h k c d', c=k_chunk_size)  # [n_k_chunks, k_chunk_size, dim]  

        axis_size = lax.psum(1, axis_name)
        q_block_idx = lax.axis_index(axis_name)
        # block_mask has shape [n_q_chunks, axis_size*n_k_chunks], we reshape it to [axis_size, n_q_chunks, n_k_chunks]
        block_mask = rearrange(block_mask, 'q (a k) -> a q k', a=axis_size)
        block_bias = rearrange(block_bias, 'q (a k) -> a q k', a=axis_size) if block_bias is not None else None
        
        def shift_right(x):
            return lax.ppermute(x, axis_name, perm=[(i, (i + 1) % axis_size) for i in range(axis_size)])

        
        def scan_kv_block(carry, _):
            max_score, numerator, denominator, k, v, k_block_idx = carry
            block_mask_chunk = block_mask[k_block_idx]
            block_bias_chunk = block_bias[k_block_idx] if block_bias is not None else None
            nsa_metadata = create_nsa_metadata(block_mask_chunk, visit_empty_groups=False, 
                                                inner_chunk_bs=inner_chunk_bs, q_chunk_size=q_chunk_size, k_chunk_size=k_chunk_size)
            numerator, denominator, max_score = _sparse_blockwise_attention_fwd(q, k, v, q_block_idx, k_block_idx, block_mask_chunk, block_bias_chunk, numerator, denominator, max_score, nsa_metadata, inner_chunk_bs, 
                                                                   q_chunk_size, k_chunk_size, disable_caching, causal)
            k, v, k_block_idx = shift_right((k, v, k_block_idx))
            return (max_score, numerator, denominator, k, v, k_block_idx), None
        

        (max_score, numerator, denominator, _, _, _), _ = lax.scan(scan_kv_block,
            init=(max_score, numerator, denominator, k, v, q_block_idx), xs=jnp.arange(0, axis_size))
        output = numerator / denominator
        output = rearrange(output, 'h q c d ->  h (q c) d') 
        return output, (block_mask, block_bias, denominator, max_score, q,k,v)

    
    @jax.jit
    def multi_nsa_attention_forward(q, k, v, block_mask, block_bias=None):
        out, (block_mask, block_bias, denominator, max_score, q,k,v) = _multi_nsa_attention(q, k, v, block_mask, block_bias)
        return out, (q, k, v, block_mask, block_bias, out, denominator, max_score)

    def multi_nsa_attention_backward(res, do):
        q, k, v, block_mask, block_bias, out, denominator, max_score = res
        n_heads = q.shape[0]
        dq, dk, dv = map(lambda x:jnp.zeros_like(x,dtype=jnp.float32), [q, k, v])
        if block_bias is not None:
            dblock_bias = jnp.zeros(block_bias.shape)
        else:
            dblock_bias = None
        axis_size = lax.psum(1, axis_name)
        q_block_idx = lax.axis_index(axis_name)
        
        def shift_right(x):
            return lax.ppermute(x, axis_name, perm=[(i, (i + 1) % axis_size) for i in range(axis_size)])

        
        def scan_kv_block(carry, _):
            dq, dk, dv, dblock_bias, k, v, k_block_idx = carry        
            block_mask_chunk = block_mask[k_block_idx]
            
            if block_bias is not None:
                block_bias_chunk = block_bias[k_block_idx]
                dblock_bias_chunk = jnp.zeros((n_heads,)+block_bias_chunk.shape, dtype=jnp.float32)
            else:
                block_bias_chunk = None
                dblock_bias_chunk = None
                
            nsa_metadata = create_nsa_metadata(block_mask_chunk, visit_empty_groups=True, inner_chunk_bs=inner_chunk_bs, q_chunk_size=q_chunk_size, k_chunk_size=k_chunk_size)
            dq, dk, dv, _, dblock_bias_chunk =  _sparse_blockwise_attention_bwd(q, k, v, q_block_idx, k_block_idx, do, dq, dk, dv, dblock_bias, block_mask_chunk, block_bias_chunk, out, denominator, max_score,
                                               nsa_metadata, inner_chunk_bs, q_chunk_size, k_chunk_size, causal, disable_caching)
            k, v, dk, dv, k_block_idx = shift_right((k, v, dk, dv, k_block_idx))
            if block_bias is not None:
                dblock_bias = dblock_bias.at[k_block_idx].set(dblock_bias_chunk.sum(0))
                
            return (dq, dk, dv, dblock_bias, k, v, k_block_idx), None
        (dq, dk, dv, dblock_bias, k, v, _), _ = lax.scan(scan_kv_block, init=(dq, dk, dv, dblock_bias, k, v, q_block_idx), xs=jnp.arange(0, axis_size))
        if dblock_bias is not None:
            dblock_bias = rearrange(dblock_bias, 'a q k -> q (a k)')
        return dq, dk, dv, None, dblock_bias

    @jax.custom_vjp
    def multi_nsa_attention(q, k, v, block_mask, block_bias=None):
        out, _ = _multi_nsa_attention(q, k, v, block_mask, block_bias)
        return out

    multi_nsa_attention.defvjp(multi_nsa_attention_forward, multi_nsa_attention_backward)
    return multi_nsa_attention
