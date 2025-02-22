
import jax
from jax import numpy as jnp
from einops import rearrange
from einops import repeat
import numpy as np
import jax.numpy as jnp
import pandas as pd

from flax.linen import dot_product_attention as flax_dot_product_attention

from src.nsa_attention_jax import create_single_device_nsa_attention as og_create_nsa_attention

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
import optax
def create_nsa_attention(q_chunk_size, k_chunk_size,inner_chunk_bs,causal):
    nsa_attention = og_create_nsa_attention(q_chunk_size, k_chunk_size, inner_chunk_bs=inner_chunk_bs, causal=causal)
    def _nsa_attention(xq, xk, xv, block_mask, block_bias):
        xq, xk, xv = map(lambda x: x[0], [xq, xk, xv]) # remove batch dim
        sa_out = nsa_attention(xq, xk, xv, block_mask, block_bias)
        return jax.device_get(sa_out[None,...]) 
    return _nsa_attention

def create_dot_product(q_chunk_size, k_chunk_size, causal=False):
    def _dot_product_attention(xq, xk, xv, block_mask, block_bias):
        ref_xq = jnp.swapaxes(xq,-2,-3).astype(jnp.float32)
        ref_xk = jnp.swapaxes(xk,-2,-3).astype(jnp.float32)
        ref_xv = jnp.swapaxes(xv,-2,-3).astype(jnp.float32)
        if block_bias is None:
            block_bias = 0
        block_attention_bias = jnp.where(block_mask, block_bias, DEFAULT_MASK_VALUE)
        
        attention_bias = repeat(block_attention_bias, 'q k -> (q nq) (k nk)', nk=k_chunk_size, nq=q_chunk_size)
        if causal:
            q_pos = jnp.arange(attention_bias.shape[0])[:,None]
            k_pos = jnp.arange(attention_bias.shape[1])[None,:]
            causal_mask = q_pos >= k_pos
            attention_bias = jnp.where(causal_mask, attention_bias, DEFAULT_MASK_VALUE)

        
            
        out = flax_dot_product_attention(ref_xq, ref_xk, ref_xv, bias=attention_bias[None,None,:,:], 
                                         force_fp32_for_softmax=True, 
                                         deterministic=True, 
                                        # precision="high", 
                                        dtype=jnp.float32
                                         )
        out = rearrange(out, 'b n h d -> b h n d')
        return jax.device_get(out)
    return _dot_product_attention







def grad_fn(func):
    def loss_fn(*args):
        return jnp.sum(func(*args))
    return jax.grad(loss_fn, argnums=(0, 1, 2, 4))


def eval_func(a,b, atol=0.001):
    if a is None or b is None:
        return None
    return np.isclose(a,b, atol=atol,rtol=0.0001).mean()


def flatten_nonzero(x):
    x = x.flatten()
    return x[x!=0]



def evaluate(xq, xk, xv, block_mask, block_bias, q_chunk_size, k_chunk_size, inner_chunk_bs, n_k_chunks, n_q_chunks, causal, verbose=False):
    dot_product_attention = create_dot_product(q_chunk_size, k_chunk_size, causal)
    nsa_attention = create_nsa_attention(q_chunk_size, k_chunk_size, inner_chunk_bs, causal)


        
    fa1_out = dot_product_attention(xq, xk, xv, block_mask, block_bias)
    dq_fa1, dk_fa1, dv_fa1, dbias_fa1 = grad_fn(dot_product_attention)(xq, xk, xv, block_mask, block_bias)
    try:
        sa_out = nsa_attention(xq, xk, xv, block_mask, block_bias)
        dq_sa, dk_sa, dv_sa, dbias_sa = grad_fn(nsa_attention)(xq, xk, xv, block_mask, block_bias)
        element = dict(n_k_chunks=n_k_chunks, n_q_chunks=n_q_chunks, inner_chunk_bs=inner_chunk_bs
                        )
        are_nz = ((dbias_sa!=0) & (dbias_fa1!=0)).flatten()
        dbias_sa = dbias_sa.flatten()[are_nz]
        dbias_fa1 = dbias_fa1.flatten()[are_nz]
        isclose = np.isclose(dbias_sa, dbias_fa1, atol=0.0001,rtol=0.0001)
        if verbose:
            print(f"{dbias_sa=}")
            print(f"{dbias_fa1=}")
            print(isclose)
        # print(f"{sa_out[...,0]=}")
        # print(f"{fa1_out[...,0]=}")
        for a,b,name in zip([dq_fa1, dk_fa1, dv_fa1, dbias_fa1, fa1_out],
                            [dq_sa, dk_sa, dv_sa, dbias_sa, sa_out],
                            [f"query_grad", f"key_grad", f"value_grad", "bias_grad", "similarity"]):
            atol = 0.1 if name == "bias_grad" else 0.001
            element[name] = eval_func(a,b, atol=atol)
        print(element)
    except Exception as e:
        raise e
    return element

def test_implementation_jit(NKC=[2,4,6,8,16],NQC=[2,4,6,8,16], NCB=[2, 4, 8], batch = 1,
                        heads= 4, q_chunk_size=128, k_chunk_size=512, head_dim = 128,
                        rand=True,
                        causal=False,
                        use_bias=False,
                        all_ones=False,
                        square=False,
                        ):
    test_results = []
    for n_k_chunks in NKC:
        for n_q_chunks in NQC:
            if causal and n_q_chunks!=n_k_chunks and square:
                continue
            # if n_q_chunks*q_chunk_size<n_k_chunks*k_chunk_size:
            #     print(f"skipping {n_q_chunks=}, {n_k_chunks=}")
            #     continue
            rng = np.random.RandomState(42)
            q_len = q_chunk_size*n_q_chunks
            k_len = k_chunk_size*n_k_chunks
            xq = rng.randn(batch, heads, q_len, head_dim)
            xk = rng.randn(batch, heads, k_len, head_dim)
            xv = rng.randn(batch, heads, k_len, head_dim)
            o_block_mask = np.ones((q_len//q_chunk_size,k_len//k_chunk_size))
            block_mask = o_block_mask.copy()
            c = 1
            # block_bias =  1+rng.randn(*block_mask.shape)*c if use_bias else None
            block_bias =  3*(rng.randn(*block_mask.shape) < 0.5).astype(float) if use_bias else None
            # xq,xk,xv = map(lambda x: 5*x, (xq,xk,xv))
            
            block_mask[1,1]=False
            if rand:
                block_mask = rng.randn(*block_mask.shape) < 0.5
                # block_mask = rng.randn(*block_mask.shape) < 0.5
                block_mask[:,0] = True
            if all_ones:
                block_mask = np.ones_like(block_mask)
            
            
            block_mask = jax.device_put(block_mask).astype(bool)
            xq,xk,xv = jax.device_put((xq,xk,xv))
            for inner_chunk_bs in NCB:
                element = evaluate(xq, xk, xv, block_mask, block_bias, q_chunk_size, k_chunk_size, inner_chunk_bs, n_k_chunks, n_q_chunks, causal)
                test_results.append(element)
    return pd.DataFrame(test_results)

def parse(s):
    l = map(int, s.split(','))
    l = list(l)
    l = [[x] for x in l]
    return dict(NKC=l[0], NQC=l[1], NCB=l[2])

# heads=1, head_dim=128, q_chunk_size=4, k_chunk_size=256, use_bias=True, causal=False works
# heads=1, head_dim=1, q_chunk_size=4, k_chunk_size=256, use_bias=True, causal=True works
# heads=1, head_dim=1, q_chunk_size=4, k_chunk_size=8, use_bias=True, causal=(False,True) works
# heads=4, head_dim=2, q_chunk_size=2, k_chunk_size=4 doesnt work
# heads=2, head_dim=128, q_chunk_size=4, k_chunk_size=256, use_bias=True doesnt work
t_results = test_implementation_jit(
                                # NKC=[4], NQC=[16], NCB=[2],
                                NKC=[2,4,6,8,16,32,64],NQC=[2,4,6,8,16], NCB=[2, 4, 8],
                                # all_ones=True,
                                # **parse("8,4,2"), all_ones=True, heads=1, head_dim=2, q_chunk_size=2, k_chunk_size=4, use_bias=True, causal=True
                                # **parse("16,4,4"), all_ones=True, heads=1, head_dim=2, q_chunk_size=2, k_chunk_size=4, use_bias=True, causal=True
                                heads=4, head_dim=64, q_chunk_size=64, k_chunk_size=256, use_bias=True, causal=True
                                
                                
                                
                                # heads=1, head_dim=128, q_chunk_size=4, k_chunk_size=256, use_bias=True, causal=True
                                # heads=1, head_dim=128, q_chunk_size=4, k_chunk_size=256, use_bias=True, causal=False
                                # heads=1, head_dim=128, q_chunk_size=4, k_chunk_size=256, use_bias=True, causal=False,
                                # heads=1, head_dim=128, q_chunk_size=4, k_chunk_size=8, use_bias=True, causal=False
                                # heads=1, head_dim=128, q_chunk_size=4, k_chunk_size=256, use_bias=True, causal=False
                                # heads=1, head_dim=128, q_chunk_size=4, k_chunk_size=256, use_bias=True, causal=False
                                # heads=1, head_dim=128, q_chunk_size=4, k_chunk_size=256, use_bias=True, causal=False
                                )
# t_results