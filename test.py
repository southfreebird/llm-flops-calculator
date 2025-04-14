
import jax
import kvax
    
mesh = 
attention_specs = 
flash_attention = 
mesh_devices = 




# Attention specs
# query: (batch, query_sequence, heads, attention_head_dim)
# kv: (batch, kv_sequence, kv_heads, attention_head_dim)
      
# Context Parallelism
mesh = jax.sharding.Mesh(mesh_devices, ("data", "context"))
with mesh, attention_specs(
    query_specs=("data", "context", None, None),
    kv_specs=("data", None, None, None),
):
    attn_out = kvax.ops.flash_attention(...)

# Tensor Parallelism
mesh = jax.sharding.Mesh(mesh_devices, ("data", "model"))
with mesh, attention_specs(
    query_specs=("data", None, "model", None),
    kv_specs=("data", None, "model", None),
):
    attn_out = kvax.ops.flash_attention(...)



