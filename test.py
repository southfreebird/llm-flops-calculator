

mesh = 
attention_specs = 
flash_attention = 



# Attention specs
# query: (batch, query_sequence, heads, attention_head_dim)
# kv: (batch, kv_sequence, kv_heads, attention_head_dim)
      
# Context Parallelism
with mesh, attention_specs(
    query_specs=("data", "context", None, None),
    kv_specs=("data", None, None, None),
):
    attn_out = flash_attention(...)

# Tensor Parallelism
with mesh, attention_specs(
    query_specs=("data", None, "model", None),
    kv_specs=("data", None, "model", None),
):
    attn_out = flash_attention(...)



