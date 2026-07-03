# Value-Scaled Attention Rollout

Value-Scaled Attention Rollout scales attention weights by the norm of the corresponding value vectors to avoid tracking irrelevant high-weight paths.

### Detailed Concept
Standard rollout only considers query-key interactions ($A$). However, if a value vector $v_j$ has a magnitude near zero, no information is transferred regardless of the attention weight. Value-scaled rollout multiplies attention maps by the norm of the value vector:
$$A_{\text{scaled}} = A \cdot \|V\|$$

- **Pros:** Eliminates false-positive attributions.
- **Cons:** Requires computing value vector norms.

### Diagram
```mermaid
flowchart TD
    Attn[Attention Matrix A] --> Mult[Multiply Element-wise]
    Val[Value Vector Norm ||V||] --> Mult
    Mult --> ScaledAttn[Value-Scaled Attention]
```
