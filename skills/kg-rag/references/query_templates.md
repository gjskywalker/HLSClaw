## Query templates for Vitis-user-guide RAG

### Intent
Use KG-RAG to answer a design-specific HLS question, not a generic optimization tutorial.
Build the query from:
- the current software-validated design in `ScratchpadInfo/Current_Code`
- target board, clock, and top function
- raw Vitis diagnostics verbatim when available

### Hard rules
- Do not ask broad questions like "How to optimize nested loops in HLS?"
- Do not invent pragma legality rules from memory. Ask KG-RAG for the exact legal pattern relevant to the observed code and error.
- If a raw Vitis error exists, include the exact error string in the query.
- Tie the query to the actual code shape: affected function, loop nest, interface/memory pattern, and intended pragma.
- Ask for legal HLS code patterns, root cause, and 2-4 fix options with tradeoffs.
- Prefer one focused query over multiple unrelated queries.

### Required input frame
Summarize these fields before forming the question:
- Design: current software-validated kernel/function structure
- Bottleneck: latency / II / timing / memory / dataflow / pragma-placement
- Raw Vitis diagnostics: exact error or warning lines if any
- Decision needed: what the next hardware rewrite or tuning pass must change

### Canonical question template
Use this structure when possible:

"For the current HLS design with top function `<top_func>` targeting `<part>` at `<freq>` MHz, the software-validated code currently has `<code_shape_summary>`. Vitis reports `<raw_error_text>` while attempting `<hardware_intent>`. Based on Vitis HLS guidance, what is the root cause, what are the legal code/pragma placement patterns for this case, and what 2-4 concrete hardware rewrite options should be tried next? Include relevant tradeoffs for latency, II, timing, memory bandwidth, and resource usage."

### Error-focused variants
- Pragma placement / scope:
  - "For the current HLS design, Vitis reports `<raw_error_text>` when applying `<pragma_kind>` around `<function_or_loop>`. What pragma placement pattern is legal for this exact case, and what code rewrite should be used instead?"
- Memory bottleneck:
  - "For the current HLS design, `<loop_or_region>` is limited by `<raw_warning_text>` on `<m_axi_or_array>`. What legal local-buffer / partition / reshape / burst patterns match this access pattern, and what should be rewritten before pragma tuning?"
- Dataflow legality:
  - "For the current HLS design, Vitis reports `<raw_error_text>` in a DATAFLOW region involving `<functions_or_ports>`. What structural rewrite is required to make the dataflow legal before further pragma exploration?"
- II / latency:
  - "For the current HLS design, `<loop_or_region>` shows `<raw_issue_text>` and current bottlenecks `<profile_summary>`. What legal loop/function/memory rewrites should be applied before trying pragma tuning again?"

### Anti-patterns
- "How to optimize this HLS design?"
- "What pragmas should I use?"
- "How to optimize nested loops in Vitis HLS?"
- Any question that omits the current code shape or raw Vitis error text when such evidence exists.
