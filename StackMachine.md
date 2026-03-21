
Verified Stack Machine Compiler
===


Verified Stack Machine Compiler
===
In this project we build a verified compiler pipeline in Lean by defining:

- An **expression language** (AST) with `lit`, `add`, `sub`, and `mul` constructors, together with a recursive evaluator `eval : Expr → Nat`.

- A **stack machine** with instructions `PUSH`, `ADD`, `SUB`, `MUL` and an executor `execute : List Instr → Stack → Option Stack` that returns `none` on stack underflow.

- A **compiler** `compile : Expr → List Instr` using postorder traversal, with a proof of **correctness**: `execute (compile e) s = some (eval e :: s)`.

- An **optimizer** using algebraic identities (identity elimination, constant folding) with a proof that `eval (optimize e) = eval e`.

- An end-to-end **pipeline** composing optimization and compilation, with a proof that `execute (pipeline e) s = some (eval e :: s)`.


Details of the Compiler
===
▸ **Expressions** are compiled to stack machine code via postorder traversal:
```lean
def compile : Expr → List Instr
  | .lit n   => [.PUSH n]
  | .add a b => compile a ++ compile b ++ [.ADD]
  | .sub a b => compile a ++ compile b ++ [.SUB]
  | .mul a b => compile a ++ compile b ++ [.MUL]
```

▸ **Correctness** is proved by structural induction using a key lemma `execute_append` that shows execution distributes over instruction concatenation:
```lean
theorem compile_correct (e : Expr) (s : Stack) :
    execute (compile e) s = some (eval e :: s)
```

▸ The **optimizer** applies smart constructors like `addOpt` that simplify `a + 0 → a`, `0 + b → b`, and fold constants. For example, `(2 + 0) * (1 * 3)` optimizes to `lit 6`, reducing 7 instructions to just `[PUSH 6]`.


Example Execution
===
▸ Consider the expression `(2 + 3) * 4`. Compilation produces:
```
compile (mul (add (lit 2) (lit 3)) (lit 4))
  = [PUSH 2, PUSH 3, ADD, PUSH 4, MUL]
```

▸ **Step-by-step execution** on an empty stack:

| Step | Instruction | Stack |
|------|-------------|-------|
| 0 | — | `[]` |
| 1 | `PUSH 2` | `[2]` |
| 2 | `PUSH 3` | `[3, 2]` |
| 3 | `ADD` | `[5]` |
| 4 | `PUSH 4` | `[4, 5]` |
| 5 | `MUL` | `[20]` |

▸ With the **optimizer**, `(2 + 0) * (1 * 3)` is simplified before compilation:
```
optimize (mul (add (lit 2) (lit 0)) (mul (lit 1) (lit 3)))
  = lit 6

pipeline: [PUSH 6]  →  execute on [] gives [6]
```
&nbsp;&nbsp;&nbsp; 7 instructions reduced to 1 — and `pipeline_correct` guarantees the result is the same.

