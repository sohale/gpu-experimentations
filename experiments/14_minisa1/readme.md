# An TableGen experiment
An experiment for an MLIR dialect, end-to-end focusing on compiler back-end development on MLIR.


Virtual machine specifications:

## 🧠 MiniISA1: Minimal Instruction Set Architecture for MLIR PoC

MiniISA1 is a minimal, Turing-complete CPU model designed for MLIR dialect development, emulation, and compiler optimization passes. It includes a basic instruction set and a vector/tensor extension for operations like dot products. The goal is to serve as a backend target for MLIR lowering, fusion, and profiling.

---

### 🔹 Architecture Overview

- **Word Size**: 16 bits
- **Address Space**: 28-bit flat space (up to 256MB emulated)
- **Registers**:  
  - `R0`–`R7` (General-purpose; `R0` is hardwired to 0)

---

### 🔹 Instruction Encoding Format

Each instruction is 16 bits unless otherwise stated. Extended instructions (VDOT-related) use `1000 xxxx` prefixes.

| Field | Description              |
|-------|--------------------------|
| `ddd` | Destination register     |
| `sss` | Source register 1        |
| `ttt` | Source register 2        |
| `iii` | Immediate values         |

---

### 🔹 Core Instruction Set

| Mnemonic             | Encoding                      | Description                                 |
|----------------------|-------------------------------|---------------------------------------------|
| `LOAD Rd, [Rs+imm8]` | `0000 ddd sss iiiiiiii`       | Rd ← MEM[Rs + imm8]                         |
| `STORE Rd, [Rs+imm8]`| `0001 ddd sss iiiiiiii`       | MEM[Rs + imm8] ← Rd                         |
| `ADD Rd, Rs, Rt`     | `0010 ddd sss ttt 000`        | Rd ← Rs + Rt                                |
| `SUB Rd, Rs, Rt`     | `0011 ddd sss ttt 000`        | Rd ← Rs - Rt                                |
| `JMP imm11`          | `0100 0 iiiiiiiiiii`          | PC ← PC + imm11 (signed)                    |
| `BEQ Rs, Rt, imm5`   | `0101 sss ttt iiiii`          | if Rs == Rt: PC ← PC + imm5 (signed)        |
| `HALT`               | `1111 0000 00000000`          | Stop execution                              |

---

### 🔹 VDOT Extension (Vector Dot Product Unit)

The VDOT unit operates via special-purpose registers and an explicit trigger. It enables MLIR lowering from `linalg.dot` or `matmul`.

#### Special VDOT Configuration Registers

| Name           | Width | Purpose                   |
|----------------|-------|---------------------------|
| `VDOT.LEN`     | 16    | Length of input vectors   |
| `VDOT.STRIDE1` | 16    | Stride for vector 1       |
| `VDOT.STRIDE2` | 16    | Stride for vector 2       |
| `VDOT.START1`  | 32    | Base address for vector 1 |
| `VDOT.START2`  | 32    | Base address for vector 2 |
| `VDOT.DST`     | 3     | Destination GPR (R1–R7)   |

#### VDOT Instruction Sequence

| Mnemonic              | Encoding                          | Description                              |
|-----------------------|-----------------------------------|------------------------------------------|
| `VDOT.START1 IMM16`   | `1000 0000 0 iiiiiiiiiiiiiii`     | Set vector 1 base address                |
| `VDOT.START2 IMM16`   | `1000 0001 0 iiiiiiiiiiiiiii`     | Set vector 2 base address                |
| `VDOT.STRIDE1 IMM16`  | `1000 0010 0 iiiiiiiiiiiiiii`     | Set stride for vector 1                  |
| `VDOT.STRIDE2 IMM16`  | `1000 0011 0 iiiiiiiiiiiiiii`     | Set stride for vector 2                  |
| `VDOT.LEN IMM16`      | `1000 0100 0 iiiiiiiiiiiiiii`     | Set number of elements                   |
| `VDOT.DST Rd`         | `1000 0101 0 00000 ddd`           | Set destination GPR                      |
| `VDOT.START`          | `1000 1111 0 0000000000000000`    | Trigger the configured VDOT operation    |

---

### 🔹 Vector Memory Instructions (Optional)

| Mnemonic                   | Semantics                                               |
|----------------------------|---------------------------------------------------------|
| `VLOAD Rd, Rs, stride, len`| Rd[i] ← MEM[Rs + i × stride] for i in [0, len)         |
| `VSTORE Rd, Rs, stride, len`| MEM[Rs + i × stride] ← Rd[i] for i in [0, len)         |

> These can be lowered into VDOT-style configurations or unrolled loops.

---

### 🔹 MLIR Relevance

- **Lowering Targets**:
  - `linalg.dot`, `linalg.matmul` → `VDOT` sequence
  - `linalg.copy`, `tensor.extract_slice` → `VLOAD`/`VSTORE`

- **Pass Development**:
  - Structured op fusion
  - Tiling, vectorization, loop elimination
  - Profiling at instruction and operand level

- **Compiler Pipeline Experiments**:
  - Instruction scheduling
  - Latency modeling
  - Buffer placement
  - IR transformation passes

---

### ✅ Summary

MiniISA is intentionally minimalist, yet extensible with tensor-centric instructions. It supports:
- Emulation of simple and vectorized workloads
- Scalable profiling with MLIR passes
- Full A–Z compiler pipeline coverage for an AI compiler engineer PoC

Next step: TableGen specification.
