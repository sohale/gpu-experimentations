# The How

Typical td files for an ISA:

`minisa01.td`	Top-level target descriptor (def Target)
`minisa01_instr_info.td`	Instruction formats + actual instruction defs
`minisa01_register_info.td`	Register class and register definitions
`minisa01_sched_model.td`	(optional) Scheduling/latency data


Header comment:
Breakdown of a td-signature-hinted comment
```td
//===-- minisa01_instr_info.td - Minisa01 Instruction Definitions -*- tablegen -*-===//
```


`//===-- ` : a file header comment, for human-read only(!).
* Typically, `//===-- Filename.td - Brief Description ---------------------------===//`, but just CL's style.



bitfield layouts

td files:
* `minisa01.td`: top level td file
* `minisa01_instr_info.td`:  instruction definitions ( classes, individual, encodings)

in other words,
* Target file	 	minisa01.td
* Instr info	 	minisa01_instr_info.td
* Register file	minisa01_register_info.td
* Schedule	 	minisa01_sched_model.td
* Classes/defs 	Minisa01Instr, Minisa01Reg


Register Definitions
Register Classes
Register Classes are "based on" Register Definitions!  (because, in TableGenLang, `def` means instantiations)

The
```md
def VDOT_SETUP : Instruction {
}
```


instruction encoding: is to drive the assembler, disassembler, and emulator table for minisa01.

Yet:
No MLIR dialects, types, or attributes—pure low-level IR/ISA.





Minisa01Reg and Minisa01Instr should be deinfed in which of my .md files?

Building .td files means: generate a .inc file based on your .td files.



Pool of tasks / (next) steps:

- [x] create minisa01_register_info.td
- [ ] Completing the VDOT config instructions in TableGen format
- [ ] Generating binary layout for these patterns (hex encoding and bit masks)
- [ ] focus on the C++ backend stubs to hook this into an emulator
- [ ] formalize the instruction table or control flow rules further
- [ ] move toward the TableGen schema design
- [ ] continue with minisa01_register_info.td (R0–R7)
- [ ] continue with minisa01.td (target top-level)
- [ ] Define Assembly Writers
- [ ] Implement the Code Emitter
- [ ] Integrate with Other Components

random

```
/mlir/llvm-project/
├── llvm/
│   └── include/
│       └── llvm/Target/Target.td
├── mlir/
│   └── include/
```


TableGen doesn’t do implicit declarations. Every field you refer to must be declared explicitly or inherited from a superclass that defines it.

  bits<16> Inst;
  let Inst{15-12} = Opcode;

akin to

class X;
def A : X;


`InstrInfo`
You don’t subclass InstrInfo
It's a record used by LLVM to collect metadata about a target's instruction set.

So, InstrInfo is a value! (a "def", v "record"). Like a singleton instance.

`InstrInfo` is, a record in TableGen
metadata about the (whole?) instruction set.

```
def MyInstrInfo : InstrInfo {
  let InstructionSet = MyInstructions;
}
```

Yet, stragely, syntactically, allows:
```
class LoadStore<bits<4> opcode> : InstrInfo  { ...
```
