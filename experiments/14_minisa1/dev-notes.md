
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
