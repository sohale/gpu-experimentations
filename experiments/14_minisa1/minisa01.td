//===-- Minisa1.td - Top-Level ISA Target Description --------*- tablegen -*-===//

include "llvm/Target/Target.td"
include "minisa01_instr_info.td"
include "minisa01_register_info.td"


def Minisa01 : Target {
  //   let InstructionSet = minisa01_instr_info;

  //  needs to be defined in a def: a def of class InstrInfo
  let InstructionSet = Minisa01InstrInfo;
  let AssemblyWriters = [ /* TBD */ ];
  let CodeEmitter = "Minisa01CodeEmitter";
}
