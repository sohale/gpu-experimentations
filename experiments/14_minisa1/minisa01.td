//===-- Minisa1.td - Top-Level ISA Target Description --------*- tablegen -*-===//

include "llvm/Target/Target.td"

def Minisa01 : Target {
  let InstructionSet = Minisa01InstrInfo;
  let AssemblyWriters = [ /* TBD */ ];
  let CodeEmitter = "Minisa01CodeEmitter";
}
