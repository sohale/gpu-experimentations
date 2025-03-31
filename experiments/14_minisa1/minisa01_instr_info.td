//===-- minisa01_instr_info.td - Minisa01 Instruction Definitions -*- tablegen -*-===//
//
//
// This file defines the instruction set for the Minisa01 architecture.
//
// This file defines the instruction encodings and formats for the Minisa01 target.
// Minisa01 is a minimal, Turing-complete ISA designed for MLIR lowering,
// emulation, and performance profiling.
//
//===----------------------------------------------------------------------===//

// non-exizst:
// include "llvm/Target/TargetInstrInfo.td"
// already included in top-level .td file?
// include "llvm/Target/TargetSchedule.td"

/// Instruction Format Classes
// Defining "bitfield layouts" for different instruction encoding types.

// Minisa01InstrInfo, the dad class for all instruction formats.
// Minisa01InstrInfo
def Minisa01InstrInfo : InstrInfo {
  bits<16> Inst;
  // Implies that, not all instruciotns are 16-bit, but the "ordinary ones"
}

// Minisa01 has some ordinary vs extended (custom) instrucitons
// The ordinary ones are regular
// good names: orgniary, regular
// bad names: core, common

// class Minisa01OrdinaryInstr
class OrdinaryInstr : InstrInfo {
  bits<16> Inst;
}
// math, arithmatic, calculations, fp, alu, fpu
// MathInstr
class MathBinaryInstr<bits<4> opcode> : InstrInfo {
  bits<16> Inst;
}


class LoadStore<bits<4> opcode> : OrdinaryInstr  {

  bits<16> Inst;

  bits<4> Opcode = opcode;
  bits<3> Rd;
  bits<3> Rs;
  bits<8> Imm;
  //`Inst`:  If you're building your own instruction class, you must explicitly declare it:
  // otherwise, inherit from `: Instruction`, ( how about `InstrInfo`? )

  // let Inst = { opcode, Rd, Rs, Imm };
  let Inst{15-12} = Opcode;
  let Inst{11-9}  = Rd;
  let Inst{8-6}   = Rs;
  let Inst{7-0}   = Imm;
}

class ALU3Reg<bits<4> opcode> : OrdinaryInstr {
  bits<3> Rd;
  bits<3> Rs;
  bits<3> Rt;

  let Inst = { opcode, Rd, Rs, Rt, 0b000 };
}

class Jump<bits<4> opcode> {
  bits<11> Imm;
  let Inst = { opcode, 0b0, Imm };
}

class BranchEQ<bits<4> opcode> {
  bits<3> Rs;
  bits<3> Rt;
  bits<5> Imm;
  let Inst = { opcode, Rs, Rt, Imm };
}

class Halt<bits<4> opcode> {
  let Inst = { opcode, 0x00 };
}

class VDotConfigImm16<bits<8> vop> {
  bits<16> Imm;
  let Inst = { 0b1000, vop, 0b0, Imm };
}

class VDotConfigDst<bits<8> vop> {
  bits<3> Rd;
  let Inst = { 0b1000, vop, 0b0, 0b00000, Rd };
}

class VDotStartTrigger {
  let Inst = { 0b10001111, 0b0, 0x0000 };
}

/// Instruction Definitions

//===----------------------------------------------------------------------===//
// Core Instruction Set
//===----------------------------------------------------------------------===//

def LOAD  : LoadStore<0b0000> {
  let Mnemonic = "LOAD";
  let hasSideEffects = 0;
}

def STORE : LoadStore<0b0001> {
  let Mnemonic = "STORE";
  let hasSideEffects = 1;
}

def ADD   : ALU3Reg<0b0010> {
  let Mnemonic = "ADD";
}

def SUB   : ALU3Reg<0b0011> {
  let Mnemonic = "SUB";
}

def MULF  : ALU3Reg<0b0110> {
  let Mnemonic = "MULF";
}

//===----------------------------------------------------------------------===//
// Control Flow Instructions
//===----------------------------------------------------------------------===//

def JMP : Jump<0b0100> {
  let Mnemonic = "JMP";
  let isBranch = 1;
  let isTerminator = 1;
}

def BEQ : BranchEQ<0b0101> {
  let Mnemonic = "BEQ";
  let isBranch = 1;
  let isConditionalBranch = 1;
}

def HALT : Halt<0b1111> {
  let Mnemonic = "HALT";
  let isTerminator = 1;
  let hasSideEffects = 1;
}

//===----------------------------------------------------------------------===//
// VDOT Extension Instructions
//===----------------------------------------------------------------------===//

def VDOT_START1   : VDotConfigImm16<0b00000000> {
  let Mnemonic = "VDOT.START1";
}

def VDOT_START2   : VDotConfigImm16<0b00000001> {
  let Mnemonic = "VDOT.START2";
}

def VDOT_STRIDE1  : VDotConfigImm16<0b00000010> {
  let Mnemonic = "VDOT.STRIDE1";
}

def VDOT_STRIDE2  : VDotConfigImm16<0b00000011> {
  let Mnemonic = "VDOT.STRIDE2";
}

def VDOT_LEN      : VDotConfigImm16<0b00000100> {
  let Mnemonic = "VDOT.LEN";
}

def VDOT_DST      : VDotConfigDst<0b00000101> {
  let Mnemonic = "VDOT.DST";
}

def VDOT_START    : VDotStartTrigger {
  let Mnemonic = "VDOT.START";
  let hasSideEffects = 1;
}

/*
def VDOT_SETUP : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins VDOTConfigRegs:$len, VDOTConfigRegs:$stride1, VDOTConfigRegs:$stride2, VDOTConfigRegs:$start1, VDOTConfigRegs:$start2, VDOTConfigRegs:$dst);
  let Opcode = 0xXX; // Replace with the actual opcode
  let AsmString = "vdot.setup $len, $stride1, $stride2, $start1, $start2, $dst";
  let Constraints = "";
}
*/


class Minisa01Instr<string mnemonic, bits<16> encoding,
                    list<dag> operands, dag pattern>
    : Instruction {
  let Mnemonic = mnemonic;
  let Encoding = encoding;
  let OperandList = operands;
  let Pattern = pattern;
}

def ADD : Minisa01Instr<"add", 0b0001,
                        [(outs GPR:$Rd), (ins GPR:$Rs, GPR:$Rt)],
                        (add GPR:$Rs, GPR:$Rt)>;
