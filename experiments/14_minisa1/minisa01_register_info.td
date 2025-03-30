//===-- minisa01_register_info.td - Minisa01 Register Definitions -*- tablegen -*-===//
//
// This file defines the register set for the Minisa01 architecture, including
// general-purpose registers R0 through R7.
//
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

/// Register Definitions
/// --------------------

def R0 : Register<"R0">;
def R1 : Register<"R1">;
def R2 : Register<"R2">;
def R3 : Register<"R3">;
def R4 : Register<"R4">;
def R5 : Register<"R5">;
def R6 : Register<"R6">;
def R7 : Register<"R7">;

/// Special-Purpose Registers:
/// VDOT Configuration Registers
/// ----------------------------

def VDOT_LEN     : Register<"VDOT.LEN">;
def VDOT_STRIDE1 : Register<"VDOT.STRIDE1">;
def VDOT_STRIDE2 : Register<"VDOT.STRIDE2">;
def VDOT_START1  : Register<"VDOT.START1">;
def VDOT_START2  : Register<"VDOT.START2">;
def VDOT_DST     : Register<"VDOT.DST">;

/// Register Classes
/// ----------------

def GPR : RegisterClass<"Minisa01", [i16], 16, (add R0, R1, R2, R3, R4, R5, R6, R7)> {
  let RegInfos = [
    (R0, "R0", ["zero"]),
    (R1, "R1", ["at"]),
    (R2, "R2", ["v0"]),
    (R3, "R3", ["v1"]),
    (R4, "R4", ["a0"]),
    (R5, "R5", ["a1"]),
    (R6, "R6", ["t0"]),
    (R7, "R7", ["t1"])
  ];
}

def VDOTConfigRegs : RegisterClass<"Minisa01", [i16], 16, (add VDOT_LEN, VDOT_STRIDE1, VDOT_STRIDE2, VDOT_START1, VDOT_START2, VDOT_DST)> {
  let RegInfos = [
    (VDOT_LEN,     "VDOT.LEN",     ["vlen"]),
    (VDOT_STRIDE1, "VDOT.STRIDE1", ["vstride1"]),
    (VDOT_STRIDE2, "VDOT.STRIDE2", ["vstride2"]),
    (VDOT_START1,  "VDOT.START1",  ["vstart1"]),
    (VDOT_START2,  "VDOT.START2",  ["vstart2"]),
    (VDOT_DST,     "VDOT.DST",     ["vdst"])
  ];
}
