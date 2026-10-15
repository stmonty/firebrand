# RV32IM (no compressed, atomic, or floating-point instructions), ilp32 soft-float ABI.
# Newlib and libgcc are built for exactly this ISA, so linked programs contain only
# instructions our reference emulator and transformer interpreter implement.
{ pkgs ? import <nixpkgs> { } }:
let
  cross = import pkgs.path {
    crossSystem = {
      config = "riscv32-none-elf";
      libc = "newlib";
      gcc = { arch = "rv32im"; abi = "ilp32"; };
    };
  };
in
cross.buildPackages.gcc
