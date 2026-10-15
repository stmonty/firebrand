/* Minimal riscv-tests environment for rv32emu (and the transformer interpreter).
 * User-level tests only: no CSRs, traps, or privilege modes. A test exits with
 * code 0 on success, or its failing test number (gp) on failure. */
#ifndef RV32EMU_RISCV_TEST_H
#define RV32EMU_RISCV_TEST_H

#define RVTEST_RV32U .macro init; .endm
#define RVTEST_RV64U RVTEST_RV32U
#define TESTNUM gp

#define RVTEST_CODE_BEGIN        \
    .section .text.start;        \
    .globl _start;               \
_start:                          \
    li TESTNUM, 0;               \
    init;

#define RVTEST_CODE_END unimp

#define RVTEST_PASS              \
    li a0, 0;                    \
    li a7, 93;                   \
    ecall;

#define RVTEST_FAIL              \
    mv a0, TESTNUM;              \
    li a7, 93;                   \
    ecall;

#define EXTRA_DATA
#define RVTEST_DATA_BEGIN .data; .balign 16; .globl begin_signature; begin_signature:
#define RVTEST_DATA_END .balign 16; .globl end_signature; end_signature:

#endif
