/* Host calls understood by riscv/emulator/rv32emu.c (see its header comment). */
#ifndef RV32_HOST_H
#define RV32_HOST_H

#include <stdint.h>

enum {
    HOST_EXIT = 93, HOST_WRITE = 64,
    HOST_FRAME = 0x10000, HOST_TICKS, HOST_SLEEP, HOST_KEY, HOST_ARGS, HOST_ROM,
};

static inline uint32_t host_call(uint32_t number, uint32_t a0, uint32_t a1, uint32_t a2) {
    register uint32_t r_a0 __asm__("a0") = a0;
    register uint32_t r_a1 __asm__("a1") = a1;
    register uint32_t r_a2 __asm__("a2") = a2;
    register uint32_t r_a7 __asm__("a7") = number;
    __asm__ volatile("ecall" : "+r"(r_a0), "+r"(r_a1) : "r"(r_a2), "r"(r_a7) : "memory");
    return r_a0;
}

static inline uint32_t host_rom(uint32_t *size) {
    register uint32_t r_a0 __asm__("a0");
    register uint32_t r_a1 __asm__("a1");
    register uint32_t r_a7 __asm__("a7") = HOST_ROM;
    __asm__ volatile("ecall" : "=r"(r_a0), "=r"(r_a1) : "r"(r_a7) : "memory");
    *size = r_a1;
    return r_a0;
}

#endif
