# Small C programs for the transformer interpreter (8 KiB RAM, riscv/runtime/tiny.ld).
ROOT := ../..
PREFIX := $(ROOT)/third_party/rv32im-gcc/bin/riscv32-none-elf-
PROGRAMS := fib sort crc32 fixed sieve gcd60 primes60
CFLAGS := -march=rv32im -mabi=ilp32 -O2 -g -ffreestanding -fno-builtin
LDFLAGS := -nostdlib -nostartfiles -T ../runtime/tiny.ld

all: $(addprefix build/,$(addsuffix .elf,$(PROGRAMS)))

build/%.elf: %.c ../runtime/tiny_crt0.S | build
	$(PREFIX)gcc $(CFLAGS) $(LDFLAGS) ../runtime/tiny_crt0.S $< -lgcc -o $@
	@$(PREFIX)size $@ | tail -1
build:
	mkdir -p $@

check: all
	@for p in $(PROGRAMS); do ../emulator/rv32emu --trace-exit build/$$p.elf 2>&1 | sed "s/^/$$p: /"; done
