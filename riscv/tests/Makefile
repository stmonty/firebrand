# Build the user-level RV32IM riscv-tests for rv32emu.
ROOT := ../..
ISA := $(ROOT)/third_party/riscv-tests/isa
PREFIX := $(ROOT)/third_party/rv32im-gcc/bin/riscv32-none-elf-
EMU := ../emulator/rv32emu

# fence_i needs self-modifying code (Zifencei); ma_data needs a misaligned-access trap policy.
UI := simple add addi and andi auipc beq bge bgeu blt bltu bne jal jalr lb lbu lh lhu lui lw \
      ld_st or ori sb sh st_ld sll slli slt slti sltiu sltu sra srai srl srli sub sw xor xori
UM := div divu mul mulh mulhsu mulhu rem remu
TESTS := $(addprefix rv32ui-,$(UI)) $(addprefix rv32um-,$(UM))
ELFS := $(addprefix build/,$(addsuffix .elf,$(TESTS)))
# The same tests linked into 8 KiB at address 0, for the transformer interpreter.
TINY := $(addprefix build_tiny/,$(addsuffix .elf,$(TESTS)))

all: $(ELFS) $(TINY)

build/rv32ui-%.elf: $(ISA)/rv32ui/%.S env/riscv_test.h | build
	$(PREFIX)gcc -march=rv32im -mabi=ilp32 -nostdlib -nostartfiles -T ../runtime/link.ld \
	    -Ienv -I$(ISA)/macros/scalar $< -o $@
build/rv32um-%.elf: $(ISA)/rv32um/%.S env/riscv_test.h | build
	$(PREFIX)gcc -march=rv32im -mabi=ilp32 -nostdlib -nostartfiles -T ../runtime/link.ld \
	    -Ienv -I$(ISA)/macros/scalar $< -o $@
build_tiny/rv32ui-%.elf: $(ISA)/rv32ui/%.S env/riscv_test.h | build_tiny
	$(PREFIX)gcc -march=rv32im -mabi=ilp32 -nostdlib -nostartfiles -T ../runtime/tiny.ld \
	    -Ienv -I$(ISA)/macros/scalar $< -o $@
build_tiny/rv32um-%.elf: $(ISA)/rv32um/%.S env/riscv_test.h | build_tiny
	$(PREFIX)gcc -march=rv32im -mabi=ilp32 -nostdlib -nostartfiles -T ../runtime/tiny.ld \
	    -Ienv -I$(ISA)/macros/scalar $< -o $@
build build_tiny:
	mkdir -p $@

check: all
	@fail=0; for t in $(TESTS); do for d in build build_tiny; do \
	    if $(EMU) $$d/$$t.elf >/dev/null 2>&1; then echo "PASS $$d/$$t"; \
	    else echo "FAIL $$d/$$t (exit $$?)"; fail=1; fi; done; done; exit $$fail
