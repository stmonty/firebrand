/* Small programs for the transformer interpreter: 8 KiB of RAM at address 0.
   Code, data, and stack share it; the stack starts at the top. Keeping memory
   small keeps every register key, memory-word key, and pc a single token. */
OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY { RAM (rwx) : ORIGIN = 0, LENGTH = 8K }
PHDRS { text PT_LOAD FLAGS(5); data PT_LOAD FLAGS(6); }

SECTIONS
{
  .text : { *(.text.start) *(.text .text.*) } > RAM :text
  .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > RAM :data
  .data : { *(.data .data.*) } > RAM :data
  .sdata : { __global_pointer$ = . + 0x800; *(.sdata .sdata.*) } > RAM :data
  .bss (NOLOAD) : { __bss_start = .; *(.sbss .sbss.*) *(.bss .bss.*) *(COMMON) . = ALIGN(4); __bss_end = .; } > RAM :data
  _end = .;
  __stack_top = ORIGIN(RAM) + LENGTH(RAM);
  /DISCARD/ : { *(.comment) *(.riscv.attributes) }
}
