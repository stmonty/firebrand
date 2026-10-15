/* Guest memory: RAM is [0, 16 MiB). Programs load at 0x10000; the stack grows down
   from the top of RAM; the heap grows up from _end. The ROM (a WAD, for DOOM) is
   mapped read-only at 0x40000000 by the emulator. */
OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY { RAM (rwx) : ORIGIN = 0x10000, LENGTH = 16M - 0x10000 }
PHDRS { text PT_LOAD FLAGS(5); data PT_LOAD FLAGS(6); }

SECTIONS
{
  .text : { *(.text.start) *(.text .text.*) } > RAM :text
  . = ALIGN(4096);                   /* Keep data out of the executable segment. */
  .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > RAM :data
  .init_array : {
    PROVIDE_HIDDEN(__init_array_start = .);
    KEEP(*(SORT(.init_array.*))) KEEP(*(.init_array))
    PROVIDE_HIDDEN(__init_array_end = .);
  } > RAM
  .fini_array : {
    PROVIDE_HIDDEN(__fini_array_start = .);
    KEEP(*(SORT(.fini_array.*))) KEEP(*(.fini_array))
    PROVIDE_HIDDEN(__fini_array_end = .);
  } > RAM
  .data : { *(.data .data.*) } > RAM
  .sdata : {
    __global_pointer$ = . + 0x800;
    *(.sdata .sdata.*)
  } > RAM
  .bss (NOLOAD) : {
    __bss_start = .;
    *(.sbss .sbss.*) *(.bss .bss.*) *(COMMON)
    . = ALIGN(4);
    __bss_end = .;
  } > RAM
  _end = .;
  /* The stack starts at the top of RAM unless the build sets a lower top
     (-Wl,--defsym=__stack_top=...), which also bounds the heap. */
  PROVIDE(__stack_top = ORIGIN(RAM) + LENGTH(RAM));
  /DISCARD/ : { *(.comment) *(.riscv.attributes) }
}
