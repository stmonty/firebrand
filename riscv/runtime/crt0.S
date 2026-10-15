/* Set the stack and global pointers, then enter C. */
    .section .text.start
    .globl _start
_start:
    .option push
    .option norelax
    la gp, __global_pointer$
    .option pop
    la sp, __stack_top
    call __runtime_start
1:  j 1b
