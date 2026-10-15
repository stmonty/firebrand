/*
 * Newlib system calls for the rv32emu guest.
 *
 * Console output goes to the host. Every file whose name ends in ".wad" opens
 * the read-only ROM image, so the engine reads its data from memory rather than
 * through a host file service. Other files do not exist; writes to them fail.
 *
 * Only three host calls remain, so the transformer interpreter can execute the
 * same program: exit, write (console text), and frame. The ROM is at a fixed
 * address and its size comes from its own WAD header; the program's arguments
 * are compiled in; and the loader provides zeroed memory beyond the image, so
 * .bss needs no clearing loop.
 */

#include <errno.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/times.h>

#include "rv32.h"

#undef errno
extern int errno;

extern char _end[], __stack_top[];
extern void __libc_init_array(void);
extern int main(int argc, char **argv);
extern void exit(int);

#define MAX_FILES 8
#define FIRST_FILE 3
#define STACK_RESERVE (256 << 10)          /* Heap stops this far below the stack top. */

static struct { int open; uint32_t offset; } files[MAX_FILES];
#define ROM_BASE 0x40000000u

static const uint8_t *rom = (const uint8_t *)ROM_BASE;
static uint32_t rom_size;
static char *heap_end = _end;

void _init(void) {}
void _fini(void) {}

/* A WAD's size is its directory's end: offset + 16 bytes per lump. */
static uint32_t wad_size(void) {
    if ((rom[0] != 'I' && rom[0] != 'P') || rom[1] != 'W' || rom[2] != 'A' || rom[3] != 'D') return 0;
    uint32_t lumps, directory;
    memcpy(&lumps, rom + 4, 4);
    memcpy(&directory, rom + 8, 4);
    return directory + 16 * lumps;
}

extern const char *const guest_argv[];
extern const int guest_argc;

void __runtime_start(void) {
    rom_size = wad_size();
    __libc_init_array();
    exit(main(guest_argc, (char **)guest_argv));
}

void _exit(int code) {
    host_call(HOST_EXIT, (uint32_t)code, 0, 0);
    for (;;) {}
}

static int is_wad(const char *name) {
    size_t n = strlen(name);
    if (n < 4) return 0;
    const char *e = name + n - 4;
    return e[0] == '.' && (e[1] | 32) == 'w' && (e[2] | 32) == 'a' && (e[3] | 32) == 'd';
}

int _open(const char *name, int flags, int mode) {
    (void)mode;
    if (!rom_size || !is_wad(name) || (flags & 3) != 0) { errno = ENOENT; return -1; }
    for (int i = 0; i < MAX_FILES; i++) {
        if (!files[i].open) {
            files[i].open = 1;
            files[i].offset = 0;
            return i + FIRST_FILE;
        }
    }
    errno = EMFILE;
    return -1;
}

static int valid(int fd) {
    return fd >= FIRST_FILE && fd < FIRST_FILE + MAX_FILES && files[fd - FIRST_FILE].open;
}

int _read(int fd, void *buffer, int length) {
    if (!valid(fd)) { errno = EBADF; return -1; }
    uint32_t *offset = &files[fd - FIRST_FILE].offset;
    uint32_t available = *offset < rom_size ? rom_size - *offset : 0;
    uint32_t n = (uint32_t)length < available ? (uint32_t)length : available;
    memcpy(buffer, rom + *offset, n);
    *offset += n;
    return (int)n;
}

int _lseek(int fd, int offset, int whence) {
    if (!valid(fd)) { errno = EBADF; return -1; }
    uint32_t *position = &files[fd - FIRST_FILE].offset;
    int64_t target = whence == 0 ? offset : whence == 1 ? (int64_t)*position + offset
                                                       : (int64_t)rom_size + offset;
    if (target < 0) { errno = EINVAL; return -1; }
    *position = (uint32_t)target;
    return (int)target;
}

int _close(int fd) {
    if (!valid(fd)) { errno = EBADF; return -1; }
    files[fd - FIRST_FILE].open = 0;
    return 0;
}

int _write(int fd, const void *buffer, int length) {
    if (fd == 1 || fd == 2) return (int)host_call(HOST_WRITE, (uint32_t)fd, (uint32_t)buffer, (uint32_t)length);
    errno = EBADF;
    return -1;
}

int _fstat(int fd, struct stat *st) {
    memset(st, 0, sizeof *st);
    if (fd < FIRST_FILE) { st->st_mode = S_IFCHR; return 0; }
    if (!valid(fd)) { errno = EBADF; return -1; }
    st->st_mode = S_IFREG;
    st->st_size = rom_size;
    return 0;
}

int _stat(const char *name, struct stat *st) {
    memset(st, 0, sizeof *st);
    if (!rom_size || !is_wad(name)) { errno = ENOENT; return -1; }
    st->st_mode = S_IFREG;
    st->st_size = rom_size;
    return 0;
}

int _isatty(int fd) { return fd < FIRST_FILE; }

void *_sbrk(int increment) {
    char *previous = heap_end;
    if (heap_end + increment > __stack_top - STACK_RESERVE) { errno = ENOMEM; return (void *)-1; }
    heap_end += increment;
    return previous;
}

int _gettimeofday(struct timeval *tv, void *tz) {
    (void)tz;
    tv->tv_sec = tv->tv_usec = 0;       /* No wall clock: DOOM keeps its own virtual time. */
    return 0;
}

clock_t _times(struct tms *buffer) { memset(buffer, 0, sizeof *buffer); return (clock_t)-1; }
int _getpid(void) { return 1; }
int _kill(int pid, int signal) { (void)pid; (void)signal; errno = EINVAL; return -1; }
int _link(const char *a, const char *b) { (void)a; (void)b; errno = EMLINK; return -1; }
int _unlink(const char *name) { (void)name; errno = ENOENT; return -1; }
int mkdir(const char *name, mode_t mode) { (void)name; (void)mode; errno = EROFS; return -1; }
