/* doomgeneric platform hooks for rv32emu and the transformer interpreter:
   320x200 palette indices, guest-side virtual time, compiled-in arguments. */

#include "doomgeneric.h"
#include "i_video.h"
#include "../runtime/rv32.h"

#ifndef DOOM_ARGS
#define DOOM_ARGS "-iwad", "doom1.wad", "-timedemo", "demo1"
#endif

/* DOOM_ZONE_MB sets the zone (DOOM's cache for WAD data) with -mb; a smaller
   zone means a smaller RAM for the transformer to carry between contexts. */
#define STRING(x) #x
#define EXPAND(x) STRING(x)
#ifdef DOOM_ZONE_MB
#define ZONE_ARGS , "-mb", EXPAND(DOOM_ZONE_MB)
#else
#define ZONE_ARGS
#endif

const char *const guest_argv[] = {"doom", DOOM_ARGS ZONE_ARGS, 0};
const int guest_argc = sizeof guest_argv / sizeof guest_argv[0] - 1;

/* Virtual time: one DOOM tic (1/35 s) per rendered frame, plus requested sleeps.
   Rounding up makes floor(ms * 35 / 1000) equal the frame count exactly. */
static uint32_t frames, slept;

void DG_Init(void) {}

void DG_DrawFrame(void) {
    frames++;
    host_call(HOST_FRAME, (uint32_t)DG_ScreenBuffer, (uint32_t)colors, 0);
}

void DG_SleepMs(uint32_t ms) { slept += ms; }

uint32_t DG_GetTicksMs(void) { return (frames * 1000 + 34) / 35 + slept; }

int DG_GetKey(int *pressed, unsigned char *key) {
    (void)pressed;
    (void)key;
    return 0;                          /* Recorded demos need no live input. */
}

void DG_SetWindowTitle(const char *title) { (void)title; }

int main(int argc, char **argv) {
    doomgeneric_Create(argc, argv);
    for (;;) doomgeneric_Tick();
}
