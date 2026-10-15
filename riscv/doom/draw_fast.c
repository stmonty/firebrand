/*
 * Faster R_DrawColumn and R_DrawSpan, built with `make FAST_DRAW=1`.
 *
 * The stock loops reload the source and colormap pointers from globals after
 * every pixel store (C's aliasing rules require it). These copy them into locals
 * and unroll by four, with the same per-pixel arithmetic. The stock functions are
 * renamed at compile time (see the Makefile) and dropped by --gc-sections.
 */

#include "doomdef.h"
#include "i_system.h"
#include "r_local.h"

extern byte *ylookup[];
extern int columnofs[];

void R_DrawColumn(void)
{
    int count = dc_yh - dc_yl;
    if (count < 0)
        return;
#ifdef RANGECHECK
    if ((unsigned)dc_x >= SCREENWIDTH || dc_yl < 0 || dc_yh >= SCREENHEIGHT)
        I_Error("R_DrawColumn: %i to %i at %i", dc_yl, dc_yh, dc_x);
#endif
    byte *dest = ylookup[dc_yl] + columnofs[dc_x];
    fixed_t fracstep = dc_iscale;
    fixed_t frac = dc_texturemid + (dc_yl - centery) * fracstep;
    const byte *source = dc_source;
    const lighttable_t *colormap = dc_colormap;

    count++;                                    /* Pixels to draw. */
    for (; count >= 4; count -= 4) {
        dest[0] = colormap[source[(frac >> FRACBITS) & 127]];
        frac += fracstep;
        dest[SCREENWIDTH] = colormap[source[(frac >> FRACBITS) & 127]];
        frac += fracstep;
        dest[2 * SCREENWIDTH] = colormap[source[(frac >> FRACBITS) & 127]];
        frac += fracstep;
        dest[3 * SCREENWIDTH] = colormap[source[(frac >> FRACBITS) & 127]];
        frac += fracstep;
        dest += 4 * SCREENWIDTH;
    }
    for (; count > 0; count--) {
        *dest = colormap[source[(frac >> FRACBITS) & 127]];
        dest += SCREENWIDTH;
        frac += fracstep;
    }
}

void R_DrawSpan(void)
{
#ifdef RANGECHECK
    if (ds_x2 < ds_x1 || ds_x1 < 0 || ds_x2 >= SCREENWIDTH || (unsigned)ds_y > SCREENHEIGHT)
        I_Error("R_DrawSpan: %i to %i at %i", ds_x1, ds_x2, ds_y);
#endif
    /* x in the top 16 bits and y in the bottom 16, each 6.10 fixed point. */
    unsigned int position = ((ds_xfrac << 10) & 0xffff0000) | ((ds_yfrac >> 6) & 0x0000ffff);
    unsigned int step = ((ds_xstep << 10) & 0xffff0000) | ((ds_ystep >> 6) & 0x0000ffff);
    byte *dest = ylookup[ds_y] + columnofs[ds_x1];
    const byte *source = ds_source;
    const lighttable_t *colormap = ds_colormap;
    int count = ds_x2 - ds_x1 + 1;              /* Pixels to draw. */

#define SPOT(p) ((int)(((p) >> 26) | (((p) >> 4) & 0x0fc0)))
    for (; count >= 4; count -= 4) {
        dest[0] = colormap[source[SPOT(position)]];
        position += step;
        dest[1] = colormap[source[SPOT(position)]];
        position += step;
        dest[2] = colormap[source[SPOT(position)]];
        position += step;
        dest[3] = colormap[source[SPOT(position)]];
        position += step;
        dest += 4;
    }
    for (; count > 0; count--) {
        *dest++ = colormap[source[SPOT(position)]];
        position += step;
    }
#undef SPOT
}
