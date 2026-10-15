# Build doomgeneric for RV32IM. Run `make` from this directory after building the
# toolchain: nix-build ../toolchain.nix -o ../../third_party/rv32im-gcc
ROOT := ../..
SRC := $(ROOT)/third_party/doomgeneric/doomgeneric
PREFIX := $(ROOT)/third_party/rv32im-gcc/bin/riscv32-none-elf-
OPT ?= -O2
ZONE_MB ?=
# RAM_TOP (bytes) moves the stack top down, making RAM compact: the transformer
# carries all of RAM between contexts. The default is the linker's 16 MiB.
RAM_TOP ?=
# FAST_DRAW=1 replaces R_DrawColumn and R_DrawSpan with draw_fast.c's versions
# (same pixels, fewer instructions per pixel).
FAST_DRAW ?=
comma := ,
BUILD := build$(subst -,_,$(OPT))$(if $(ZONE_MB),_mb$(ZONE_MB))$(if $(RAM_TOP),_ram$(RAM_TOP))$(if $(FAST_DRAW),_fastdraw)

ARCH := -march=rv32im -mabi=ilp32
CFLAGS := $(ARCH) $(OPT) -g -ffunction-sections -fdata-sections -w \
          -DCMAP256 -DDOOMGENERIC_RESX=320 -DDOOMGENERIC_RESY=200 \
          -DNORMALUNIX -DLINUX -D_DEFAULT_SOURCE -I$(SRC) \
          $(if $(ZONE_MB),-DDOOM_ZONE_MB=$(ZONE_MB))
LDFLAGS := $(ARCH) -nostartfiles -T ../runtime/link.ld -Wl,--gc-sections -Wl,-Map,$(BUILD)/doom.map \
           $(if $(RAM_TOP),-Wl$(comma)--defsym=__stack_top=$(RAM_TOP))

DOOM := dummy am_map doomdef doomstat dstrings d_event d_items d_iwad d_loop d_main d_mode d_net \
        f_finale f_wipe g_game hu_lib hu_stuff info i_cdmus i_endoom i_joystick i_scale i_sound \
        i_system i_timer memio m_argv m_bbox m_cheat m_config m_controls m_fixed m_menu m_misc \
        m_random p_ceilng p_doors p_enemy p_floor p_inter p_lights p_map p_maputl p_mobj p_plats \
        p_pspr p_saveg p_setup p_sight p_spec p_switch p_telept p_tick p_user r_bsp r_data r_draw \
        r_main r_plane r_segs r_sky r_things sha1 sounds statdump st_lib st_stuff s_sound tables \
        v_video wi_stuff w_checksum w_file w_main w_wad z_zone w_file_stdc i_input i_video doomgeneric
OBJS := $(addprefix $(BUILD)/,$(addsuffix .o,$(DOOM))) $(BUILD)/doomgeneric_rv32.o \
        $(BUILD)/syscalls.o $(BUILD)/crt0.o $(if $(FAST_DRAW),$(BUILD)/draw_fast.o)

$(BUILD)/doom.elf: $(OBJS)
	$(PREFIX)gcc $(LDFLAGS) $(OBJS) -lm -lc -lgcc -o $@
	$(PREFIX)size $@

$(BUILD)/%.o: $(SRC)/%.c | $(BUILD)
	$(PREFIX)gcc $(CFLAGS) -c $< -o $@
$(BUILD)/doomgeneric_rv32.o: doomgeneric_rv32.c | $(BUILD)
	$(PREFIX)gcc $(CFLAGS) -c $< -o $@
$(BUILD)/draw_fast.o: draw_fast.c | $(BUILD)
	$(PREFIX)gcc $(CFLAGS) -c $< -o $@
ifneq ($(FAST_DRAW),)
# The stock loops are renamed out of the way.
$(BUILD)/r_draw.o: CFLAGS += -DR_DrawColumn=R_DrawColumn_stock -DR_DrawSpan=R_DrawSpan_stock
endif
$(BUILD)/syscalls.o: ../runtime/syscalls.c ../runtime/rv32.h | $(BUILD)
	$(PREFIX)gcc $(ARCH) $(OPT) -g -c $< -o $@
$(BUILD)/crt0.o: ../runtime/crt0.S | $(BUILD)
	$(PREFIX)gcc $(ARCH) -c $< -o $@
$(BUILD):
	mkdir -p $@

clean:
	rm -rf build_*
