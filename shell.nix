{ pkgs ? import <nixpkgs> {} }:
let
  # Triton's bundled ptxas is a generic Linux binary; start it with Nix's loader.
  ptxas = pkgs.writeShellScript "ptxas" ''
    exec ${pkgs.stdenv.cc.bintools.dynamicLinker} "$TRITON_BUNDLED_PTXAS" "$@"
  '';
in
pkgs.mkShell {
  packages = [ pkgs.python3 pkgs.python3Packages.pip pkgs.ruff ];
  # /run/opengl-driver/lib holds the NVIDIA driver's libcuda for PyTorch and Triton.
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ] + ":/run/opengl-driver/lib";
  TRITON_LIBCUDA_PATH = "/run/opengl-driver/lib";
  TRITON_PTXAS_PATH = "${ptxas}";
  # Triton compiles a small C launcher against Python's headers.
  C_INCLUDE_PATH = "${pkgs.python3}/include/${pkgs.python3.libPrefix}";
  shellHook = ''
    export TRITON_BUNDLED_PTXAS="$PWD/.venv/lib/${pkgs.python3.libPrefix}/site-packages/triton/backends/nvidia/bin/ptxas"
  '';
}
