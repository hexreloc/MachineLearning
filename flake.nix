{
  description = "Machine Learning environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };

        python = pkgs.python313.withPackages (ps: with ps; [
          numpy
          scipy
          pandas
          matplotlib
          scikit-learn
          torch
          torchvision
          torchaudio
          jupyterlab
          ipykernel
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            python
          ];
        };
      });
}
