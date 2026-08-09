{
  description = "Machine Learning";

  inputs.nixpkgs.url = "nixpkgs/nixos-unstable";
  outputs = { self, nixpkgs }:

  let
    system = "x86_64-linux";
    pkgs = import nixpkgs { inherit system; };
    myEnv = pkgs.python3.withPackages (ps: with ps; [
        pandas
        numpy
        matplotlib
        requests
        jupyterlab
        scipy
    ]);
  in {
    devShells.${system}.default = pkgs.mkShell{
      packages = [
        myEnv
      ];
  };
  };
}
