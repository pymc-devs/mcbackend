"""
This script regenerates the `meta.py` module
from protobuf definitions of core metadata structures.
"""
import pathlib
import shutil
import subprocess

DP_HERE = pathlib.Path(__file__).parent.absolute()
DP_ROOT = DP_HERE.parent
DP_PROJ = DP_ROOT / "mcbackend"
FP_PROTO = DP_HERE / "meta.proto"


def run():
    if not FP_PROTO.exists():
        raise FileNotFoundError(f"Couldn't find {FP_PROTO}.")

    subprocess.check_call(
        ["protoc", "-I", str(DP_HERE), "--python_betterproto_out=" + str(DP_HERE), str(FP_PROTO)]
    )
    for fp_in, fp_out in [
        (DP_HERE / "__init__.py", DP_PROJ / "meta.py"),
        (DP_HERE / "npproto" / "__init__.py", DP_PROJ / "npproto" / "__init__.py"),
    ]:
        shutil.move(str(fp_in), str(fp_out))
    return


if __name__ == "__main__":
    run()
