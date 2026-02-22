import subprocess, shlex, os, pathlib


def run_docker(cmd: str, finn_cfg: dict, build_dir: str, name: str = None) -> int:
    volumes = finn_cfg["path"]["VOLUMES"]
    vols = [volumes] if isinstance(volumes, str) else list(volumes)
    tmp = finn_cfg["path"].get("tmp")
    if tmp:
        vols.append(tmp)
    mounts = " ".join(f"-v {shlex.quote(v)}:{shlex.quote(v)}" for v in vols)
    
    docker_cmd = f"""\
set -euo pipefail

export FINN_XILINX_PATH={shlex.quote(finn_cfg["path"]["FINN_XILINX_PATH"])}
export FINN_XILINX_VERSION={shlex.quote(finn_cfg["path"]["FINN_XILINX_VERSION"])}
export FINN_DOCKER_EXTRA="{f'--name {name} ' if name else ''}{mounts} -w {shlex.quote(finn_cfg["path"]["CWD"])} -e PYTHONBREAKPOINT=0"
{f'export FINN_HOST_BUILD_DIR={shlex.quote(tmp)}' if tmp else ''}

cd {shlex.quote(finn_cfg["path"]["FINN_PATH"])}
./run-docker.sh {cmd}
"""
    out_log = build_dir / pathlib.Path(f"docker.out.log"); out_log.parent.mkdir(parents=True, exist_ok=True)
    err_log = build_dir / pathlib.Path(f"docker.err.log"); err_log.parent.mkdir(parents=True, exist_ok=True)
    with open(out_log or os.devnull, "w", encoding="utf-8") as fo, open(err_log or os.devnull, "w", encoding="utf-8") as fe:
        proc = subprocess.run(["bash", "-s"], input=docker_cmd, text=True, stdout=fo, stderr=fe)

    return proc.returncode
