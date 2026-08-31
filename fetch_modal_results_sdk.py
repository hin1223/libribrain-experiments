"""Downloads everything under a Modal volume prefix using the Python SDK
directly, bypassing `modal volume get`'s buggy directory-download path
(hits '[Errno 21] Is a directory' on nearly every directory that mixes
top-level files with nested subdirectories -- which is most of ours).

Usage:
    python fetch_modal_results_sdk.py results modal_backup_results
    python fetch_modal_results_sdk.py checkpoints modal_backup_checkpoints
"""
import os
import sys
import modal

VOLUME_NAME = "libribrain-vol"


def download_prefix(remote_prefix: str, local_dir: str):
    volume = modal.Volume.from_name(VOLUME_NAME)
    os.makedirs(local_dir, exist_ok=True)

    entries = list(volume.listdir(remote_prefix, recursive=True))
    files = [e for e in entries if not e.path.endswith("/") and e.type != 2]
    print(f"Found {len(entries)} entries ({len(files)} files) under {remote_prefix}/")

    ok, failed = 0, []
    for entry in entries:
        path = entry.path
        local_path = os.path.join(local_dir, os.path.relpath(path, remote_prefix))
        try:
            parent = os.path.dirname(local_path)
            # a directory-marker entry processed earlier may have left a stray
            # empty file exactly where a real subdirectory needs to go now
            if os.path.isfile(parent):
                os.remove(parent)
            os.makedirs(parent, exist_ok=True)
            with open(local_path, "wb") as f:
                for chunk in volume.read_file(path):
                    f.write(chunk)
            ok += 1
            if ok % 25 == 0:
                print(f"  ...{ok} files downloaded so far")
        except Exception as e:
            # clean up any stray empty file this attempt may have created
            # (e.g. directory-marker entries fail inside read_file, after
            # open() has already created the file on disk)
            if os.path.isfile(local_path):
                os.remove(local_path)
            print(f"FAILED: {path} ({e})")
            failed.append(path)

    print(f"\nDone: {ok} files downloaded, {len(failed)} failed.")
    if failed:
        print("Failed paths:")
        for p in failed:
            print(" ", p)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    download_prefix(sys.argv[1], sys.argv[2])
