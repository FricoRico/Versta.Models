"""Clear the executable-stack bit on PyMNN's native library after install.

The `mnn` 3.6.1 pip wheel ships `_mnncengine*.so` with `PT_GNU_STACK` marked
executable; the system loader refuses it ("cannot enable executable stack").
Flipping the bit in the ELF header is equivalent to `execstack -c` without a
host binutils install. Run once after each `uv sync`:

  uv run python -m versta.train.glyphmatte.fix_mnn_execstack
"""

import struct

from pathlib import Path


def patch(libs_root: Path) -> int:
    """Zeroes the X bit on every PT_GNU_STACK program header of the MNN engine
    library inside the given site-packages root.

    Args:
        libs_root (Path): The site-packages directory containing the .so.

    Returns:
        int: How many segments were changed (0 when already clean).
    """
    hits = list(libs_root.rglob("_mnncengine*.so"))
    assert hits, f"_mnncengine*.so not found under {libs_root}"
    total = 0
    for so in hits:
        b = bytearray(so.read_bytes())
        assert b[:4] == b"\x7fELF"
        (e_phoff,) = struct.unpack_from("<Q", b, 32)
        e_phentsize, e_phnum = struct.unpack_from("<HH", b, 54)
        for i in range(e_phnum):
            off = e_phoff + i * e_phentsize
            (p_type, p_flags) = struct.unpack_from("<II", b, off)
            if p_type == 0x6474E551 and p_flags & 1:  # PT_GNU_STACK + PF_X
                struct.pack_into("<I", b, off + 4, p_flags & ~1)
                total += 1
        so.write_bytes(b)
    return total


if __name__ == "__main__":
    # MNN itself may be unimportable pre-patch; resolve site-packages directly.
    import sysconfig

    site = Path(sysconfig.get_paths()["purelib"])
    n = patch(site)
    print(f"patched {n} PT_GNU_STACK segment(s) under {site}")
