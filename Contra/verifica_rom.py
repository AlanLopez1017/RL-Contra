import os


def check_nes_rom_header(rom_path):
    with open(rom_path, "rb") as f:
        header = f.read(16)
        if header[:4] != b"NES\x1a":
            return "❌ No es una ROM válida iNES (.nes)"
        else:
            prg_rom_size = header[4]
            chr_rom_size = header[5]
            return f"✅ ROM válida. PRG ROM: {prg_rom_size * 16}KB, CHR ROM: {chr_rom_size * 8}KB"


if __name__ == "__main__":
    rom = "contra.nes"
    rom_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), rom)
    print("Verificando ROM en:", rom_path)
    print(check_nes_rom_header(rom_path))
