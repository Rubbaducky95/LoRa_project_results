import os
import re
import shutil


ROOT = r"C:\Users\ruben\Documents\LoRa Project\test_data2"
SF_FOLDER_RE = re.compile(r"^SF(\d+)$")
CSV_RE = re.compile(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$")


def main():
    moved = 0
    already = 0

    for dn in sorted(os.listdir(ROOT)):
        dist_path = os.path.join(ROOT, dn)
        if not (os.path.isdir(dist_path) and dn.startswith("distance_")):
            continue

        # Ensure SF subfolders exist.
        for sf in range(7, 13):
            os.makedirs(os.path.join(dist_path, f"SF{sf}"), exist_ok=True)

        # Move top-level CSV files into SF folders.
        for name in os.listdir(dist_path):
            src = os.path.join(dist_path, name)
            if not os.path.isfile(src):
                continue
            m = CSV_RE.match(name)
            if not m:
                continue
            sf = int(m.group(1))
            target_dir = os.path.join(dist_path, f"SF{sf}")
            dst = os.path.join(target_dir, name)
            if os.path.abspath(src) == os.path.abspath(dst):
                already += 1
                continue
            if os.path.exists(dst):
                os.remove(dst)
            shutil.move(src, dst)
            moved += 1

        # If there are nested CSVs not in correct SF folder, normalize them.
        for sub in os.listdir(dist_path):
            sub_path = os.path.join(dist_path, sub)
            if not os.path.isdir(sub_path):
                continue
            if SF_FOLDER_RE.match(sub) is None:
                continue
            for name in os.listdir(sub_path):
                src = os.path.join(sub_path, name)
                if not os.path.isfile(src):
                    continue
                m = CSV_RE.match(name)
                if not m:
                    continue
                sf = int(m.group(1))
                correct_dir = os.path.join(dist_path, f"SF{sf}")
                dst = os.path.join(correct_dir, name)
                if os.path.abspath(src) == os.path.abspath(dst):
                    already += 1
                    continue
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)
                moved += 1

    print(f"Moved files: {moved}")
    print(f"Already in place: {already}")


if __name__ == "__main__":
    main()

