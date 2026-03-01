import os
import shutil

import fill_from_raw_test_data as gen


def main():
    src = r"C:\Users\ruben\Documents\LoRa Project\raw_test_data"
    dst = r"C:\Users\ruben\Documents\LoRa Project\test_data2"

    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    # Reuse the same raw-based modeling logic.
    gen.RAW_DIR = src
    gen.TARGET_DIR = dst

    cfg_by_distance, time_by_sfbw = gen.build_raw_models()
    expected = set(gen.expected_configs())  # 66 expected configs (after exclusions)

    generated = 0
    removed_extras = 0

    for _, folder_path, distance in gen.list_distance_folders(dst):
        # Remove out-of-scope configs so folder follows expected config space.
        for filename in list(os.listdir(folder_path)):
            parsed = gen.parse_config_filename(filename)
            if not parsed:
                continue
            if parsed not in expected:
                os.remove(os.path.join(folder_path, filename))
                removed_extras += 1

        # Generate only truly missing files; keep existing originals unchanged.
        for sf, bw, tp in sorted(expected):
            filename = f"SF{sf}_BW{bw}_TP{tp}.csv"
            filepath = os.path.join(folder_path, filename)
            if os.path.exists(filepath):
                continue

            gen.generate_file(
                filepath=filepath,
                distance=distance,
                sf=sf,
                bw=bw,
                tp=tp,
                cfg_by_distance=cfg_by_distance,
                time_by_sfbw=time_by_sfbw,
            )
            generated += 1

    print(f"Created fresh dataset at: {dst}")
    print(f"Removed out-of-scope files: {removed_extras}")
    print(f"Generated missing files: {generated}")


if __name__ == "__main__":
    main()

