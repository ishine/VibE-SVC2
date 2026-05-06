import argparse
import os
import shutil
from glob import glob

from tqdm import tqdm

DATASET_DIR = "dataset/VocalSet"

REMOVE_FLIST = [
    "FULL/female2/arpeggios/fast_piano/fast_piano_arps_f.wav",
    "FULL/female2/arpeggios/lip_trill/lip_trill_arps.wav",
    "FULL/female2/arpeggios/vibrado/slow_vibrato_arps.wav",
    "FULL/female2/scales/straight/f2_scales_straight_u(1).wav",
    "FULL/female2/scales/vibrato/f2_scales_vibrato_a(1).wav",
    "FULL/female2/scales/vocal_fry/scales_vocal_fry.wav",
    "FULL/female3/scales/fast_piano/scales_fast_piano_f.wav",
    "FULL/female9/arpeggios/fast_forte/arps_fast_piano_c.wav",
    "FULL/male3/scales/lip_trill/scales_lip_trill.wav",
    "FULL/male3/arpeggios/fast_piano/arps_c_fast_piano.wav",
]
RENAME_FLIST = {
    "FULL/female2/arpeggios/vibrado/f2_arpeggios_vibrato_a.wav": "FULL/female2/arpeggios/vibrato/f2_arpeggios_vibrato_a.wav",
    "FULL/female2/arpeggios/vibrado/f2_arpeggios_vibrato_e.wav": "FULL/female2/arpeggios/vibrato/f2_arpeggios_vibrato_e.wav",
    "FULL/female2/arpeggios/vibrado/f2_arpeggios_vibrato_i.wav": "FULL/female2/arpeggios/vibrato/f2_arpeggios_vibrato_i.wav",
    "FULL/female2/arpeggios/vibrado/f2_arpeggios_vibrato_o.wav": "FULL/female2/arpeggios/vibrato/f2_arpeggios_vibrato_o.wav",
    "FULL/female2/arpeggios/vibrado/f2_arpeggios_vibrato_u.wav": "FULL/female2/arpeggios/vibrato/f2_arpeggios_vibrato_u.wav",
    "FULL/female4/arpeggios/straight/arpeggios_straight_a.wav": "FULL/female4/arpeggios/straight/f4_arpeggios_straight_a.wav",
    "FULL/female4/arpeggios/straight/arpeggios_straight_e.wav": "FULL/female4/arpeggios/straight/f4_arpeggios_straight_e.wav",
    "FULL/female4/arpeggios/straight/arpeggios_straight_i.wav": "FULL/female4/arpeggios/straight/f4_arpeggios_straight_i.wav",
    "FULL/female4/arpeggios/straight/arpeggios_straight_o.wav": "FULL/female4/arpeggios/straight/f4_arpeggios_straight_o.wav",
    "FULL/female4/arpeggios/straight/arpeggios_straight_u.wav": "FULL/female4/arpeggios/straight/f4_arpeggios_straight_u.wav",
    "FULL/female4/scales/straight/scales_straight_a.wav": "FULL/female4/scales/straight/f4_scales_straight_a.wav",
    "FULL/female4/scales/straight/scales_straight_e.wav": "FULL/female4/scales/straight/f4_scales_straight_e.wav",
    "FULL/female4/scales/straight/scales_straight_i.wav": "FULL/female4/scales/straight/f4_scales_straight_i.wav",
    "FULL/female4/scales/straight/scales_straight_o.wav": "FULL/female4/scales/straight/f4_scales_straight_o.wav",
    "FULL/female4/scales/straight/scales_straight_u.wav": "FULL/female4/scales/straight/f4_scales_straight_u.wav",
    "FULL/male8/excerpts/vibrato/m9_caro_vibrato.wav": "FULL/male8/excerpts/vibrato/m8_caro_vibrato.wav",
    "FULL/male8/excerpts/straight/row_straight.wav": "FULL/male8/excerpts/straight/m8_row_straight.wav",
    "FULL/male10/excerpts/vibrato/caro_vibrato.wav": "FULL/male10/excerpts/vibrato/m10_caro_vibrato.wav",
    "FULL/male10/excerpts/vibrato/row_vibrato.wav": "FULL/male10/excerpts/vibrato/m10_row_vibrato.wav",
    "FULL/male10/excerpts/vibrato/dona_vibrato.wav": "FULL/male10/excerpts/vibrato/m10_dona_vibrato.wav",
    "FULL/male10/scales/fast_piano/scales_c_fast_piano_a.wav": "FULL/male10/scales/fast_piano/m10_scales_c_fast_piano_a.wav",
    "FULL/male10/scales/fast_piano/scales_c_fast_piano_e.wav": "FULL/male10/scales/fast_piano/m10_scales_c_fast_piano_e.wav",
    "FULL/male10/scales/fast_piano/scales_c_fast_piano_i.wav": "FULL/male10/scales/fast_piano/m10_scales_c_fast_piano_i.wav",
    "FULL/male10/scales/fast_piano/scales_c_fast_piano_o.wav": "FULL/male10/scales/fast_piano/m10_scales_c_fast_piano_o.wav",
    "FULL/male10/scales/fast_piano/scales_c_fast_piano_u.wav": "FULL/male10/scales/fast_piano/m10_scales_c_fast_piano_u.wav",
    "FULL/male10/scales/fast_piano/scales_f_fast_piano_a.wav": "FULL/male10/scales/fast_piano/m10_scales_f_fast_piano_a.wav",
    "FULL/male10/scales/fast_piano/scales_f_fast_piano_e.wav": "FULL/male10/scales/fast_piano/m10_scales_f_fast_piano_e.wav",
    "FULL/male10/scales/fast_piano/scales_f_fast_piano_i.wav": "FULL/male10/scales/fast_piano/m10_scales_f_fast_piano_i.wav",
    "FULL/male10/scales/fast_piano/scales_f_fast_piano_o.wav": "FULL/male10/scales/fast_piano/m10_scales_f_fast_piano_o.wav",
    "FULL/male10/scales/fast_piano/scales_f_fast_piano_u.wav": "FULL/male10/scales/fast_piano/m10_scales_f_fast_piano_u.wav",
    "FULL/male1/scales/straight/m1_straight_tone_scales_u.wav": "FULL/male1/scales/straight/m1_scales_straight_u.wav",
}


def raw_file_correction():
    # remove unrelated files
    print("Remove duplicated files...")
    for fpath in tqdm(REMOVE_FLIST):
        if os.path.exists(f"{DATASET_DIR}/{fpath}"):
            os.remove(f"{DATASET_DIR}/{fpath}")

    os.makedirs(f"{DATASET_DIR}/FULL/female2/arpeggios/vibrato", exist_ok=True)
    # rename files
    print("Rename raw files...")
    for src_fpath in tqdm(RENAME_FLIST.keys()):
        target_fpath = RENAME_FLIST[src_fpath]
        if os.path.exists(f"{DATASET_DIR}/{src_fpath}"):
            shutil.copy(f"{DATASET_DIR}/{src_fpath}", f"{DATASET_DIR}/{target_fpath}")
            os.remove(f"{DATASET_DIR}/{src_fpath}")


def flatten_filename(args):
    techniques = ["straight", "vibrato", "belt", "breathy", "vocal_fry"]
    all_files = []
    for tech in techniques:
        all_files += glob(f"{DATASET_DIR}/FULL/*/*/{tech}/*.wav")
    print("Flatten file names...")
    for fpath in tqdm(all_files):
        spk = fpath.split("/")[-4]
        method = fpath.split("/")[-3]
        style = fpath.split("/")[-2]
        key = "nan"
        if method == "excerpts":
            content = os.path.basename(fpath).split("_")[1]
        else:
            content = os.path.splitext(os.path.basename(fpath))[0].split("_")[-1]

        target_fpath = (
            f"{args.raw_dir}/VocalSet#{spk}/{spk}#{method}#{style}#{key}#{content}.wav"
        )

        os.makedirs(f"{args.raw_dir}/VocalSet#{spk}", exist_ok=True)
        shutil.copy(fpath, target_fpath)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw_dir", type=str, default="./dataset/vocalset", help="path to source dir"
    )
    return parser.parse_args()


def main(args):
    raw_file_correction()
    flatten_filename(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
