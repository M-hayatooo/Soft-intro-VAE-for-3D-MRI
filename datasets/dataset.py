import pickle
from tqdm import tqdm

import numpy as np
import pandas as pd

import pathlib

# == Base ==
DATA_DIR = pathlib.Path("/data") / "radiology_datas"

# == Dataset ==
ADNI1 = DATA_DIR / "ADNI1"
ADNI2 = DATA_DIR / "JHU-radiology" / "20170509"
ADNI2_2 = DATA_DIR / "JHU-radiology" / "MNI_skull_stripped" / "output"
PPMI = DATA_DIR / "JHU-radiology" / "PPMI"
FourRTNI = DATA_DIR / "JHU-radiology" / "4RTNI"

BLACKLIST_DIR = DATA_DIR / "util" / "lists"

DATA_CSV = {
    "ADNI": DATA_DIR / "JHU-radiology" / "ADNIMERGE.csv",
    "PPMI": DATA_DIR / "JHU-radiology" / "PPMI.csv",
    "4RTNI": FourRTNI / "csv" / "4RTNI_DATA.csv",
}

DATA_DIRS_DICT = {
    "ADNI1": ADNI1,
    "ADNI2": ADNI2,
    "ADNI2-2": ADNI2_2,
    "PPMI": PPMI,
    "4RTNI": FourRTNI / "SkullStripped",
}

DATA_PREFIX_DICT = {
    "fullsize": "fullsize",
    "half": "half_",
}
# == Label Encoder ==
CLASS_MAP = {
    "CN": 0,
    "AD": 1,
    "EMCI": 2,
    "LMCI": 3,
    "MCI": 4,
    "SMC": 5,
    "Control": 6,
    "FControl": 6,
    "PD": 7,
    "SWEDD": 8,
    "Prodromal": 9,
    "CBD": 10,
    "PSP": 11,
    "Oth": 12,
}


def read_voxel(path):
    """
    pathを受け取ってvoxelを返すだけ
    Args
    ----------
    path : pathlib
        pklファイルへのパス
    Return
    ----------
    voxel : numpy.array
        pklファイルの中身
    """
    with open(path, "rb") as rf:
        voxel = pickle.load(rf)
    return np.array(voxel).astype("f")


def get_uid(path):
    """
    pathを受け取ってuidを返すだけ
    Args
    ----------
    path : pathlib
        pklファイルへのパス
    Return
    ----------
    uid : int
        uid
    """
    uid = path.name
    for key, value in DATA_DIRS_DICT.items():
        if str(value) in str(path):

            if key == "ADNI2":
                uid = path.name.split("_")[-2]
                uid = int(uid[1:])

            elif key == "ADNI2-2":
                uid = path.name.split("_")[-4]
                uid = int(uid[1:])

            elif key == "PPMI":
                uid = path.name.split("_")[-4]
                uid = int(uid)

            elif key == "4RTNI":
                uid = path.name.split("_")[-4]
                uid = int(uid)

            return uid


def collect_pids(dirs):
    """
    ディレクトリ内に存在するpatiantを集める
    Args
    ----------
    path : pathlib
        pklファイルへのパス
    Return
    ----------
    pid : list of str
        pids
    """
    patiants = []
    for dir_path in dirs:
        [patiants.append(f.name) for f in dir_path.iterdir()]
    return patiants


def get_blacklist():
    """
    brain/util/listsの中にいるblacklistたちをuidのリストで返す
    Args
    ----------
    Return
    ----------
    uid : list of int
        uids
    """
    key = "**/uids.txt"
    excluded_uid_paths = BLACKLIST_DIR.glob(key)
    excluded_uids = []
    for path in excluded_uid_paths:
        with open(path, "r") as rf:
            [excluded_uids.append(int(uid.rstrip("\n"))) for uid in rf]
    return excluded_uids


def load_csv_data(pids):

    df = pd.read_csv(DATA_CSV["ADNI"])
    adni = df[["PTID", "AGE", "PTGENDER"]]
    adni.columns = ["PID", "AGE", "SEX"]

    df = pd.read_csv(DATA_CSV["PPMI"])
    ppmi = df[["Subject", "Age", "Sex"]]
    ppmi.columns = ["PID", "AGE", "SEX"]

    df = pd.read_csv(DATA_CSV["4RTNI"])
    fourrtni = df[["SUBID", "AGE_AT_TP0", "SEX"]]
    fourrtni.columns = ["PID", "AGE", "SEX"]

    df = adni.append(ppmi).append(fourrtni)
    df.iloc[:, 2] = df["SEX"].apply(lambda x: x[0] if x in ["Male", "Female"] else x)
    df.iloc[:, 1] = df["AGE"].apply(lambda x: int(x))
    df.iloc[:, 0] = df["PID"].apply(lambda x: str(x))

    return df


def load_data(
    kinds=["ADNI2", "ADNI2-2", "PPMI", "4RTNI"],
    classes=[
        "CN",
        "AD",
        "MCI",
        "EMCI",
        "LMCI",
        "SMC",
        "Control",
        "PD",
        "SWEDD",
        "Prodromal",
        "PSP",
        "CBD",
        "Oth",
        "FControl",
    ],
    size="half",
    csv=False,
    pids=[],
    uids=[],
    unique=False,
    blacklist=False,
    dryrun=False,
):
    """
    Args
    ----------
    kind : list
        ADNI2, ADNI2-2, PPMI をリストで指定
    classes : list
        CN, AD, MCI, EMCI, LMCI, SMC,
        Control, PD, SWEDD, Prodromal,
        PSP, CBD, Oth,
        をリストで指定
    size    : str
        fullsize, half
    pids    : list of str
        取得したい患者のpidをリストで指定
    uids    : list of str
        取得したい患者のuidをリストで指定
    unique  : bool
    blacklist  : bool
    dryrun  : bool
        trueの場合にvoxelを読み込まないでその他の情報だけ返す
    Return
    ----------
    dataset: list
        情報がいっぱい詰まったリストだよ
    """
    dirs = []
    for key in kinds:
        for c in classes:
            dirname = DATA_DIRS_DICT[key].resolve() / c
            if dirname.exists():
                dirs.append(DATA_DIRS_DICT[key].resolve() / c)

    dataset = []
    key = "**/*" + DATA_PREFIX_DICT[size] + "*.pkl"
    if dryrun:
        print(f"[--DRYRUN--]")
        print(f"[SIZE] {size}")
        print(f"[KINDS] {kinds}")
        print(f"[CLASSES] {classes}")
        print(f"[PATIANT] {len(pids)} of patiants")
        print(f"[TARGET] {uids}")
        print(f"[UNIQUE] {unique}")
        print(f"[BLACKLIST] {blacklist}")

    for dir_path in dirs:
        for file_path in dir_path.glob(key):
            data = {}
            data["uid"] = get_uid(file_path)
            data["pid"] = file_path.parent.name
            data["label"] = dir_path.name
            data["nu_label"] = CLASS_MAP[dir_path.name]
            data["path"] = file_path
            dataset.append(data)

    if uids:
        dataset = [data for data in dataset if data["uid"] in uids]

    if unique:
        dataset_unique = []
        for pid in collect_pids(dirs):
            # pidごとにdataを取り出しそれらのuidをソートして最新のものを選択
            dataset_unique.append(
                sorted(
                    [data for data in dataset if data["pid"] == pid],
                    key=lambda data: data["uid"],
                )[-1]
            )
        dataset = dataset_unique

    if pids:
        dataset = [data for data in dataset if data["pid"] in pids]

    if blacklist:
        exclude_uids = get_blacklist()
        dataset = [data for data in dataset if data["uid"] not in exclude_uids]

    if dryrun:
        return np.array(dataset)

    if csv:
        df = load_csv_data([data["pid"] for data in dataset])
        [
            data.update(
                AGE=df[df.PID == data["pid"]].AGE.values[0],
                SEX=df[df.PID == data["pid"]].SEX.values[0],
            )
            if data["pid"] in df.PID.values
            else data.update(AGE=None, SEX=None,)
            for data in dataset
        ]

    [data.update(voxel=read_voxel(data["path"])) for data in tqdm(dataset, leave=False)]

    return np.array(dataset)