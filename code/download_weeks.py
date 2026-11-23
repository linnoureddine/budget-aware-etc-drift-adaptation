import csv, os, shutil, sys, time, datetime, platform
from pathlib import Path
from remotezip import RemoteZip

URL = "https://zenodo.org/records/10608607/files/CESNET-TLS-Year22.zip?download=1"
# same environment variables as common.py
RAW = Path(os.environ.get("CESNET_RAW", r"C:\cesnet_raw"))
LOG = Path(os.environ.get("CESNET_OUT", ".")) / "logs" / f"download_log_{platform.node()}.csv"

# downloads only the weeks we need from the zip on Zenodo
# default is chain 1 then chain 2, or pass weeks: python download_weeks.py 18
DEFAULT_WEEKS = [18, 22, 23, 24, 25, 26, 27, 28, 29, 30, 17,
                 36, 40, 41, 42, 43, 44, 45, 46, 47, 48, 35]
WEEKS = [int(a) for a in sys.argv[1:]] or DEFAULT_WEEKS

def log(week, name, size):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "week", "file", "bytes"])
        w.writerow([datetime.datetime.now().isoformat(timespec="seconds"), week, name, size])

def download_week(week):
    tag = f"WEEK-2022-{week:02d}"
    with RemoteZip(URL, timeout=300, support_suffix_range=False) as z:
        members = [i for i in z.infolist()
                   if f"/{tag}/" in i.filename and not i.is_dir()]
        print(f"{tag}: {len(members)} files")
        for info in members:
            dest = RAW / tag / Path(info.filename).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and dest.stat().st_size == info.file_size:
                continue
            tmp = dest.with_name(dest.name + ".part")
            with z.open(info) as src, open(tmp, "wb") as out:
                shutil.copyfileobj(src, out, length=1024 * 1024)
            if tmp.stat().st_size != info.file_size:
                raise RuntimeError(f"Size mismatch: {dest.name}")
            tmp.replace(dest)
            log(tag, dest.name, info.file_size)
            print(f"  {dest.name}  {info.file_size/1024**2:.1f} MB")

for week in WEEKS:
    for attempt in range(1, 6):
        try:
            download_week(week)
            break
        except Exception as e:
            print(f"  Week {week} attempt {attempt} failed: {e}")
            time.sleep(60)
    else:
        print(f"  WEEK {week} FAILED; run the script again later")

print(f"\nDone. Free space on {RAW.anchor or RAW}: {shutil.disk_usage(RAW.anchor or RAW).free/1024**3:.1f} GB")