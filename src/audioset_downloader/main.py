import io
import sys
from yt_dlp import YoutubeDL 
import pandas as pd
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
import click
from tqdm import tqdm
from contextlib import redirect_stderr
import os

__all__ = [
    "audioset_dl",
    "print_classes"
]

root = os.path.abspath(os.path.dirname(__file__))


@click.command()
@click.option("--output-dir", "-o", type=str, default="./",
              help="Target directory for the downloads (default='./')")
@click.option("--class-name", "-c", type=str, default=["Snoring"], multiple=True,
              help="Name of the class to download (default=Snoring). Can be repeated to select intersection of classes.")
@click.option("--class-union", "-u", is_flag=True,
              help="Toggle whether class names should intersect (default) or use union")
@click.option("--mixed", "-m", is_flag=True,
              help="Download examples with possibly more than one class label (default: False)")
@click.option("--exclude-eval-set", "-xe", is_flag=True,
              help="Exclude examples from the eval set (default=False)")
@click.option("--exclude-balanced-set", "-xb", is_flag=True,
              help="Exclude examples from the balanced set (default=False)")
@click.option("--exclude-unbalanced-set", "-xu", is_flag=True,
              help="Exclude examples from the unbalanced set (default=False)")
@click.option("--n-examples", "-n", type=int, default=None,
              help="Number of examples to download (default=all matching)")
@click.option("--full-source", "-f", is_flag=True,
              help="Download full examples instead of 10-second segments (default=False)")
@click.option("--most-viewed", "-mv", is_flag=True,
              help="If set with --n-examples, download most viewed examples")
@click.option("--most-liked", "-ml", is_flag=True,
              help="If set with --n-examples, download most liked examples")
@click.option("--cookies-file", "-cf", type=str, default=None,
              help="Path to cookies.txt file to access age-restricted or private videos")
def download_cli(*args, **kwargs):
    """Download examples of a specific class from Google's AudioSet"""
    audioset_dl(*args, **kwargs)


@click.command()
def print_classes():
    with open(os.path.join(root, "class_names.txt"), "r") as f:
        cls = f.read()
    print(cls)


def audioset_dl(
        output_dir="outputs",
        class_name=("Snoring",),
        class_union=False,
        mixed=False,
        exclude_eval_set=False,
        exclude_balanced_set=False,
        exclude_unbalanced_set=False,
        n_examples=None,
        full_source=False,
        most_viewed=False,
        most_liked=False,
        cookies_file=None,
):
    output_dir = os.path.abspath(output_dir)
    ontology = pd.read_json(os.path.join(root, "ontology.json"))
    audioset = pd.DataFrame()
    if not exclude_eval_set:
        eval_ = pd.read_csv(os.path.join(root, "csv", "eval_segments.csv"), header=2, quotechar='"', skipinitialspace=True)
        eval_["dir"] = "eval"
        audioset = pd.concat((audioset, eval_))
    if not exclude_balanced_set:
        balanced = pd.read_csv(os.path.join(root, "csv", "balanced_train_segments.csv"), header=2, quotechar='"', skipinitialspace=True)
        balanced["dir"] = "balanced"
        audioset = pd.concat((audioset, balanced))
    if not exclude_unbalanced_set:
        unbalanced = pd.read_csv(os.path.join(root, "csv", "unbalanced_train_segments.csv"), header=0, quotechar='"', skipinitialspace=True)
        unbalanced["dir"] = "unbalanced"
        audioset = pd.concat((audioset, unbalanced))

    if mixed or len(class_name) > 1:
        cls_id = []
        for name in class_name:
            cls_id += [ontology.id[ontology.name.str.fullmatch(name)].item()]
        sep = '||' if class_union else ''
        regex = sep.join(f"(?=.*{w})" for w in cls_id)
        regex = r'{}'.format(regex)
        cls_exmp = audioset.positive_labels.str.contains(regex, regex=True)
    else:
        cls_id = ontology.id[ontology.name.str.fullmatch(class_name[0])].item()
        cls_exmp = audioset.positive_labels.str.fullmatch(cls_id)

    def _download(exmp):
        if not mixed and exmp.positive_labels != cls_id:
            return
        start, end = dt.timedelta(seconds=exmp.start_seconds), dt.timedelta(seconds=exmp.end_seconds)
        
        opts = {
            "quiet": True,
            "ignoreerrors": True,
            "format": "bestaudio",
            "outtmpl": f"{output_dir}/{exmp.dir}/%(title)s.%(ext)s",
            "external_downloader": "ffmpeg",
            "external_downloader_args": [*(("-ss", str(start), "-to", str(end)) if not full_source else tuple()),
                                         "-loglevel", "panic"]
        }

        if cookies_file:
            opts["cookiefile"] = cookies_file

        dler = YoutubeDL(opts)
        try:
            with redirect_stderr(io.StringIO()):
                dler.extract_info(exmp._1)
        except KeyboardInterrupt:
            raise
        except Exception:
            pass

    pool = ThreadPoolExecutor(max_workers=200)
    subset = audioset[cls_exmp]
    if n_examples is not None:
        if most_viewed:
            subset = subset.sort_values("views", ascending=False).iloc[:n_examples]
        elif most_liked:
            subset = subset.sort_values("likes", ascending=False).iloc[:n_examples]
        else:
            subset = subset.sample(n=n_examples)

    futures = [pool.submit(_download, exmp) for exmp in subset.itertuples()]
    for _ in tqdm(as_completed(futures, timeout=None), total=cls_exmp.sum() if n_examples is None else n_examples,
                  file=sys.stdout, ncols=88):
        continue


if __name__ == '__main__':
    audioset_dl(
        output_dir="./",
        class_name=(
            "Electronic music",
            "Techno",
            "House music",
            "Dubstep",
            "Electro",
            "Oldschool jungle",
            "Electronica",
            "Electronic dance music",
            "Trance music",
        ),
        class_union=True,
        exclude_unbalanced_set=False,
        most_viewed=True,
        n_examples=20,
        cookies_file="mis_cookies.txt"  # Aquí usas el nombre que quieras
    )
