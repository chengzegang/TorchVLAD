import click
from . import vlad
import torch


@click.command()
@click.option("--train-dir", required=True, help="Path to training data")
@click.option("--index-dir", help="Path to index data")
@click.option("--test-dir", help="Path to test data")
@click.option("--db-dir", default="db", help="Path to database")
@click.option("--cahce-dir", default="cache", help="Path to cache")
@click.option("--image-shape", default=(256, 256), help="Image shape")
@click.option("--batch-size", default=32, help="Batch size")
@click.option("--device", default="cuda", help="Device to use")
@click.option("--num-workers", default=4, help="Number of workers")
@click.option("--num-clusters", default=128, help="Number of clusters")
@click.option("--num-features", default=128, help="Number of features")
@click.option("--patch-size", default=32, help="Patch size")
@click.option("--angle-bins", default=8, help="Angle bins")
@click.option("--spatial-bins", default=4, help="Spatial bins")
def run(**kwargs):
    if kwargs["index_dir"] is None:
        kwargs["index_dir"] = kwargs["train_dir"]

    if kwargs["test_dir"] is None:
        kwargs["test_dir"] = kwargs["index_dir"]

    vlad.train(**kwargs)
    index_df, _ = vlad.index(**kwargs)
    res_mat, res_df = vlad.test(**kwargs, index_df=index_df)
    torch.save(res_mat.cpu(), "retrivals.pt")
    res_df.to_csv("retrivals.csv")
    print("Done")
