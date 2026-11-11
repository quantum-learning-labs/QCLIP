from modules.utils.general import get_device
from modules.data_pipeline.embeddings import ImgStream, embed_images
from modules.data_pipeline.retrieval import BaseDatasetRetriever
from modules.data_pipeline.datasets import build_swap_set
from pathlib import Path
import argparse, os
import pandas as pd

# python scripts/retrieve.py svo-probes                                         download, parse captions, embed images
# python scripts/retrieve.py svo-probes --swap test -o swap_full.pkl            subject/object swap set from processed/test.pkl
# python scripts/retrieve.py svo-probes --swap train --keep_all -o train_swapneg.pkl   training set + swapped-caption negatives
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs='?', default='aro', help="Dataset name (see retrieval.DATASET_SOURCES)")
    parser.add_argument("--swap", type=str, default=None, help="Only build a swap set from this processed split")
    parser.add_argument("--keep_all", action="store_true", help="Swap set: keep every row, swap_valid marks real swaps")
    parser.add_argument("-o", "--out", type=str, default=None, help="Swap set file name in the processed directory")
    args = parser.parse_args()

    DATASET = args.dataset
    PARSER_PATH = os.environ.get('BOBCAT_PATH', '/cs/research/pplv/comp_bridge/bobcat')   # Bobcat parser model dir
    ROOT_PATH = os.getcwd()
    OUT_DIR = ROOT_PATH + '/data/' + DATASET + '/processed'
    IMG_PATH = ROOT_PATH + '/data/' + DATASET + '/raw/images.zip'
    DEV = get_device()

    if args.swap is not None:
        out_path = Path(OUT_DIR) / (args.out or f"{args.swap}_swap.pkl")
        if out_path.exists():
            raise FileExistsError(f"{out_path} already exists")
        swap_df = build_swap_set(pd.read_pickle(Path(OUT_DIR) / f"{args.swap}.pkl"), keep_all=args.keep_all)
        swap_df.to_pickle(out_path)
        print(f"Swap set written to {out_path}")
    else:
        from modules.symbolic import text_processor
        print(f"Retrieving {DATASET} dataset from {ROOT_PATH}/data...")
        retriever = BaseDatasetRetriever(dataset_name=DATASET, data_root=ROOT_PATH+'/data')
        retriever.retrieve()

        print(f"Processing columns: {', '.join(retriever.text_labels)}...")
        print(f"Backend: {DEV}, Batch size: {64}, Output Directory: {PARSER_PATH}")
        functor = text_processor.TextProcessor(PARSER_PATH, 64, DEV)
        functor.text2diagram(path=OUT_DIR, dataset=retriever.data, text_labels=retriever.text_labels)

        generator = ImgStream(IMG_PATH, file_type='zip')
        embed_images(generator, f'{OUT_DIR}/image_embeddings.pt', device=DEV)
