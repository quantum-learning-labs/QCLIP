from pathlib import Path
from modules.config import parse_args, setup_exp
from modules.data_pipeline.embeddings import ImgStream, embed_images
from modules.data_pipeline.retrieval import BaseDatasetRetriever
from modules.symbolic.text_processor import TextProcessor

# python scripts/retrieve.py -cfg configs/vqc.yaml: download the config's dataset, parse its captions into diagrams
# (Bobcat model directory from $BOBCAT_PATH, default ./bobcat) and embed its images into the config's img_path, under
# data/<dataset>/. Datasets without an official split are stored whole (processed/all.pkl); configs split them when loaded.
if __name__ == '__main__':
    args = parse_args("Download and process the dataset of a config")
    if args.config is None:
        raise ValueError("retrieve.py needs a config (-cfg)")
    config, DEV, _ = setup_exp(args.config, seed=args.seed, overrides=args.set)
    dataset = config['dataset']['name']
    data_root = Path.cwd() / 'data'
    out_dir = data_root / dataset / 'processed'

    print(f"Retrieving {dataset} into {data_root}...")
    retriever = BaseDatasetRetriever(dataset_name=dataset, data_root=data_root)
    retriever.retrieve()

    print(f"Processing columns: {', '.join(retriever.text_labels)} | Backend: {DEV}")
    TextProcessor(batch_size=64, device=DEV).text2diagram(path=out_dir, dataset=retriever.data, text_labels=retriever.text_labels)
    embed_images(ImgStream(retriever.images_zip, file_type='zip'), config['dataset']['train']['img_path'], device=DEV)
