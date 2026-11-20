from lambeq import BobcatParser
from tqdm import tqdm
import os, pickle
import pandas as pd
from pathlib import Path

from modules.symbolic.grammar import strip_sent
from modules.symbolic.functor import tree2einsum
from modules.symbolic.root import apply_root

class TextProcessor:
    def __init__(self, parser_path: str | None = None, batch_size: int = 32, device = 'cpu'):
        self.parser_path = parser_path or os.getenv("BOBCAT_PATH", "./bobcat")   # Bobcat parser model directory
        self.parser = BobcatParser(model_name_or_path=self.parser_path, cache_dir=self.parser_path, device=device, batch_size=batch_size)

    def text2diagram(self, path, dataset, text_labels: list[str]):
        # path should be data_root / dataset_name / 'processed'
        if not isinstance(path, Path):
            path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for split, data in dataset.items():
            print(f"Processing {split} split with {len(data)} rows...")
            if not isinstance(data, pd.DataFrame):
                df = data.to_pandas()
            else:
                df = data
            
            for label in text_labels:
                new_label = f"{label}_diagram"
                sentence_arr = []
                row_mapping = []

                first_entry = df[label].iloc[0]
                is_arr = hasattr(first_entry, '__iter__') and not isinstance(first_entry, (str, bytes))
                for idx, entry in enumerate(df[label]):
                    captions = entry if is_arr else [entry]
                    for caption in captions:
                        sentence_arr.append(strip_sent(caption))
                        row_mapping.append(idx)
                    
                tree_arr = self.parser.sentences2trees(sentence_arr, suppress_exceptions=True)
                processed_tns_nested = [[] for _ in range(len(df))]
                miss_count = 0

                for flat_idx, tree in tqdm(enumerate(tree_arr), total=len(tree_arr)):
                    row_idx = row_mapping[flat_idx]
                    if tree is not None:
                        try:
                            processed_tree = tree._resolved().collapse_noun_phrases()
                            tn = tree2einsum(processed_tree, simplify=False)
                            apply_root(tn, 'word')   # rows whose root cannot be resolved are dropped, as before
                            processed_tns_nested[row_idx].append(tn)   # stored unrooted: text.root is applied when compiled
                        except Exception as e:
                            miss_count += 1
                            print(f"Error normalizing tree at flat index {flat_idx}: {e}")
                final_tns = []
                for tn_list in processed_tns_nested:
                    if not tn_list: 
                        final_tns.append(None)
                    else:
                        final_tns.append(tn_list if is_arr else tn_list[0])
                df[new_label] = final_tns

            start_len = len(df)
            df = df.dropna().reset_index(drop=True)
            print(f"[{split}] Dropped {start_len - len(df)} unparseable rows out of {len(df)} and {miss_count} parsing errors.")

            fpath = path / f"{split}.pkl"
            with open(fpath, "wb") as f:
                pickle.dump(df, f)