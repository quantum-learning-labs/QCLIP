import re
from tqdm import tqdm

TOKEN_RE = re.compile(r"[a-z0-9']+")

def tokenize(text):
    # word-level tokens: lower-cased runs of letters, digits and apostrophes (punctuation dropped)
    return TOKEN_RE.findall(str(text).lower())

class TokenCompiler:
    # model_type microclip: a caption becomes its word tokens (no grammar). For each <label>_diagram column the caption
    # text is the <label> column (corrected_sentence, swapped_sentence, true_caption, false_caption); without it, the
    # diagram's words in box order. Writes <label>_einsum (token tuple, read by the text model) and <label>_symbols.
    def __init__(self):
        self.id = type(self).__name__

    def compile_dataset(self, df, **_):
        for col in [c for c in df.columns if c.endswith('_diagram')]:
            label = col[:-len('_diagram')]
            tokens = []
            for i, diagram in enumerate(tqdm(df[col], desc=f"Tokenising {label}")):
                text = df[label].iloc[i] if label in df.columns else None
                if isinstance(text, str):
                    tokens.append(tuple(tokenize(text)))
                elif isinstance(diagram, list):
                    tokens.append(tuple(t for box in diagram for t in tokenize(box[0])))
                else:
                    tokens.append(float('nan'))
            df[f"{label}_einsum"] = tokens
            df[f"{label}_symbols"] = [list(t) if isinstance(t, tuple) else t for t in tokens]
        return df
