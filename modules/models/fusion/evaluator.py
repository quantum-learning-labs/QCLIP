import torch

def eval_img_choice(image_model, text_model, dataloader, device):
    image_model.eval()
    text_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            m1_emb = text_model(batch["caption"]).flatten(1).contiguous()
            pos_m2_emb = image_model(batch["pos_image"].to(device)).flatten(1).contiguous()
            neg_m2_emb = image_model(batch["neg_image"].to(device)).flatten(1).contiguous()
            pos_score = torch.sum(m1_emb.conj() * pos_m2_emb, dim=1).abs()
            neg_score = torch.sum(m1_emb.conj() * neg_m2_emb, dim=1).abs()
            correct += (pos_score > neg_score).sum().item()
            total += len(batch['caption'])
    return correct / total

def eval_txt_choice(image_model, text_model, dataloader, device):
    image_model.eval()
    text_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            m1_emb = image_model(batch["image"].to(device)).flatten(1).contiguous()
            pos_m2_emb = text_model(batch["pos_caption"]).flatten(1).contiguous()
            neg_m2_emb = text_model(batch["neg_caption"]).flatten(1).contiguous()
            pos_score = torch.sum(m1_emb.conj() * pos_m2_emb, dim=1).abs()
            neg_score = torch.sum(m1_emb.conj() * neg_m2_emb, dim=1).abs()
            correct += (pos_score > neg_score).sum().item()
            total += len(batch['image'])
    return correct / total