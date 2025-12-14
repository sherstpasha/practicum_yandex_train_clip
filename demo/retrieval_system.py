import time
import torch
import faiss
import pandas as pd
import base64
from io import BytesIO
from typing import List, Dict, Tuple
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import os


class ProductSearchSystem:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        checkpoint_path: str = "clip_best.pth",
        index_path: str = "clip_image.index",
        data_path: str = "archive",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        self.model.eval()

        self.df = pd.read_csv(f"{data_path}/data.csv")
        self.df = self.df.drop_duplicates(subset=["description"])
        self.df = self.df.dropna()
        self.df = self.df.reset_index(drop=True)

        self.index = faiss.read_index(index_path)
        self.data_path = data_path

    def search(
        self, query: str, top_k: int = 5, max_size: int = 400
    ) -> Tuple[List[Dict], float]:
        if not query or not query.strip():
            return [], 0.0

        start_time = time.time()

        self.model.eval()
        with torch.no_grad():
            inputs = self.processor(
                text=[query], return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            text_emb = self.model.get_text_features(**inputs)
            text_emb = text_emb.cpu().numpy()
            faiss.normalize_L2(text_emb)

        _, I = self.index.search(text_emb, top_k)
        top_indices = I[0]

        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            img_path = os.path.join(self.data_path, "data", row["image"])

            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")

                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=90, optimize=True)
                img_base64 = base64.b64encode(buffer.getvalue()).decode()

                results.append(
                    {
                        "image_base64": img_base64,
                        "display_name": row["display name"],
                        "description": row["description"],
                    }
                )

        elapsed_time = (time.time() - start_time) * 1000

        return results, elapsed_time
