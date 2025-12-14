from fastapi import FastAPI
from pydantic import BaseModel
from retrieval_system import ProductSearchSystem

search_system = ProductSearchSystem(
    checkpoint_path="../clip_best.pth",
    index_path="../clip_image.index",
    data_path="../archive",
)

app = FastAPI()


class SearchQuery(BaseModel):
    query: str
    top_k: int = 5


@app.get("/")
def root():
    return {"status": "ok", "message": "Product search server running"}


@app.post("/search")
def search(q: SearchQuery):
    results, search_time = search_system.search(q.query, top_k=q.top_k)

    print(f"Search '{q.query}' | Top-{q.top_k} | Time: {search_time:.2f} ms")

    return {
        "results": results,
        "search_time": search_time,
        "query": q.query,
        "top_k": q.top_k,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)
