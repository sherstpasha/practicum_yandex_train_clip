import gradio as gr
import requests
import base64
from io import BytesIO
from PIL import Image


API_URL = "http://127.0.0.1:8001/search"


last_query = {"value": ""}


def perform_search(query, top_k):
    if not query.strip():
        return []

    payload = {"query": query, "top_k": int(top_k)}

    try:
        response = requests.post(API_URL, json=payload)
        data = response.json()

        results = data.get("results", [])
        search_time = data.get("search_time", 0)

        print(
            f"Поиск '{query}' | Топ-{top_k} | Время: {search_time:.2f} мс | Найдено: {len(results)}"
        )

        # Декодируем base64 изображения
        gallery_items = []
        for item in results:
            img_data = base64.b64decode(item["image_base64"])
            img = Image.open(BytesIO(img_data))
            caption = f"{item['display_name']}\n{item['description'][:100]}..."
            gallery_items.append((img, caption))

        return gallery_items

    except requests.exceptions.ConnectionError:
        print(
            "Ошибка: Не удается подключиться к серверу. Убедитесь, что сервер запущен."
        )
        return []
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        return []


def on_query_change(query, top_k):
    if not query:
        last_query["value"] = ""
        return []

    stripped_query = query.strip()

    if query.endswith(" ") and stripped_query and stripped_query != last_query["value"]:
        last_query["value"] = stripped_query
        return perform_search(stripped_query, int(top_k))

    return []


light_css = """
<style>
body, .gradio-container {
    background: white !important;
    color: black !important;
}

input, textarea {
    background: white !important;
    color: black !important;
    border: 1px solid #ccc !important;
}

.gr-button {
    background: #e6e6e6 !important;
    color: #000 !important;
    border: 1px solid #bbb !important;
}

.gr-button:hover {
    background: #dcdcdc !important;
}
</style>
"""


with gr.Blocks(title="CLIP Product Search") as demo:
    gr.HTML(light_css)

    gr.Markdown("<h1 style='color:#000;'>CLIP Product Search</h1>")
    gr.Markdown("Enter a product description and press space for automatic search")

    with gr.Row():
        with gr.Column(scale=4):
            query_input = gr.Textbox(
                placeholder="For example: red skirt, blue sunglasses, mickey mouse...",
                label="Search Query",
                lines=1,
            )
        with gr.Column(scale=1):
            top_k_slider = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                label="Top-K",
                info="Number of results",
            )

    search_btn = gr.Button("Search", variant="primary")

    gallery = gr.Gallery(
        label="Search Results",
        show_label=True,
        columns=5,
        rows=1,
        height="auto",
        object_fit="contain",
    )

    search_btn.click(
        fn=perform_search, inputs=[query_input, top_k_slider], outputs=gallery
    )

    query_input.change(
        fn=on_query_change, inputs=[query_input, top_k_slider], outputs=gallery
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)
