import gradio as gr
from fastapi import Body, FastAPI

from gen_model import llm_model_infer

app = FastAPI()


@app.get("/")
async def health_check():
    return {"status": "OK"}


@app.post("/chat")
async def chat(user_input: str):
    response = llm_model_infer(user_input)
    return {"response": response}


# Gradio interface
frontend_interface = gr.Interface(
    fn=llm_model_infer,
    inputs="textbox",
    outputs="textbox",
    title="Agent answer",
)

# frontend_interface.launch()
app = gr.mount_gradio_app(app, frontend_interface, path="/gradio")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
