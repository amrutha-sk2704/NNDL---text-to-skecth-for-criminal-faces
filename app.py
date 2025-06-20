import gradio as gr
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from huggingface_hub import snapshot_download
from PIL import Image
import torch
import numpy as np
import cv2
import os
import insightface
from insightface.app import FaceAnalysis

# ==== SETUP ====
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load SD15 + LoRA for sketch generation
model_id = "runwayml/stable-diffusion-v1-5"
model_dir = snapshot_download(
    repo_id=model_id,
    local_dir="./sd15_cache",
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=1
)

pipe_sketch = StableDiffusionPipeline.from_pretrained(
    model_dir,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    safety_checker=None,
    use_safetensors=True
)
pipe_sketch.unet.load_attn_procs("./sketch_lora_model")
pipe_sketch = pipe_sketch.to(device)

# Load ControlNet pipeline
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-scribble",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
pipe_color = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

# Load Face Recognition
app_face = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app_face.prepare(ctx_id=0)
KNOWN_DIR = "criminal_dataset"
TOP_K = 3

# ==== FUNCTIONS ====

def generate_sketch(prompt):
    image = pipe_sketch(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    image.save("sketches/generated_sketch_4.png")
    return image

def sketch_to_face(sketch_img, prompt2):
    sketch = cv2.cvtColor(np.array(sketch_img), cv2.COLOR_RGB2GRAY)
    sketch = cv2.resize(sketch, (512, 512))
    sketch = cv2.bitwise_not(sketch)
    sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
    sketch_image = Image.fromarray(sketch)

    output = pipe_color(prompt2, image=sketch_image, num_inference_steps=30, guidance_scale=8.5).images[0]
    output.save("colored_face_4.png")
    return output

def recognize_face(test_img):
    img_bgr = cv2.cvtColor(np.array(test_img), cv2.COLOR_RGB2BGR)
    test_faces = app_face.get(img_bgr)
    if not test_faces:
        return "❌ No face detected", []

    test_embedding = test_faces[0].embedding
    results = []

    for filename in os.listdir(KNOWN_DIR):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        path = os.path.join(KNOWN_DIR, filename)
        img = cv2.imread(path)
        if img is None:
            continue
        faces = app_face.get(img)
        if not faces:
            continue
        sim = float(test_embedding @ faces[0].embedding.T)
        results.append((filename, sim, img))

    results.sort(key=lambda x: x[1], reverse=True)
    top = results[:TOP_K]
    result_text = "\n".join([f"{i+1}. {f} - Score: {s:.2f}" for i, (f, s, _) in enumerate(top)])
    top_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for _, _, img in top]
    return result_text, top_imgs

# ==== UI ====

with gr.Blocks() as demo:
    gr.Markdown("## 🧠 AI Criminal Sketch Matching System")

    with gr.Row():
        prompt1 = gr.Textbox(label="📝 Step 1: Enter prompt for sketch", value="A sketch of an elderly man with a bald head and white beard...")
        sketch_btn = gr.Button("🎨 Generate Sketch")
        sketch_img = gr.Image(label="Generated Sketch")

    with gr.Row():
        reload_btn = gr.Button("♻️ Reload Sketch")
    
    with gr.Row():
        prompt2 = gr.Textbox(label="🎭 Step 2: Enter prompt for realistic face", value="A realistic photo of an elderly bald man with a long white beard...")
        face_btn = gr.Button("🧬 Generate Real Face")
        face_img = gr.Image(label="Realistic Face Output")

    with gr.Row():
        recog_btn = gr.Button("🔍 Run Face Recognition")
        recog_text = gr.Textbox(label="Top 3 Matches")
        recog_gallery = gr.Gallery(label="Matching Faces", columns=3)

    sketch_btn.click(generate_sketch, inputs=prompt1, outputs=sketch_img)
    reload_btn.click(generate_sketch, inputs=prompt1, outputs=sketch_img)
    face_btn.click(sketch_to_face, inputs=[sketch_img, prompt2], outputs=face_img)
    recog_btn.click(recognize_face, inputs=face_img, outputs=[recog_text, recog_gallery])

demo.launch()