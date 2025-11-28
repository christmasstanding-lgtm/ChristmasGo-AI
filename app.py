import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_challenge(prompt):
    full_prompt = f"Crée un défi festif pour le jeu ChristmasGo sur le thème : {prompt}"
    result = generator(full_prompt, max_new_tokens=100, temperature=0.8, top_p=0.95)
    return result[0]['generated_text'].replace(full_prompt, "").strip()

interface = gr.Interface(
    fn=generate_challenge,
    inputs=gr.Textbox(placeholder="Ex: 'Rennes', 'Sapin', 'Chocolat chaud'", label="Thème du défi"),
    outputs=gr.Textbox(label="Défi généré"),
    title="🎄 Générateur de défis ChristmasGo",
    description="Entrez un thème festif et recevez un défi original pour le jeu ChristmasGo.",
    theme="soft",
    examples=[["Pôle Nord"], ["Bonhomme de neige"], ["Traîneau magique"]]
)

if __name__ == "__main__":
    interface.launch()
