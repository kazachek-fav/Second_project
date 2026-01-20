from flask import Flask, render_template, request
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Инициализация пайплайнов
pipe = pipeline("text-classification", model="blanchefort/rubert-base-cased-sentiment")
sentiment_analyzer = pipeline("text-generation", model="ai-forever/rugpt3medium_based_on_gpt2")
tokenizer = AutoTokenizer.from_pretrained("ai-forever/rugpt3medium_based_on_gpt2")
model = AutoModelForCausalLM.from_pretrained("ai-forever/rugpt3medium_based_on_gpt2")

app = Flask(__name__)

def generate_recommendation(mood):
    prompt = (f"Посоветуй один популярный мультфильм для человека, с настроением {mood}. "
              f'Назови только один фильм и объясни кратко почему ты его выбрал!')
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=1200     ,
        do_sample=True,
        top_p=0.98,
        temperature=0.87
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Обрезка по длине подсказки
    return text[len(prompt):].strip()

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendation = ""
    user_text = ""

    if request.method == 'POST':
        user_text = request.form['message']

        result = pipe(user_text)[0]
        label = result['label']

        if label == 'POSITIVE':
            mood = 'весёлое'
        elif label == 'NEGATIVE':
            mood = 'ужасное'
        else:
            mood = 'нормальное'
        
        ai_text = generate_recommendation(mood)
        recommendation = f'Настроение: {mood}, <br>Рекомендация: {ai_text}'

    return render_template("index.html", recommendation=recommendation, user_text=user_text)

if __name__ == "__main__":
    app.run(debug=True, port=5001)