from flask import Flask, request, render_template, jsonify
from langchain_huggingface import HuggingFaceEndpoint
from langchain import PromptTemplate, LLMChain
import os

app = Flask(__name__)

# Set Hugging Face API token from environment or Google Colab userdata
sec_key = os.getenv("HUGGINGFACEHUB_API_TOKEN", "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")

# Initialize Hugging Face model endpoint
repo_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7, huggingfacehub_api_token=sec_key)

def suggest_restaurant(cuisine):
    """
    Suggests a restaurant based on the given cuisine.
    
    Args:
    cuisine (str): The type of cuisine desired.
    
    Returns:
    str: A restaurant suggestion with popular menu items.
    """
    prompt_template = PromptTemplate(
        template="Suggest a restaurant that serves {cuisine} cuisine and provide some popular menu items.",
        input_variables=["cuisine"]
    )

    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run({"cuisine": cuisine})
    return response

# Flask routes
@app.route("/", methods=["GET", "POST"])
def home():
    suggestion = None
    if request.method == "POST":
        cuisine = request.form.get("cuisine")
        if cuisine:
            suggestion = suggest_restaurant(cuisine)
    return render_template("index.html", suggestion=suggestion)

if __name__ == "__main__":
    app.run(debug=True)
