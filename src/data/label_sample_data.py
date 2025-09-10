import re
import pandas as pd
import google.generativeai as genai
import json
from tqdm import tqdm
from src import config
import os



def clean_for_llm(text: str) -> str:
    """
    Cleans text minimally for input to a Large Language Model.
    """
    if not isinstance(text, str):
        return ""


    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\n+', '\n', text)
    text = text.strip()
    
    return text

def get_llm_labels(review_text: str, model) -> list:
    """
    Sends a single review to the Gemini API and returns the structured labels.
    """

    prompt_template = """
# ROLE
You are an expert sentiment analyst specializing in customer feedback for the InstaPay mobile application in Egypt.
# TASK
Your task is to carefully read the following customer review and extract all the distinct aspects the user is talking about. For each aspect, you must determine the sentiment polarity.
# RULES
1. Extract only aspects explicitly mentioned. Do not invent aspects.
2. Keep the `aspect_term` concise and standardized in Arabic (e.g., "أداء التطبيق", "خدمة العملاء", "الرسوم").
3. The `polarity` must be one of these three exact values: "positive", "negative", "neutral".
4. If a review does not contain any clear aspects, return an empty list `[]`.
# OUTPUT FORMAT
Provide the output ONLY in a valid JSON format, as a list of objects. Do not write any explanations or text outside the JSON structure.
# CUSTOMER REVIEW
Here is the review you need to analyze:
"{review_text}"
"""
    try:
        prompt = prompt_template.format(review_text=review_text)
        response = model.generate_content(prompt)
        
   
        json_text = response.text.strip().replace('```json', '').replace('```', '').strip()
        
        return json.loads(json_text)
    except Exception as e:
        print(f"An error occurred for a review. Error: {e}. Returning empty list.")
        return []



if __name__ == '__main__':

    print("Setting up Gemini API...")

    genai.configure(api_key=config.GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')

    # --- 2. تحميل البيانات وتنظيفها ---
    print("Loading and cleaning data...")
    DATA_PATH = config.EXTRACTED_DATA_DIR / 'instapay_reviews.csv'
    LABELED_OUTPUT_PATH = config.PROCESSED_DATA_DIR / 'llm_labeled_reviews.csv'
    
    df = pd.read_csv(DATA_PATH)
    

    df_sample = df.sample(n=700, random_state=42)
    
    df_sample['cleaned_review_text'] = df_sample['review_text'].apply(clean_for_llm)
    

    df_cleaned = df_sample.dropna(subset=['cleaned_review_text'])
    df_cleaned = df_cleaned[df_cleaned['cleaned_review_text'] != '']
    
    print(f"Successfully cleaned {len(df_cleaned)} reviews. Starting labeling process...")
    

    tqdm.pandas(desc="Labeling Reviews with Gemini")
    

    df_cleaned['llm_labels'] = df_cleaned['cleaned_review_text'].progress_apply(
        lambda text: get_llm_labels(text, model)
    )
    

    df_cleaned.to_csv(LABELED_OUTPUT_PATH, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ Done! Labeled data saved to: {LABELED_OUTPUT_PATH}")
    print("\n--- Sample of the final output ---")
    print(df_cleaned[['cleaned_review_text', 'llm_labels']].head())