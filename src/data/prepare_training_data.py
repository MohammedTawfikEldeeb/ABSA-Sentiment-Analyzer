import pandas as pd
import ast
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger= logging.getLogger(__name__)

def safe_literal_eval(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError, TypeError):
        return []

def prepare_training_data(file_path):

    logger.info("Preparing training data")
    df = pd.read_csv(file_path)
    df['labels_list'] = df['llm_labels'].apply(safe_literal_eval)
    df_exploded = df.explode('labels_list').reset_index(drop=True)
    df_exploded['aspect_term'] = df_exploded['labels_list'].apply(lambda x: x.get('aspect_term') if isinstance(x, dict) else None)
    df_exploded['polarity'] = df_exploded['labels_list'].apply(lambda x: x.get('polarity') if isinstance(x, dict) else None)
    df_final = df_exploded.dropna(subset=['aspect_term', 'polarity'])
    df_final = df_final[['review_text', 'aspect_term', 'polarity']]

    return df_final


if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    logging.info("Starting data preparation process...")
    prepared_df = prepare_training_data(input_path)
    
    prepared_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logging.info(f"Training data saved successfully to {output_path}")