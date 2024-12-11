import pandas as pd
import re
import time
import os
import ollama
from tqdm import tqdm
from codecarbon import EmissionsTracker

model_name = 'llama3.2'

def label_and_reason_sentiment(content):
    full_prompt = f"""Assess the sentiment of the following text by identifying the presence of sentiment indicators such as emotional language, positive or negative expressions, and tone shifts.

If you find strong sentiment indicators, mark them accordingly and provide a reasoning for why the sentiment is positive, negative, or neutral.

Now, assess the following text:

Text: {content}

Sentiment Indicators Checklist:
- Emotional Language: Uses words that convey strong feelings, such as joy, anger, sadness, or excitement.
- Positive Expressions: Words or phrases that promote positive feelings, praise, or optimism.
- Negative Expressions: Words or phrases that convey criticism, pessimism, or negative attitudes.
- Tone Shifts: Noticeable changes in tone that affect how the content is perceived, potentially altering the sentiment.
- Balanced or Neutral Tone: Absence of strong emotional language, implying a more neutral or objective sentiment.

Provide three separate assessments with 'Positive', 'Negative', or 'Neutral' followed by one-line concise reasoning on why you chose that sentiment without further elaboration.

Response format required:
1. [Positive/Negative/Neutral] [Reasoning], 2. [Positive/Negative/Neutral] [Reasoning], 3. [Positive/Negative/Neutral] [Reasoning]"""

    try:
        response = ollama.chat(model=model_name, messages=[
            {'role': 'user', 'content': full_prompt}
        ])
        return response['message']['content']
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def convert_analysis_to_dataframe(row, analysis_str):
    results = row.to_dict()
    analysis_str = analysis_str.replace('\n', ' ').replace('\r', ' ').strip()
    # Pattern to extract labels and reasoning
    pattern = r'\d+\.\s*(Positive|Negative|Neutral)\s+(.*?)(?=(?:\d+\.\s*(?:Positive|Negative|Neutral)|$))'
    matches = re.findall(pattern, analysis_str, flags=re.IGNORECASE)
    label_count = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    reasonings = []
    for label, reasoning in matches:
        label_capitalized = label.capitalize()
        if label_capitalized in label_count:
            label_count[label_capitalized] += 1
            reasonings.append(f"{label_capitalized}: {reasoning.strip()}")
    # Assign the label that appears most frequently
    max_count = max(label_count.values())
    if max_count == 0:
        print(f"No label assigned for row {row.name}")
        return None
    # Get labels with max count
    labels_with_max_count = [label for label, count in label_count.items() if count == max_count]
    # If there's a tie, pick the first label
    assigned_label = labels_with_max_count[0]
    results["predicted"] = assigned_label
    results["reasoning"] = " | ".join(reasonings)
    return results

def process_csv(batch_size=10):
    # file_path = "data/dataset-sentiment.csv" # use this for full data
    file_path = "data/sample.csv" # Replace with your data CSV

    
    required_columns = ['text', 'label']
    
    try:
        # Read the CSV file and sample 20 random rows
        df = pd.read_csv(file_path)
        present_columns = [col for col in required_columns if col in df.columns]
        if len(present_columns) < 2:
            print(f"Warning: Required columns not found in {file_path}")
            return
        data = df[present_columns]
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return
    
    processed_file = f"{model_name}.csv"
    processed_texts = set()
    
    if os.path.exists(processed_file):
        processed_data = pd.read_csv(processed_file)
        processed_texts = set(processed_data['text'].values)
    
    print("Processed texts: ", len(processed_texts))
    data = data[~data['text'].isin(processed_texts)]
    
    results_list = []
    
    tracker = EmissionsTracker()
    tracker.start()
    
    try:
        for index, row in tqdm(data.iterrows(), total=len(data), desc='Processing'):
            content = row.get('text', '').replace('\n', ' ').replace('\r', ' ').strip()
            if content:
                start_time = time.time()
                analysis_result = label_and_reason_sentiment(content)
                if analysis_result:
                    result = convert_analysis_to_dataframe(row, analysis_result)
                    if result:
                        results_list.append(result)
                        # Print reasoning
                        print(f"Reasoning for row {index}: {result['reasoning']}")
                else:
                    print(f"Analysis failed for row {index}")
                elapsed_time = time.time() - start_time
                print(f"Processed row index: {index}, Time taken: {elapsed_time:.2f} seconds.")
    
            if len(results_list) >= batch_size:
                df_results = pd.DataFrame(results_list)
                columns_to_save = ['text', 'label', 'predicted', 'reasoning']
                df_results = df_results[columns_to_save]
                header = not os.path.exists(processed_file)
                df_results.to_csv(processed_file, mode='a', header=header, index=False)
                results_list.clear()
    except KeyboardInterrupt:
        print("\nProcess interrupted! Saving current progress...")
    finally:
        if results_list:
            df_results = pd.DataFrame(results_list)
            columns_to_save = ['text', 'label', 'predicted', 'reasoning']
            df_results = df_results[columns_to_save]
            header = not os.path.exists(processed_file)
            df_results.to_csv(processed_file, mode='a', header=header, index=False)
        emissions = tracker.stop()
        print(f"Final results saved to {processed_file}.")
        print(f"Estimated CO₂ emissions: {emissions:.6f} kg")

if __name__ == "__main__":
    process_csv()
