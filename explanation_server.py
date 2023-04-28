import spacy
from flask import Flask, request, jsonify
import pickle
from happytransformer import GENSettings
from transformers import pipeline
from spacy.lang.en.stop_words import STOP_WORDS
from heapq import nlargest
from happytransformer import GENSettings, HappyGeneration, GENTrainArgs

distill_bert_clsr_for_sentiment = pipeline("sentiment-analysis")
distill_bert_cnn_summarization = pipeline("summarization")
nlp = spacy.load('en_core_web_sm')

app = Flask(__name__)

#training cases for prompt
#this should be modified with more cases
training_case_prompts = """Keywords: supplier urbani, Truffles, VegetablesMushrooms
Output: Both items are belonged to supplier urbani. Both are Truffeles and they belonged to the category of VegetablesMushrooms.
###
Keywords: peppers, FoodProduceFresh, Grocery
Output: These two items are peppers. They are belonging to Grocery and FoodProduceFresh category.
###
Keywords: FruitsOranges, FoodProduceFresh, FruitsCitrus
Output: These items are simmilar because they are FruitsOranges. These are belonging to the category FruitsCitrus and FoodProduceFresh.
###
Keywords: cranberries, Fruits, dried, FoodProduceFresh
Output: Both items are cranberries. Both are dried Fruits. These will belong to FoodProduceFresh category.
###"""

# Summarize the reviews
def summarize_reviews(reviews):
    # Import the puctuations
    from string import punctuation

    # importing the stopwords
    stop_words = list(STOP_WORDS)

    doc = nlp(reviews)
    punctuation = punctuation + '\n'

    # word feq calculation
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stop_words:
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1

    # maximum freq value calculation
    max_frequency = max(word_frequencies.values())

    # normalized freq
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency

    # sentence tokenization
    sentence_tokens = [sent for sent in doc.sents]

    # sentence score calculation with tokens
    sentence_score = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_score.keys():
                    sentence_score[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_score[sent] += word_frequencies[word.text.lower()]

    # sentences percentage with max score current 30%
    percentage = 0.3
    select_length = int(len(sentence_tokens) * percentage)

    # summary by the nlargest
    summary = nlargest(select_length, sentence_score, key=sentence_score.get)

    final_summary = [word.text for word in summary]
    summary = ''.join(final_summary)
    return summary



def generate_prompt_string(training_case_prompt, keywords_for_generate):
  concated_keywords = ", ".join(keywords_for_generate)
  string_prompt = training_case_prompt + "\nKeywords: "+ concated_keywords + "\nOutput:"
  return string_prompt

meta_data_to_train = pickle.load(open('meta_data_to_train.pkl','rb'))
reviews_dataframe = pickle.load(open('reviews_dataframe.pkl','rb'))

@app.route('/api',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    initial_asin = data['initial_asin']
    rec_asin = data['rec_asin']

    initial_product = meta_data_to_train.loc[meta_data_to_train['asin'] == initial_asin]
    recommended_product = meta_data_to_train.loc[meta_data_to_train['asin'] == rec_asin]

    product1 = initial_product['bag_of_words'].to_string()
    product2 = recommended_product['bag_of_words'].to_string()

    common_feature_set = set(product1.split()).intersection(set(product2.split()))
    generation_neo_model = HappyGeneration("GPT-NEO", "EleutherAI/gpt-neo-1.3B")
    args = GENSettings(do_sample=True, top_k=100, max_length=100, min_length=10, no_repeat_ngram_size=3, num_beams=5,
                       early_stopping=True)

    prompt_with_inputs = generate_prompt_string(training_case_prompts, common_feature_set)

    generated_text = generation_neo_model.generate_text(prompt_with_inputs, args=args)

    rows_for_rev_categorization = reviews_dataframe.loc[reviews_dataframe['asin'] == rec_asin]
    positive_reviews = []
    negative_reviews = []

    for index, row in rows_for_rev_categorization.iterrows():
        sentiment = distill_bert_clsr_for_sentiment(row['reviewText'])
        if (sentiment[0]['label'] == 'NEGATIVE'):
            negative_reviews.append(row['reviewText'])
        else:
            positive_reviews.append(row['reviewText'])

    print(len(negative_reviews))
    print(len(positive_reviews))

    sum_negative_reviews = ""
    if (len(negative_reviews) > 5):
        combined_review = ''.join(negative_reviews)
        sum_neg_rev = summarize_reviews(combined_review)
        sum_negative_reviews = distill_bert_cnn_summarization(sum_neg_rev)[0]['summary_text']
    else:
        combined_review = ''.join(negative_reviews)
        sum_negative_reviews = distill_bert_cnn_summarization(combined_review)[0]['summary_text']

    sum_positive_reviews = ""
    if (len(positive_reviews) > 5):
        combined_review = ''.join(positive_reviews)
        sum_pos_rev = summarize_reviews(combined_review)
        sum_positive_reviews = distill_bert_cnn_summarization(sum_pos_rev)[0]['summary_text']
    else:
        combined_review = ''.join(positive_reviews)
        sum_positive_reviews = distill_bert_cnn_summarization(combined_review)[0]['summary_text']

    obj = {
        "compared_text" : generated_text,
        "positive_summary": sum_positive_reviews,
        "negative_summary": sum_negative_reviews
    }

    output = obj
    print(output)
    return jsonify(output)

@app.route('/api',methods=['GET'])
def status():
    return "up"


if __name__ == '__main__':
    app.run(port=5001, debug=True)