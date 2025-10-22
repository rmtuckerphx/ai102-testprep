# AI-102 Test Prep

## Azure AI Language
* Can submit 1 or more documents (1000 max) under 5120 chars

```py
from azure.ai.textanalytics import TextAnalyticsClient

client.detect_language(documents=[text])[0]
client.analyze_sentiment(documents=[text])[0]
client.extract_key_phrases(documents=[text])[0].key_phrases
client.recognize_entities(documents=[text])[0].entities
client.recognize_linked_entities(documents=[text])[0].entities
client.recognize_pii_entities(documents=[text])
client.begin_analyze_healthcare_entities(documents) #poller
client.begin_analyze_actions(documents, actions=[]) #poller
```

## Custom Text Classification
* Single label or Multiple label
* Higher the quality, clarity and variation of data set, more accurate the model
* False positive - model predicts x, but the file isn't labeled x.
* False negative - model doesn't predict label x, but the file in fact is labeled x.
* Recall - ratio of true positives to all that was labeled.
* Precision - ratio of true positives to all identified positives.
* F1 Score - A function of recall and precision, intended to provide a single score to maximize for a balance of each component
* Need a Storage Account

```py
from azure.ai.textanalytics import TextAnalyticsClient
client.begin_single_label_classify(batchedDocuments,project_name=project_name,deployment_name=deployment_name)
```

## Custom Named Entity Recognition (NER)
* High quality data has: diversity, distribution and accuracy
* Train with at least 10 files but no more than 100,000
* Consistency, precision and completeness

```py
from azure.ai.textanalytics import TextAnalyticsClient
client.begin_recognize_custom_entities(batchedDocuments,project_name=project_name,deployment_name=deployment_name)
```

## Custom Question Answering (CQA)
* Sources: web sites, files
* Add chit chat
* Improve knowledge base performance with active **learning** and **synonyms**
* Import Excel, TSV
* Create Language resource, enable CQA. A Search service and deployment will be created for you.
* Then create a project in Language Studio, add sources, test and deploy

```py
from azure.ai.language.questionanswering import QuestionAnsweringClient
client.get_answers(question=user_question,project_name=ai_project_name,deployment_name=ai_deployment_name)
```

## Conversation Language Understanding (CLU)
* Entities can be learned (trained), list (closed), prebuilt or regex

```py
from azure.ai.language.conversations import ConversationAnalysisClient
client.analyze_conversation()
```

## Text Translate
* Custom model use category id

```py
from azure.ai.translation.text import TextTranslationClient
client.get_supported_languages(scope="translation")
client.translate(body=input_text_elements, to_language=[targetLanguage])
```
