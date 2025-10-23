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

## Text Translation
* Custom model use category id
* REST endpoint
	* Global - api.congnitive.microsofttranslator.com
	* Americas - api-nam.congnitive.microsofttranslator.com
	* /translate?from=en&to=es&to=de

```py
from azure.ai.translation.text import TextTranslationClient
client.get_supported_languages(scope="translation")
client.translate(body=input_text_elements, to_language=[targetLanguage])
```

## Speech-to-Test (STT)
* Needs location (eastus) and key

```py
import azure.cognitiveservices.speech as speech_sdk

speech_recognizer = speech_sdk.SpeechRecognizer(speech_config, audio_config)
speech = speech_recognizer.recognize_once_async().get()
output= speech.text
```

## Text-to-Speech (TTS)
* Needs location (eastus) and key

```py
import azure.cognitiveservices.speech as speech_sdk

speech_synthesizer = speech_sdk.SpeechSynthesizer(speech_config, audio_config,)
speak = speech_synthesizer.speak_text_async(response_text).get()
speak = speech_synthesizer.speak_ssml_async(responseSsml).get()
```

## Speech Translation
* Needs location (eastus) and key
* Event-based synthesis - audio stream from 1 source to 1 target language; TranslationRecognizer 
* Manual synthesis - from 1 source to many target languages; TranslationRecognizer to SpeechSynthesizer

```py
import azure.cognitiveservices.speech as speech_sdk

translation_config = speech_sdk.translation.SpeechTranslationConfig(subscription=subscription_key, region=service_region)

translation_config.speech_recognition_language = "en-GB"
translation_config.add_target_language("de")
translation_config.add_target_language("fr")

audio_config_in =  speech_sdk.audio.AudioConfig(use_default_microphone=True)

translator = speech_sdk.translation.TranslationRecognizer(translation_config, audio_config = audio_config_in)
translator.recognize_once_async().get()
```

## Azure AI Voice Live agent
* Real-time, bidirectional communication
* Uses WebSocket connections
* Supports: STT, TTS, avatar streaming, audio processing
* Authentication: Microsoft Entra (keyless; token-based) and API key
* Requires Cognitive Services User role
* Async only
* WebRTC - for avatar streaming


## Azure AI Content Understanding
* Documents and forms, images, videos and audio recordings
* Gen-AI service
* Create analyzer trained on content schema you define
* Schema - content sample or analyzer template
* Hub in AI Foundry
* Templates: invoice analysis, image analysis, speech transcript analysis, Video analysis
	

## Document Intelligence
* Prebuild models:
   - Invoice
   - Receipt
   - US Tax (W-2, 1098, 1099, 1040)
   - ID document (driver's license, passport)
   - Business card
   - Health insurance card
   - Marriage certificate
   - Mortgage docs
* More general models
   - Read - extract text and languages
   - General document - extract text, keys, values, entities and selection marks
   - Layout - extract text and structure information
* Document types: JPEG, PNG, BMP, TIFF or PDF
* Read model support Microsoft Office files
* 500MB max file size
* First 2000 pages analyzed
* Custom model needs ocr.json, fields,json and labels.json; 5-6 sample forms
* Custom template model
   - training takes a few minutes; 100+ languages supported
   - Template, form, structured
* Custom neural model
   - deep learning; includes layout
   - Structured, semi-structured, unstructured

```py
from azure.ai.formrecognizer import DocumentAnalysisClient
poller = client.begin_analyze_document_from_url(
     fileModelId, fileUri, locale=fileLocale
)
```

## Azure AI Search
* Indexer used to build a search index
* Knowledge store built at same time
* Enrichment pipeline builds a document that includes content from the original source
* Skills used in pipeline to process content
* Skills grouped into Skillsets
* Indexer uses final document with implicit and explicit mapping to index the documents
* Built-in skills - detect language, extract entities, extract key phrases, translate text, remove PII, extract text from images, etc.
* Custom skills - API call or Azure Function
* Fields in an index can be set as: key, searchable, filterable, sortable, facetable, retrievable
* Knowledge score in a skillset consists of projections of the enriched data, JSON, tables or image files
* Sample pipeline
	* Source - Azure Blob Storage
 	* Cracking - Vision API (OCR)
 	* Preparation - Translator API
	* Destination - Azure Blob Storage
 * table projection - useful for analytics and Microsoft Power BI, Azure Table Storage
 * object projection - JSON, Azure Blob Storage
 * file projection - binary, Azure Blob Storage
	
## Azure AI Vision
* Authentication: Microsoft Entra (keyless; token-based) and API key
* JPEG, PNG, GIF, or BMP format
* Less than 4MB
* Greater than 50x50px
* VisualFeatures: TAGS, OBJECTS, CAPTION, DENSE_CAPTIONS, PEOPLE, SMART_CROPS, READ
	
```py
from azure.ai.vision.imageanalysis import ImageAnalysisClient
client = ImageAnalysisClient(
    endpoint="<YOUR_RESOURCE_ENDPOINT>",
    credential=AzureKeyCredential("<YOUR_AUTHORIZATION_KEY>")
)
result = client.analyze(
    image_data=<IMAGE_DATA_BYTES>, # Binary data from your image file
    visual_features=[VisualFeatures.CAPTION, VisualFeatures.TAGS],
    gender_neutral_caption=True,
)
```

## Face Detection
* Face detection - bounding box
* Face attributes - head pose, glasses, mask, blur, exposure, noise, occlusion, accessories, quality
* Facial landmark location - features such as eye corners, pupils, tip of nose, etc.
* Face comparison - compare faces across multiple images for similarity and verification
* Facial recognition - train model on collection of faces and identify those people in new images
* Facial liveness - video fake or live
* Face detection model:
  * DETECTION01 (default)
  * DETECTION02 (improved for small, blurry, side)
  * DETECTION03 (most accurate for small or rotated; advanced)
* Face recognition model:
  * RECOGNITION01 (legacy)
  * RECOGNITION02 (improved)
  * RECOGNITION03 (high accuracy)
  * RECOGNITION04 (most accurate; production)
* Face attribute features: HEAD_POSE, OCCLUSION, ACCESSORIES, etc.

```py
from azure.ai.vision.face import FaceClient
face_client = FaceClient(
    endpoint="<YOUR_RESOURCE_ENDPOINT>",
    credential=AzureKeyCredential("<YOUR_RESOURCE_KEY>"))
result = face_client.detect(
        image_content=image_data.read(),
        detection_model=FaceDetectionModel.DETECTION01,
        recognition_model=FaceRecognitionModel.RECOGNITION01,
        return_face_id=True,
        return_face_attributes=features,
    )
```

## AI Custom Vision - Image Classification
* Needs 2 resources: Custom Vision training and Custom Vision prediction

```py	
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
prediction_client = CustomVisionPredictionClient(endpoint="<YOUR_PREDICTION_RESOURCE_ENDPOINT>", credentials=credentials)
results = prediction_client.classify_image("<YOUR_PROJECT_ID>",
                                           "<YOUR_PUBLISHED_MODEL_NAME>",
                                           image_data)                                                 
```

## AI Custom Vision - Object Detection
* Needs 2 resources: Custom Vision training and Custom Vision prediction
	
```py
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
prediction_client = CustomVisionPredictionClient(endpoint="<YOUR_PREDICTION_RESOURCE_ENDPOINT>", credentials=credentials)
results = prediction_client.detect_image("<YOUR_PROJECT_ID>",
                                           "<YOUR_PUBLISHED_MODEL_NAME>",
                                           image_data)
```

## Azure Video Indexer
* Extract insights from video: face identification, text recognition, object labels, scene segmentation, etc.
* Create custom models: people, language, brand
* Website
* Use a download link to share a large file from OneDrive

## Azure AI Anomaly Detector
* Detects anomalies in time series data
* Uses ML to identify unexpected patterns
* Great for IoT sensor data
* Univariate - changes in one variable
* Multivariate - changes in multiple variables with correlations

## Extracting Text
* audio - Azure AI Speech
* PDF, TIFF, JPG, BMP - Azure AI Document Intelligence
* PNG, JPG, GIF, BMP - Azure AI Vision (OCR)
* video - Azure AI Vision

## Azure OpenAI
* Needs endpoint, key and deployment name
* Each model deployment has a unique deployment name
* Access multiple deployments from the same base endpoint
* Temperature - how likely model will pick top choice
	* 0.0 = more deterministic
 	* 1.1 = more random, creative
* Top-p - affects scope of choice for next word
	* 1.0 = all words considered
 	* 0.5 = Only most probable words that together make up 50% of the probability mass are considered
* Max response tokens - upper bound for output tokens
* prompt_tokens - input
* completion_tokens - output
* Subscription charged for both input and output tokens
* Add content filter to remove hate speech and more

```
openai.ChatCompletion.create
response.choice[0].text
```

## Private endpoint
* Add private endpoint to resource (ex: Azure Storage, Azure AI Language, etc.)
* Update virtual network rules on resource to turn off internet traffic
* In VNet, add service endpoint

## Connecting to Resources in Azure AI Foundry

```py
from azure.ai.projects import AIProjectClient

project_endpoint = os.getenv("OPENAI_ENDPOINT")
project_client = AIProjectClient(            
	credential=DefaultAzureCredential(),
    endpoint=project_endpoint)
```

## Open AI Chat Client

```py
import os
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from openai import AzureOpenAI

# Only run if this file is executed directly
def execute_chat():
    print("Chat module executed directly.")
    try:
        # connect to the project
        project_endpoint = os.getenv("OPENAI_ENDPOINT")
        if not project_endpoint:
            raise ValueError("OPENAI_ENDPOINT environment variable is not set")
        
        project_client = AIProjectClient(            
                credential=DefaultAzureCredential(),
                endpoint=project_endpoint,
            )
        
        # Get a chat client
        chat_client = project_client.get_openai_client(api_version="2024-10-21")
        
        # Get a chat completion based on a user-provided prompt
        user_prompt = input("Enter a question:")
        
        response = chat_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": user_prompt}
            ]
        )
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    execute_chat()
```
