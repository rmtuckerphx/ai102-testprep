# AI-102 Test Prep

## Azure AI Service
* Allows you to access multiple cognitive services through a single endpoint and credential

## Azure AI Language
* Can submit 1 or more documents (1000 max) under 5120 chars

```py
from azure.ai.textanalytics import TextAnalyticsClient

client.detect_language(documents=[text])[0]
client.analyze_sentiment(documents=[text])[0]
client.extract_key_phrases(documents=[text])[0].key_phrases # returns nouns, no confidence score
client.recognize_entities(documents=[text])[0].entities # returns entities with confidence score
client.recognize_linked_entities(documents=[text])[0].entities
client.recognize_pii_entities(documents=[text]) # returns redacted text
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
* Entities can be:
	*  learned (trained)
 	*  list (closed)
  	*  prebuilt - builtin recognizers
  	*  regex
  	*  Pattern.any - from LUIS; used for long entities (ex: book titles, airport names)

```py
from azure.ai.language.conversations import ConversationAnalysisClient
client.analyze_conversation()
```

## Text Translation
* Custom model use category id
* Glossary file is associated with the source, not the target
* REST endpoint
	* Global - api.congnitive.microsofttranslator.com
	* Americas - api-nam.congnitive.microsofttranslator.com
	* /translate?from=en&to=es&to=de

```py
from azure.ai.translation.text import TextTranslationClient
client.get_supported_languages(scope="translation")
client.translate(body=input_text_elements, to_language=[targetLanguage])
```

## Speech-to-Text (STT)
* Needs location (eastus) and key

```py
import azure.cognitiveservices.speech as speech_sdk

speech_recognizer = speech_sdk.SpeechRecognizer(speech_config, audio_config)
speech = speech_recognizer.recognize_once_async().get()
output= speech.text
```

## Text-to-Speech (TTS)
* Needs location (eastus) and key
* FromWavFileOutput - location of output audio file
* Speech Synthesis Markup Language (SSML)
	* name - voice used
 	* effect - eq_car, eq_telecomhp8k

```py
import azure.cognitiveservices.speech as speech_sdk

speech_synthesizer = speech_sdk.SpeechSynthesizer(speech_config, audio_config,)
speak = speech_synthesizer.speak_text_async(response_text).get()
speak = speech_synthesizer.speak_ssml_async(responseSsml).get()
```
SSML
```
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
    <voice name="en-US-AvaMultilingualNeural" effect="eq_car">
        Good morning!
        <mstts:express-as style="cheerful" styledegree="2">
            That'd be just amazing!
        </mstts:express-as>
    </voice>
</speak>
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
* Better for unstructured data
* Provides broader multimodal analysis and advanced reasoning
* Gen-AI service
* Create analyzer trained on content schema you define
* Schema - content sample or analyzer template
* Hub in AI Foundry
* Templates: invoice analysis, image analysis, speech transcript analysis, Video analysis
	

## Azure AI Document Intelligence
* Previously called FormRecognizer
* Extracts fields, text and data from documents and forms
* Best for structured document extraction
* Requires Azure Storage account
* More general models
   - Read (prebuilt-read) - extract text and languages, QR codes
   - Document (prebuilt-document) - extract text, keys, values, entities and selection marks
   - Layout (prebuilt-layout) - extract text and structure information
* Prebuilt models:
   - Invoice (prebuilt-invoice)
   - Bank Statement (prebuilt-bankStatement.us)
   - Check (prebuilt-check.us)
   - Contract (prebuilt-contract)
   - Credit Card (prebuilt-creditCard)
   - Receipt (prebuilt-receipt)
   - US Tax (prebuilt-tax.us.1040, .1095A, .1098, .1099A, .w2)
   - ID document (prebuilt-idDocument) - driver's license, passport
   - Business card (prebuilt-businessCard)
   - Health insurance card (prebuilt-healthInsuranceCard.us)
   - Marriage certificate (prebuilt-marriageCertificate.us)
   - Mortgage docs (prebuilt-mortgage.us.1003, prebuilt-mortgage.us.closingDisclosure)
   - Pay stub (prebuilt-payStub.us)
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
   - example: hand-written surveys

```py
# OLD
from azure.ai.formrecognizer import DocumentAnalysisClient
document_analysis_client = DocumentAnalysisClient(
	endpoint=endpoint, credential=AzureKeyCredential(key)
)
poller = client.begin_analyze_document_from_url(
     "prebuilt-layout", fileUrl, locale=fileLocale
)
```

```py
# NEW
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

document_intelligence_client = DocumentIntelligenceClient(
	endpoint=endpoint, credential=AzureKeyCredential(key)
)

poller = document_intelligence_client.begin_analyze_document(
	"prebuilt-layout", AnalyzeDocumentRequest(url_source=formUrl)
)
result = poller.result()

# result.pages
# page.lines
# page.selection_marks
# page.words
# result.tables
# table.cells
```

ARM Template:
```json
{
	"resources": [
		{
			"type": "Microsoft.CongnitiveServices/accounts",
			"apiVersion": "2023-05-01",
			"name": "DocumentIntelligenceDemo",
			"location": "eastus",
			"sku": {
				"name": "F0"
			},
			"kind": "FormRecongnizer"
		}
	]
}
```


## Azure AI Search
* Indexer used to build a search index
* Knowledge store built at same time
* Add replicas to increase query throughput or move to higher service tier
* Customer managed key (CMK) - security and encryption at rest
* Enrichment pipeline builds a document that includes content from the original source
* Skills used in pipeline to process content
* Skills grouped into Skillsets
* Indexer uses final document with implicit and explicit mapping to index the documents
* Built-in skills
	* detect language
 	* extract entities
  	* extract key phrases
  	* translate text
  	* remove PII
  	* extract text from images
* Custom skills - API call or Azure Function
* Fields in an index can be set as:
	* key
 	* searchable
  	* filterable
  	* sortable
  	* facetable
  	* retrievable
* Typeahead includes autocomplete box and suggestion list
	* add suggester (with fields) to search index definition
 	* set analyzer property
 	* call API endpoint for suggester or autocomplete
* Knowledge score in a skillset consists of projections of the enriched data, JSON, tables or image files
* Sample pipeline
	* Source - Azure Blob Storage
 	* Cracking - Vision API (OCR)
 	* Preparation - Translator API
	* Destination - Azure Blob Storage
* Projections
	* table projection - useful for analytics and Microsoft Power BI, Azure Table Storage
	* object projection - JSON, Azure Blob Storage
	* file projection - binary, Azure Blob Storage
* replace key
 	* add new query key
  	* change the app to use the new key
   	* delete the old key
	
## Azure AI Vision
* Authentication: Microsoft Entra (keyless; token-based) and API key
* JPEG, PNG, GIF, or BMP format
* For PDF, Office and HTML documents, use Document Intelligence
* Less than 4MB
* Greater than 50x50px
* VisualFeatures:
	* CAPTION - sentence that describes the image contents
	* READ (OCR) - extract printed or hand-written text
	* DENSE_CAPTIONS - one sentence for up to 10 regions in image
	* TAGS - tags for objects, living beings, scenery and actions
	* OBJECTS - object detection
	* SMART_CROPS - find region for thumbnail w/ priority for faces
	* PEOPLE - detect people
	
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

result = client.analyze_from_url(
    image_url="https://aka.ms/azsdk/image-analysis/sample.jpg",
    visual_features=[VisualFeatures.CAPTION],
    gender_neutral_caption=True,  # Optional (default is False)
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

## AI Custom Vision - General
* These things apply to both Image Classification and Object Detection
* Needs 2 resources: Custom Vision training and Custom Vision prediction
* Steps to Create
	* Open Computer Vision portal
 	* Create/open a project
 	* Upload sample images
  	* Tag/label the images (Object Detect, Tag/label the regions)
  	* Train the model
  	* Test the model
  	* Publish the model

## AI Custom Vision - Image Classification
* Tag images
* Compact models for edge/mobile deployment
* Domains
	* General
	* General [A1] - large dataset, more training and inference time
	* General [A2] - shorter training time, better inference speed
	* Food
	* Landmarks
	* Retail
	* Adult
	* General (compact) - special postprocessing logic
	* General (compact) [S1] - no postprocessing logic
	* Landmarks (compact)
	* Retail (compact)
* Classification Types
	* Multi-label (multiple tags per images)
	* Multi-class (single tag per image) 

```py	
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
prediction_client = CustomVisionPredictionClient(endpoint="<YOUR_PREDICTION_RESOURCE_ENDPOINT>", credentials=credentials)
results = prediction_client.classify_image("<YOUR_PROJECT_ID>",
                                           "<YOUR_PUBLISHED_MODEL_NAME>",
                                           image_data)                                                 
```

## AI Custom Vision - Object Detection
* Compact models for edge/mobile deployment
* Need at least 15 images
* Domains
	* General
	* General [A1] - better accuracy
	* Logo
	* Products on Shelves
	* General (compact) - special postprocessing logic
	* General (compact) [S1] - no postprocessing logic
 	
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

## Azure AI Immersive Reader
* improve reading comprehension
* new readers, language learners, dyslexia, etc.
* features
	* isolate content
 	* display pictures for common words
  	* highlight parts of speech
  	* read content aloud
  	* translate content in real-time
  	* split words into syllables
   
## Content Safety
* API that can be integrated into any application or service
* Moderate Text Content - run tests on text
* Moderate Image Content - run tests on images
* Monitor Online Activity - real-time monitoring
* Prompt Shields - protects against prompt injection and jailbreak attacks
* Protected Material Detection - identifies copyrighted content in text and code
* Groundedness Detection - identifies hallucinated or ungrounded AI outputs

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
* Roles
	* Cognitive Services OpenAI Contributor - upload datasets, fine-tune models, create/update model deployments
 	* Cognitive Services OpenAI User - use the model
* Grounding (list in order of least to most effort)
	* one-shot/few-shot examples to system/user prompt
 	* grounding content - Azure Blog Storage, Azure AI Search (RAG)
	* tools
```
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url="https://YOUR-RESOURCE-NAME.openai.azure.com/openai/v1/",
)

messages = [
    {"role": "system", "content": "You are a helpful assistant that explains complex topics simply."},
    {"role": "user", "content": "Explain machine learning to a 5-year-old."}
]

completion = client.chat.completions.create(
	model="gpt-3.5-turbo",  # Specify the model to use (e.g., "gpt-4o", "gpt-3.5-turbo")
	messages=messages,
	max_tokens=100,  # Limit the length of the generated response
	temperature=0.7  # Control the randomness of the output (0.0 for deterministic, 1.0 for very creative)
)
print(completion.choices[0].message.content)
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

## Docker / Container
* Docker Image comes from Microsoft Container Registry
* Deploy container image to host computer
	* Export package file
  	* Move package to Docker input directory
  	* Run the container
* Docker run
	* Container image name starts with "mcr". ex: `mcr.microsoft.com/azure-cognitive-services/textanalytics/sentiment`
 	* **Billing** - AI Services endpoint URI
  	* **ApiKey** - key

## Azure DevOps / Azure CLI
* Identify Azure AI Services account - `az congnitiveservices account show`
